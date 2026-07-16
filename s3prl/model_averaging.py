import os
import yaml
import glob
import torch
import random
import argparse
import logging
import torchaudio
import numpy as np
from argparse import Namespace
from torch.distributed import is_initialized, get_world_size

from s3prl import hub
from s3prl.utility.helper import backup, get_time_tag, hack_isinstance, is_leader_process, override
from s3prl.utility.helper import is_leader_process, get_model_state, show, defaultdict
from torch.nn.parallel import DistributedDataParallel as DDP

from huggingface_hub import HfApi, HfFolder

class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces


class Runner():
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, logging, checkpoint saving
    """
    def __init__(self, args, config):
        torchaudio.set_audio_backend('soundfile')
        self.args = args
        self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu') if self.args.init_ckpt else {}
        self.upstream = self._get_upstream()

    def _load_weight(self, model, name):
        init_weight = self.init_ckpt.get(name)
        if init_weight:
            show(f'[Runner] - Loading {name} weights from the previous experiment')
            model.load_state_dict(init_weight)


    def _init_model(self, model, name, trainable, interfaces=None):
        for interface in interfaces or []:
            assert hasattr(model, interface), interface

        self._load_weight(model, name)

        if is_initialized() and trainable and any((p.requires_grad for p in model.parameters())):
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
            for interface in interfaces or []:
                setattr(model, interface, getattr(model.module, interface))

        return ModelEntry(model, name, trainable, interfaces)


    def _get_upstream(self):
        Upstream = getattr(hub, self.args.upstream)
        ckpt_path = self.args.upstream_ckpt
        upstream_refresh = self.args.upstream_refresh

        if is_initialized() and get_rank() > 0:
            torch.distributed.barrier()
            upstream_refresh = False

        model = Upstream(
            ckpt = ckpt_path,
            model_config = self.args.upstream_model_config,
            refresh = upstream_refresh,
        ).to(self.args.device)

        if is_initialized() and get_rank() == 0:
            torch.distributed.barrier()

        return self._init_model(
            model = model,
            name = 'Upstream',
            trainable = self.args.upstream_trainable,
            interfaces = ["get_downsample_rates"]
        )


    def _get_featurizer(self):
        model = Featurizer(
            upstream = self.upstream.model,
            feature_selection = self.args.upstream_feature_selection,
            layer_selection = self.args.upstream_layer_selection,
            upstream_device = self.args.device,
            normalize = self.args.upstream_feature_normalize,
            fixed_length = self.args.fix_feature_len,
            ignore_length_dif = self.args.ignore_length_dif
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Featurizer',
            trainable = True,
            interfaces = ['output_dim', 'downsample_rate']
        )


def get_downstream_args():
    parser = argparse.ArgumentParser()

    # train or test for this experiment

    # distributed training
    parser.add_argument('--backend', default='nccl', help='The backend for distributed training')
    parser.add_argument('--local_rank', type=int,
                        help=f'The GPU id this process should use while distributed training. \
                               None when not launched by torch.distributed.launch')

    # use a ckpt as the experiment initialization
    # if set, all the args and config below this line will be overwrited by the ckpt
    # if a directory is specified, the latest ckpt will be used by default

    # only load the parameters in the checkpoint without overwriting arguments and config, this is for evaluation
    parser.add_argument('-i', '--init_ckpt', metavar='CKPT_PATH', help='Load the checkpoint for evaluation')

    # configuration for the experiment, including runner and downstream
    parser.add_argument('-c', '--config', help='The yaml file for configuring the whole experiment except the upstream model')

    # downstream settings
    parser.add_argument('-d', '--downstream', help='\
        Typically downstream dataset need manual preparation.\
        Please check downstream/README.md for details'
    )

    # upstream settings
    parser.add_argument('--hub', default="torch", choices=["torch", "huggingface"],
        help='The model Hub used to retrieve the upstream model.')

    upstreams = [attr for attr in dir(hub) if attr[0] != '_']
    parser.add_argument('-u', '--upstream',  help=""
        'Upstreams with \"_local\" or \"_url\" postfix need local ckpt (-k) or config file (-g). '
        'Other upstreams download two files on-the-fly and cache them, so just -u is enough and -k/-g are not needed. '
        'Please check upstream/README.md for details. '
        f"Available options in S3PRL: {upstreams}. "
    )
    parser.add_argument('-k', '--upstream_ckpt', metavar='{PATH,URL,GOOGLE_DRIVE_ID}', help='Only set when the specified upstream need it')
    parser.add_argument('-g', '--upstream_model_config', help='The config file for constructing the pretrained model')
    parser.add_argument('-r', '--upstream_refresh', action='store_true', help='Re-download cached ckpts for on-the-fly upstream variants')
    parser.add_argument('-f', '--upstream_trainable', action='store_true', help='Fine-tune, set upstream.train(). Default is upstream.eval()')
    parser.add_argument('-s', '--upstream_feature_selection', default='hidden_states', help='Specify the layer to be extracted as the representation')
    parser.add_argument('-l', '--upstream_layer_selection', type=int, help='Select a specific layer for the features selected by -s')
    parser.add_argument('--upstream_feature_normalize', action='store_true', help='Specify whether to normalize hidden features before weighted sum')
    parser.add_argument('--upstream_model_name', default="model.pt", help='The name of the model file in the HuggingFace Hub repo.')
    parser.add_argument('--upstream_revision', help="The commit hash of the specified HuggingFace Repository")
    parser.add_argument('-x', '--fix_feature_len', action='store_true', help="Fix the feature length")
    parser.add_argument('-q', '--ignore_length_dif', action='store_true', help="Fix the feature length")
    parser.add_argument('--update_results', action='store_true', help="update results on my google sheet (Fabian)")
    parser.add_argument('--json_file', type=str, default=None,
                        help='Optional Google service-account JSON for experiment logging to Google Sheets; omit to disable.')
    parser.add_argument('--sheet_row', type=int, default=1,help='the row number on where this pre-trained model experiment will be in the downstream experiment results.  set to 1 for now.')
    parser.add_argument('--current_row', type=int, default=2,help='pretrain-excel sheet row for model that is currently runing.')
    parser.add_argument('--find_best_checkpoint', action='store_true', help="to iterate folder and fend the best train loss checkpoint... (Fabian)")

    # experiment directory, choose one to specify
    # expname uses the default root directory: result/downstream
    parser.add_argument('-n', '--expname', help='Save experiment at result/downstream/expname')
    parser.add_argument('-p', '--expdir', help='Save experiment at expdir')
    parser.add_argument('-a', '--auto_resume', action='store_true', help='Auto-resume if the expdir contains checkpoints')
    parser.add_argument('--push_to_hf_hub', default=False, help='Push all files in experiment directory to the Hugging Face Hub. To use this feature you must set HF_USERNAME and HF_PASSWORD as environment variables in your shell')
    parser.add_argument('--hf_hub_org', help='The Hugging Face Hub organisation to push fine-tuned models to')

    # options
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')
    parser.add_argument('--cache_dir', help='The cache directory for pretrained model downloading')
    parser.add_argument('--verbose', action='store_true', help='Print model infomation')
    parser.add_argument('--disable_cudnn', action='store_true', help='Disable CUDNN')
    parser.add_argument('--features_path', help='The option for online feature saving.', type=str)
    parser.add_argument('--pre_extract_dir', help='The path for using offline feature preextracted by utility/feature_extractor.', type=str)

    args = parser.parse_args()
    backup_files = []

    if args.expdir is None:
        args.expdir = f'result/downstream/{args.expname}'

    if args.auto_resume:
        if os.path.isdir(args.expdir):
            ckpt_pths = glob.glob(f'{args.expdir}/states-*.ckpt')
            if len(ckpt_pths) > 0:
                args.past_exp = args.expdir


    print('[Runner] - Start a new experiment')
    os.makedirs(args.expdir, exist_ok=True)


    return args


def main():
    logging.basicConfig(level=logging.INFO)

    torch.multiprocessing.set_sharing_strategy('file_system')
    torchaudio.set_audio_backend('soundfile')
    hack_isinstance()

    # get config and arguments
    args  = get_downstream_args()
    if args.cache_dir is not None:
        torch.hub.set_dir(args.cache_dir)

    # When torch.distributed.launch is used
    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(args.backend)

    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    


    runner = Runner(args)


if __name__ == '__main__':
    main()
