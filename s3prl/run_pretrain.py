# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ run_pretrain.py ]
#   Synopsis     [ scripts for running the pre-training of upstream models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import re
import yaml
import glob
import random
import argparse
import importlib
from shutil import copyfile
from argparse import Namespace
import time
#-------------#
import torch
import numpy as np
#-------------#
from pretrain.runner import Runner
from utility.helper import override


######################
# PRETRAIN ARGUMENTS #
######################
def get_pretrain_args():
    parser = argparse.ArgumentParser()

    # use a ckpt as the experiment initialization
    # if set, all the following args and config will be overwrited by the ckpt, except args.mode
    parser.add_argument('--compute_pretrain_statistics', action='store_true', 
                        help='Compute pretrain stats for a single epoch without saving parameters')
    parser.add_argument('-e', '--past_exp', metavar='{CKPT_PATH,CKPT_DIR}', help='Resume training from a checkpoint')
    parser.add_argument('-o', '--override', help='Used to override args and config, this is at the highest priority')

    # configuration for the experiment, including runner and downstream
    parser.add_argument('--backend', default='nccl', help='The backend for distributed training')
    parser.add_argument('--local_rank', type=int,
                        help=f'The GPU id this process should use while distributed training. \
                               None when not launched by torch.distributed.launch')
    parser.add_argument('-c', '--config', metavar='CONFIG_PATH', help='The yaml file for configuring the whole experiment, except the upstream model')

    # upstream settings
    parser.add_argument('-u', '--upstream', choices=os.listdir('pretrain/'))
    parser.add_argument('-g', '--upstream_config', metavar='CONFIG_PATH', help='The yaml file for configuring the upstream model')

    # experiment directory, choose one to specify
    # expname uses the default root directory: result/pretrain
    parser.add_argument('-n', '--expname', help='Save experiment at expdir/expname')
    parser.add_argument('-p', '--expdir', help='Save experiment at expdir')
    parser.add_argument('-a', '--auto_resume', action='store_true', help='Auto-resume if the expdir contains checkpoints')
    parser.add_argument('--logfile', type=str)
    parser.add_argument('--json_file', type=str, default=None,
                        help='Optional Google service-account JSON for experiment logging to Google Sheets; omit to disable.')
    parser.add_argument('--sheet_row', type=int, default=1,help='the row number on where this pre-trained model experiment will be in the downstream experiment results.  set to 1 for now.')
    parser.add_argument('--current_row', type=int, default=2,help='pretrain-excel sheet row for model that is currently runing.')
    parser.add_argument('--find_best_checkpoint', action='store_true', help="to iterate folder and fend the best train loss checkpoint... (Fabian)")

    # options
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')
    parser.add_argument('--multi_gpu', action='store_true', help='Enables multi-GPU training')

    args = parser.parse_args()


    if args.expdir is None:
        args.expdir = f'result/pretrain/{args.expname}'

    if args.auto_resume:
        if os.path.isdir(args.expdir):
            ckpt_pths = glob.glob(f'{args.expdir}/states-*.ckpt')
            if len(ckpt_pths) > 0:
                args.past_exp = args.expdir

    if args.past_exp:
        # determine checkpoint path
        if os.path.isdir(args.past_exp):
            ckpt_pths = glob.glob(f'{args.past_exp}/states-*.ckpt')
            assert len(ckpt_pths) > 0
            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
            ckpt_pth = ckpt_pths[-1]
        else:
            ckpt_pth = args.past_exp
        
        print(f'[Runner] - Resume from {ckpt_pth}')

        # load checkpoint
        ckpt = torch.load(ckpt_pth, map_location='cpu')

        def update_args(old, new, preserve_list=None):
            out_dict = vars(old)
            new_dict = vars(new)

            
            for key in list(new_dict.keys()):
                if key in preserve_list:
                    new_dict.pop(key)
            out_dict.update(new_dict)
            return Namespace(**out_dict)
        
        # overwrite args
        cannot_overwrite_args = [
            'mode', 'evaluate_split', 'override',
            'backend', 'local_rank', 'past_exp',
        ]

        args = update_args(args, ckpt['Args'], preserve_list=cannot_overwrite_args)
        
        # overwrite args and config
        os.makedirs(args.expdir, exist_ok=True)
        args.init_ckpt = ckpt_pth
        
        #config = ckpt['Config'] # fix because the multidistiller saved it wrongly..! why is this saved like this? it seems there was a mistake in the John code...
        
        
        upstream_dirs = [u for u in os.listdir('pretrain/') if re.search(f'^{u}_|^{u}$', args.upstream)]
        assert len(upstream_dirs) == 1
        print(f"RUNNING NORMAL LOAD OF CONFIG FILE IN past_exp PART BECAUSE THE MULTIDISTILLER DID NOT SAVED CONFIG STUFF PROPERLY")
        print(f"you need to ERASE THIS HACK LATER ON AND FIX THE CONFIG FILE SAVING INSIDE THE ckpt IN THE MULTIDISTILLER LATER ON!")
        if args.config is None:
            args.config = f'pretrain/{upstream_dirs[0]}/config_runner.yaml'
        with open(args.config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)



        print(f"the updated config file is {config}")

    else:
        print('[Runner] - Start a new experiment')
        args.init_ckpt = None

        assert args.expname is not None
        if args.expdir is None:
            args.expdir = f'result/pretrain/{args.expname}'
        os.makedirs(args.expdir, exist_ok=True)

        upstream_dirs = [u for u in os.listdir('pretrain/') if re.search(f'^{u}_|^{u}$', args.upstream)]
        assert len(upstream_dirs) == 1

        if args.config is None:
            args.config = f'pretrain/{upstream_dirs[0]}/config_runner.yaml'
        with open(args.config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        if os.path.isfile(args.config):
            copyfile(args.config, f'{args.expdir}/config_runner.yaml')
        else:
            raise FileNotFoundError('Wrong file path for runner config.')
        
        if args.upstream_config is None:
            default_upstream_config = f'pretrain/{upstream_dirs[0]}/config_model.yaml'
            assert os.path.isfile(default_upstream_config)
            args.upstream_config = default_upstream_config
        if os.path.isfile(args.upstream_config):
            copyfile(args.upstream_config, f'{args.expdir}/config_model.yaml')
        else:
            raise FileNotFoundError('Wrong file path for model config.')

    if args.override is not None and args.override.lower() != "none":
        override(args.override, args, config)
        os.makedirs(args.expdir, exist_ok=True)

    return args, config


########
# MAIN #
########
def main():
    # get config and arguments
    args, config = get_pretrain_args()

    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    runner = Runner(args, config)
    eval('runner.train')()
    runner.logger.close()


if __name__ == '__main__':
    main()
