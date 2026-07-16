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
#-------------#
import torch
import numpy as np
#-------------#
from pretrain.runner_get_checkpoint  import Runner
from utility.helper import override


######################
# PRETRAIN ARGUMENTS #
######################
def get_pretrain_args():
    parser = argparse.ArgumentParser()

    # use a ckpt as the experiment initialization
    # if set, all the following args and config will be overwrited by the ckpt, except args.mode
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


    args.expdir = f'result/pretrain/{args.expname}'


    print('[Runner] - Start a new experiment')
    args.init_ckpt = None

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
