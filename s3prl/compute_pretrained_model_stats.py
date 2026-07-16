# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ compute_pretrained_model_stats.py ]
#   Synopsis     [ Script for computing model pretraining statistics without saving parameters ]
"""*********************************************************************************************"""

import os
import glob
import re
import yaml
import random
import argparse
import importlib
from shutil import copyfile
from argparse import Namespace
import time
import torch
import numpy as np
from pretrain.runner_stats import Runner
from utility.helper import override

def get_pretrain_stats_args():
    parser = argparse.ArgumentParser()
    
    # Add new flag for computing statistics
    parser.add_argument('--compute_pretrain_statistics', action='store_true', 
                        help='Compute pretrain stats for a single epoch without saving parameters')

    # Other arguments (same as run_pretrain.py)
    parser.add_argument('-e', '--past_exp', metavar='{CKPT_PATH,CKPT_DIR}', help='Resume training from a checkpoint')
    parser.add_argument('-o', '--override', help='Used to override args and config, this is at the highest priority')
    parser.add_argument('--backend', default='nccl', help='The backend for distributed training')
    parser.add_argument('-c', '--config', metavar='CONFIG_PATH', help='The yaml file for configuring the whole experiment')
    parser.add_argument('-u', '--upstream', choices=os.listdir('pretrain/'))
    parser.add_argument('-g', '--upstream_config', metavar='CONFIG_PATH', help='The yaml file for configuring the upstream model')
    parser.add_argument('-n', '--expname', help='Save experiment at expdir/expname')
    parser.add_argument('--logfile', type=str)
    parser.add_argument('--json_file', type=str, default="/path/to/json/file")
    parser.add_argument('--sheet_row', type=int, default=1, help='Row in the sheet')
    parser.add_argument('--current_row', type=int, default=2, help='Excel row for current model')
    parser.add_argument('--device', default='cuda', help='model.to(device)')
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--multi_gpu', action='store_true', help='Enables multi-GPU training')
    
    args = parser.parse_args()
    args.expdir = f'result/pretrain/{args.expname}'
    ckpt_pths = glob.glob(f'{args.expdir}/states-*.ckpt')
    assert len(ckpt_pths) > 0
    ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
    ckpt_pth = ckpt_pths[-1]
    print(f'[Runner] - Computing stats from model {ckpt_pth}')
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
    print(f"args is {args}")
    args.init_ckpt = ckpt_pth

    return args

def main():
    args = get_pretrain_stats_args()
    args.config = args.config or f'pretrain/{args.upstream}/config_runner.yaml'
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Add restriction for a single epoch
    if args.compute_pretrain_statistics:
        config['runner']['n_epochs'] = 3  # Force single epoch
        config['runner']['total_steps'] = 100000

    runner = Runner(args, config)
    eval('runner.train')()
    runner.logger.close()

if __name__ == '__main__':
    main()
