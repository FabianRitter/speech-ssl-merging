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
import pdb
import os
import math
import glob
import random
import importlib
from tqdm import tqdm
from collections import defaultdict
import yaml
import time
import json
#-------------#
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
#-------------#
from optimizers import get_optimizer, get_grouped_parameters
from schedulers import get_scheduler
from fvcore.nn import FlopCountAnalysis


def log_gpu_memory():
    memory_allocated = torch.cuda.memory_allocated()
    memory_reserved = torch.cuda.memory_reserved()
    return memory_allocated // 1e6, memory_reserved // 1e6

import gspread
from oauth2client.service_account import ServiceAccountCredentials

#### i WILL NEED A DICTIONARY WITH THE DATASETS STATS.
# I need this function to recover best train loss of some checkpoints I didnt analize.

def list_sheets(json_file):
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_file, scope)
    client = gspread.authorize(creds)

    # List all spreadsheets available to the service account
    sheet_list = client.openall()
    print("Available Sheets:")
    for sheet in sheet_list:
        print(sheet.title)


### for better management of experiments that are being run #### -> this will update an excell sheet automatically.
def authenticate_google_sheets(json_file, sheet_name, worksheet_name):
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_file, scope)
    client = gspread.authorize(creds)
    sheet = client.open(sheet_name)
    worksheet = sheet.worksheet(worksheet_name)
    return worksheet


def determine_cluster():
    current_dir = os.getcwd()
    print(f"current_dir in determine_cluster is {current_dir}")
    if current_dir.startswith("/home/project/") or current_dir.startswith("/data/projects"):
        return "NSCC CLUSTER"
    elif current_dir.startswith("/export/home2"):
        return "NTU CLUSTER"
    elif current_dir.startswith("/livingrooms/fabian/"):
        return "battleship cluster"
    else:
        return "Unknown Cluster"

    

def col_to_letter(col):
    """Convert a column index to a letter (e.g., 1 -> 'A')"""
    result = ""
    while col > 0:
        col, remainder = divmod(col - 1, 26)
        result = chr(65 + remainder) + result
    return result
    

def update_currently_running_experiments(args,config, sheet, epoch=None, total_epochs=None, global_time_difference=None, average_epoch_times=None, average_epoch_std=None, avg_memory_allocated=None, std_memory_allocated=None, total_flops=None, macs=None):
    running_where = determine_cluster()
    upstream_config = yaml.load(open(args.upstream_config, "r"), Loader=yaml.FullLoader)
    upstream_parameters = upstream_config[args.upstream] # should be either distiller or multi_distiller , hopefully


    # Determine the status
    if epoch is not None and total_epochs is not None:
        status = f"{epoch}/{total_epochs}"
    else:
        status = "just started running"
    
    # Define the base starting column index ('A' -> 1)
    base_start_col = 1
    num_values_cols = 24  # Number of columns to fetch/update including new ones
    
    # Calculate the starting column for the current fold
    start_col_index = base_start_col
    end_col_index = start_col_index + num_values_cols - 1
    
    start_col = col_to_letter(start_col_index)
    end_col = col_to_letter(end_col_index)
    col_range = f'{start_col}{args.current_row}:{end_col}{args.current_row}'
    # Fetch the existing row from the sheet
    current_general_stuff = sheet.get(col_range)
    if not current_general_stuff or len(current_general_stuff) == 0:
        current_general_stuff = [[]]  # Ensure there's at least an empty row structure

    # Ensure the row has enough columns by padding with empty strings
    while len(current_general_stuff[0]) < num_values_cols:
        current_general_stuff[0].append("")

    # GPU model information
    gpu_model = torch.cuda.get_device_name()

    # Prepare values to update
    if args.upstream == "distiller":
        values_general_stuff = [
            args.expdir.split("/")[-1],  # Experiment directory
            "DistilHub normal style",  # Style
            "l1 + cos",  # Loss function
            "",  # Placeholder for additional info
            "hubert_base",  # Base model
            "teacher model",  # Model type
            "None",  # Translator type
            config['optimizer']['name'],  # Optimizer
            config['optimizer']['lr'],  # Learning rate
            running_where,  # Cluster info
            os.getenv('USER'),  # User
            status,  # Current status
            args.sheet_row,  # Row in the sheet
            args.expdir,  # Experiment directory
            args.logfile,  # Log file path
            "",  # Placeholder
            global_time_difference,  # Total training time
            average_epoch_times,  # Average epoch time
            average_epoch_std,  # Epoch time standard deviation
            avg_memory_allocated,  # Avg GPU memory allocated
            std_memory_allocated,  # GPU memory std deviation
            gpu_model,  # GPU model
            total_flops,  # FLOPS per batch
            macs,  # MACs per batch
        ]
    else:
        values_general_stuff = [
            args.expdir.split("/")[-1],  # Experiment directory
            "DistilHub normal style",  # Style
            "l1 + cos",  # Loss function
            "",  # Placeholder for additional info
            upstream_parameters["teacher_names"][0],  # Teacher names
            upstream_parameters["initialize_from"][0],  # Init model
            upstream_parameters["translator_type"],  # Translator type
            config['optimizer']['name'],  # Optimizer
            config['optimizer']['lr'],  # Learning rate
            running_where,  # Cluster info
            os.getenv('USER'),  # User
            status,  # Current status
            args.sheet_row,  # Row in the sheet
            args.expdir,  # Experiment directory
            args.logfile,  # Log file path
            "",  # Placeholder
            global_time_difference,  # Total training time
            average_epoch_times,  # Average epoch time
            average_epoch_std,  # Epoch time standard deviation
            avg_memory_allocated,  # Avg GPU memory allocated
            std_memory_allocated,  # GPU memory std deviation
            gpu_model,  # GPU model
            total_flops,  # FLOPS per batch
            macs,  # MACs per batch  
        ]

    # Update the existing row with the new values
    for i, value in enumerate(values_general_stuff):
        if value is not None:
            current_general_stuff[0][i] = value

    # Update the sheet
    print("Updating currently running experiment details...")
    sheet.update(col_range, current_general_stuff)

    
    # print(f"upstream_parameters  ... {upstream_parameters}")
    # if not any(current_general_stuff):
    #     # If the row is empty, add the initial values
    #     if args.upstream == "distiller":
    #         values_general_stuff = [[args.expdir.split("/")[-1], "DistilHub normal style", "l1 + cos", "", "hubert_base", "teacher model", "None", config['optimizer']['name'], config['optimizer']['lr']  ,running_where  ,os.getenv('USER'), status, args.sheet_row, args.expdir ,args.logfile, "" ]]
    #     else:
    #         values_general_stuff = [[args.expdir.split("/")[-1], "DistilHub normal style", "l1 + cos", "", upstream_parameters["teacher_names"][0], upstream_parameters["initialize_from"][0], upstream_parameters["translator_type"], config['optimizer']['name'], config['optimizer']['lr']  ,running_where  ,os.getenv('USER'), status, args.sheet_row, args.expdir ,args.logfile, "" ]]
    #     print(f"Adding currently running experiment details")
    #     sheet.update(col_range, values_general_stuff)
    # else:
    #     if args.upstream == "distiller":
    #         values_general_stuff = [[args.expdir.split("/")[-1], "DistilHub normal style", "l1 + cos", "", "hubert_base", "teacher model", "None", config['optimizer']['name'], config['optimizer']['lr']  ,running_where  ,os.getenv('USER'), status, args.sheet_row, args.expdir ,args.logfile, "" ,global_time_difference, average_epoch_times, average_epoch_std, avg_memory_allocated, std_memory_allocated, gpu_model]]
    #     else:
    #         values_general_stuff = [[args.expdir.split("/")[-1], "DistilHub normal style", "l1 + cos", "", upstream_parameters["teacher_names"][0], upstream_parameters["initialize_from"][0], upstream_parameters["translator_type"], config['optimizer']['name'], config['optimizer']['lr']  ,running_where  ,os.getenv('USER'), status, args.sheet_row, args.expdir ,args.logfile, "" ,global_time_difference, average_epoch_times, average_epoch_std, avg_memory_allocated, std_memory_allocated,gpu_model]]
    #     # Update only the status column, keep other values unchanged
    #     current_general_stuff[0][11] = status  # Assuming status is the 11th column (index 10)
    #     current_general_stuff[0][16] = global_time_difference
    #     current_general_stuff[0][17] = average_epoch_times
    #     current_general_stuff[0][18] = average_epoch_std
    #     current_general_stuff[0][19] = avg_memory_allocated
    #     current_general_stuff[0][20] = std_memory_allocated
    #     current_general_stuff[0][21] = gpu_model
    #     print(f"Updating status to {status}")
    #     sheet.update(col_range, current_general_stuff)



def get_dataset_stats(loader):
    import torchaudio
    """
    Compute the mean and standard deviation of the dataset on mel-spectrogram!.
    """
    count = 0
    wav_sum = 0
    wav_sqsum = 0

    mean = []
    std = []
    target_length = 1024 # using the same of SSAST for now. It is harder to manipulate for now the parameter.
    
    for data in loader: ### batch iteration
        audio_input, _ , audio_length, pad = data
        wav_sum += torch.sum(audio_input)
        wav_sqsum += torch.sum(audio_input**2)
        count += audio_length.sum()
        
        fbank = torchaudio.compliance.kaldi.fbank(audio_input, htk_compat=True, sample_frequency=16000, use_energy=False,
                                                  window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        
        n_frames = fbank.shape[0]
        residual_frames = target_length - n_frames

        # cut and pad
        if residual_frames > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, residual_frames))
            fbank = m(fbank)
        elif residual_frames < 0:
            fbank = fbank[0:target_length, :]


        cur_mean = torch.mean(fbank)
        cur_std = torch.std(fbank)
        mean.append(cur_mean.item())
        std.append(cur_std.item())
        #print(f"Batch mean: {cur_mean}, Batch std: {cur_std}")
    
    wav_mean = wav_sum / count
    wav_var = (wav_sqsum / count) - (wav_mean**2)
    wav_std = np.sqrt(wav_var)

    mean_value_fbank = np.mean(mean)
    std_value_fbank = np.mean(std)
    print(f"Final dataset mean_value_fbank: {mean_value_fbank}, Final dataset std_value_fbank: {std_value_fbank}")
    print(f"Final dataset wav_mean: {wav_mean}, Final dataset wav_std: {wav_std}")


    return wav_mean, wav_std, mean_value_fbank, std_value_fbank


##########
# RUNNER #
##########
class Runner():
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, tensorboard logging, checkpoint saving
    """
    def __init__(self, args, config):
        self.args = args
        self.global_start_time = time.time()
        self.global_end_time = 0
        self.config = config
        self.logger = SummaryWriter(args.expdir)                                                 

        self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu') if self.args.init_ckpt else {}
        self.upstream = self._get_upstream()
        self.worksheet = authenticate_google_sheets(json_file=args.json_file, sheet_name=f'SLLM_encoder_eval' ,worksheet_name='Pre-trained-models')
        update_currently_running_experiments(self.args,self.config, self.worksheet)


    def _get_upstream(self):
        init_upstream = self.init_ckpt.get('Upstream_Config')
        if init_upstream:
            self.args.upstream_config = init_upstream
        module_path = f'pretrain.{self.args.upstream}.pretrain_expert'
        Upstream = getattr(importlib.import_module(module_path), 'UpstreamPretrainExpert')
        upstream = Upstream(self.config['pretrain_expert']['datarc'], 
                            self.args.upstream_config,
                            self.args.device,
                            self.args.multi_gpu).to(self.args.device)

        assert hasattr(upstream, 'device')
        assert hasattr(upstream, 'forward')
        assert hasattr(upstream, 'load_model')
        assert hasattr(upstream, 'add_state_to_save')
        assert hasattr(upstream, 'on_before_zero_grad')
        assert hasattr(upstream, 'get_train_dataloader')

        if self.init_ckpt != {}:
            print('[Runner] - Loading upstream weights from the previous experiment')
            upstream.load_model(self.init_ckpt)
        if hasattr(upstream, 'loss_to_device'):
            print('[Runner] - Loss to device')
            upstream.loss_to_device()
        return upstream


    def _get_optimizer(self, model_params):
        optimizer = get_optimizer(
            model_params, 
            self.config['runner']['total_steps'],
            self.config['optimizer']
        )

        if self.init_ckpt != {}:
            init_optimizer = self.init_ckpt.get('Optimizer')
            assert init_optimizer
            print('[Runner] - Loading optimizer weights from the previous experiment')
            optimizer.load_state_dict(init_optimizer)
        return optimizer


    def _get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            optimizer,
            self.config['runner']['total_steps'],
            self.config['scheduler']
        )

        if self.init_ckpt != {}:
            init_scheduler = self.init_ckpt.get('Scheduler')
            assert init_scheduler
            print('[Runner] - Loading scheduler weights from the previous experiment')
            scheduler.load_state_dict(init_scheduler)
        return scheduler


    def train(self):
        # set model train mode
        epoch_times = []
        self.upstream.train()

        # prepare data
        gradient_accumulate_steps = self.config['runner']['gradient_accumulate_steps']
        train_batch_size = self.config['pretrain_expert']['datarc']['train_batch_size']
        print('[Runner] - Accumulated batch size:', train_batch_size * gradient_accumulate_steps)
        dataloader = self.upstream.get_train_dataloader()
        print(f"self.config['pretrain_expert']['datarc']['data_stats']['wav_mean'] {self.config['pretrain_expert']['datarc']['data_stats']['wav_mean']}")


        # set epoch
        n_epochs = self.config['runner']['n_epochs']
        if n_epochs > 0: 
            total_steps = int(n_epochs * len(dataloader.dataset) / gradient_accumulate_steps)
            print(f'[Runner] - Training for {n_epochs} epochs, which is equivalent to {total_steps} steps')
        else:
            total_steps = self.config['runner']['total_steps']
            n_epochs = int(total_steps * gradient_accumulate_steps / len(dataloader.dataset))
            print(f'[Runner] - Training for {total_steps} steps, which is approximately {n_epochs} epochs')



        # set amp
        amp = self.config['runner'].get('fp16', False)
        if amp:
            print('[Runner] - Enabled fp16 training')
            scaler = torch.cuda.amp.GradScaler()

        # set optimizer
        model_params = [self.upstream.model]
        optimizer = self._get_optimizer(model_params)

        # set scheduler
        scheduler = None
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(optimizer)

        # set progress bar
        pbar = tqdm(total=total_steps, dynamic_ncols=True, desc='overall')

        all_loss = 0
        backward_steps = 0
        records = defaultdict(list)
        prefix = f'{self.args.upstream}/train-'

        # Initialize variable to track the lowest loss
        best_train_loss = float('inf')

        # if self.args.find_best_checkpoint:
        #     print("[Runner] - Finding best checkpoint...")
        #     self.find_best_checkpoint(dataloader, self.args, amp = amp)
        #     return

        memory_logs = []
        convergence_logs = []
        flops_calculated = True
        macs=None
        total_flops=None
        while pbar.n < pbar.total:
            epoch_start_time = time.time()
            epoch_train_loss = 0

            for data in tqdm(dataloader, dynamic_ncols=True, desc='train'):
                # try/except block for forward/backward
                try:
                    if pbar.n >= pbar.total:
                        break
                    global_step = pbar.n + 1

                     # FLOPS calculation for the first batch (to avoid overhead in every iteration)
                    if not flops_calculated:
                        pdb.set_trace()
                        #example_input = data.to(self.args.device)  # Analizis per batch
                        flop_analysis = FlopCountAnalysis(self.upstream, data)
                        total_flops = flop_analysis.total()
                        macs = flops_analysis.by_operator()["aten::mm"]  # Example for matrix multiplication MACs
                        print(f"[Runner] - FLOPS for forward pass: {total_flops}")
                        print(f"MACs per batch: {macs}")

                        flops_calculated = True

                    with torch.cuda.amp.autocast(enabled=amp):
                        loss, records = self.upstream(
                            data,
                            records=records,
                            global_step=global_step,
                            log_step=self.config['runner']['log_step'],
                        )

                    if gradient_accumulate_steps > 1:
                        loss = loss / gradient_accumulate_steps
                    if self.args.multi_gpu:
                        loss = loss.sum()
                    if amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Log GPU memory
                    memory_allocated, memory_reserved = log_gpu_memory()
                    memory_logs.append((memory_allocated, memory_reserved))
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {global_step}')
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise

                # record loss
                all_loss += loss.item()
                epoch_train_loss += loss.item()
                del loss
                
                # whether to accumulate gradient
                backward_steps += 1
                if backward_steps % gradient_accumulate_steps > 0:
                    continue
                    
                # unscale
                if amp:
                    scaler.unscale_(optimizer)

                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.upstream.model.parameters(), self.config['runner']['gradient_clipping'])
                if math.isnan(grad_norm):
                    print(f'[Runner] - Error : grad norm is NaN at global step {global_step}')

                # optimize
                if amp:
                    scaler.step(optimizer)
                    scaler.update()
                elif not math.isnan(grad_norm):
                    optimizer.step()

                self.upstream.on_before_zero_grad()
                optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()

                # Record the loss for this batch
                #epoch_train_loss += loss.item()
                # logging
                
                all_loss = 0      
                pbar.update(1)
            
            epoch_end_time = time.time()
            epoch_times.append(epoch_end_time - epoch_start_time)
            # Calculate stats for the current epoch
            average_epoch_times = sum(epoch_times) / len(epoch_times)
            squared_differences = [(x - average_epoch_times) ** 2 for x in epoch_times]
            average_epoch_std = math.sqrt(sum(squared_differences) / len(epoch_times))
            avg_memory_allocated = sum([log[0] for log in memory_logs]) / len(memory_logs)
            std_memory_allocated = math.sqrt(
                sum((log[0] - avg_memory_allocated) ** 2 for log in memory_logs) / len(memory_logs)
            )
            avg_memory_requested = sum([log[1] for log in memory_logs]) / len(memory_logs)
            std_memory_requested = math.sqrt(
                sum((log[1] - avg_memory_requested) ** 2 for log in memory_logs) / len(memory_logs)
            )
            print(f"Average GPU Memory Allocated: {avg_memory_allocated :.2f} MB (Std: {std_memory_allocated :.2f} MB)")
            print(f"Average GPU Memory Reqiesyed: {avg_memory_requested :.2f} MB (Std: {std_memory_requested :.2f} MB)")


            # Prepare logs for the current epoch
            epoch_log = {
                "epoch": len(epoch_times),
                "epoch_time": epoch_times[-1],
                "average_epoch_time": average_epoch_times,
                "epoch_time_std": average_epoch_std,
                "average_memory_allocated_MB": avg_memory_allocated,
                "memory_std_MB": std_memory_allocated,
                "average_memory_requested_MB": avg_memory_requested,
                "memory_std_requested_MB": std_memory_requested,
                "best_train_loss": best_train_loss if 'best_train_loss' in locals() else None,
                "convergence_logs": None,
            }

            update_currently_running_experiments(self.args, self.config, self.worksheet, pbar.n , pbar.total, 
            average_epoch_times=average_epoch_times, 
            average_epoch_std=average_epoch_std, 
            avg_memory_allocated=avg_memory_allocated, 
            std_memory_allocated=std_memory_allocated,
            total_flops=total_flops,  # Add FLOPS
            macs=macs                 # Add MACs
            )

        average_epoch_times = sum(epoch_times) / len(epoch_times)
        # Calculate standard deviation
        squared_differences = [(x - average_epoch_times) ** 2 for x in epoch_times]
        average_epoch_std = math.sqrt(sum(squared_differences) / len(epoch_times))
        ### here I will send the info of times and std and others to the excell sheet where i am adding more columns.
        avg_memory_allocated = sum([log[0] for log in memory_logs]) / len(memory_logs)
        # Calculate standard deviations
        std_memory_allocated = math.sqrt(
            sum((log[0] - avg_memory_allocated) ** 2 for log in memory_logs) / len(memory_logs)
        )

        self.global_end_time = time.time()
        global_time_difference = self.global_end_time - self.global_start_time

        print(f"Total Training Time (Global): {global_time_difference:.2f} seconds")
        print(f"Average GPU Memory Allocated: {avg_memory_allocated :.2f} MB (Std: {std_memory_allocated :.2f} MB)")
        print(f"Average Epoch Time: {average_epoch_times:.2f}s")
        print(f"Epoch Time Standard Deviation: {average_epoch_std:.2f}s")
        
        update_currently_running_experiments(self.args, self.config, self.worksheet, pbar.n , pbar.total, global_time_difference, average_epoch_times, average_epoch_std, avg_memory_allocated, std_memory_allocated, total_flops=total_flops,macs=macs)
        pbar.close()
