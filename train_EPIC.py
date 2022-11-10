import datetime
from os import path
import math
# import git

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as distributed
import torch.nn as nn
from model.network import FrameReorderNet
from model.trainer import Trainer
# from dataset.static_dataset import StaticTransformDataset
# from dataset.vos_dataset import VOSDataset
from dataset.EPIC_dataset import EPICDataset
from dataset.EPIC_testdataset import EPICtestDataset
from argparse import ArgumentParser
from util.exp_handler import *
import pathlib
import sys
sys.path.append('./visualize')
from visualize.visualize_eval_result_eps import visualize_eval_result
import wandb
from glob import glob
import shutil

def get_EPIC_parser():
    parser = ArgumentParser()

    # Enable torch.backends.cudnn.benchmark -- Faster in some cases, test in your own environment
    parser.add_argument('--benchmark', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_true')

    # Data parameters
    parser.add_argument('--epic_root', help='EPIC data root', default='./EPIC_train') # TODO
    parser.add_argument('--val_data_root', help='EPIC val data root', default='./val_data') # TODO
    parser.add_argument('--yaml_root', help='yaml root', default='./EPIC_train/EPIC100_state_positive_train.yaml')
    parser.add_argument('--val_yaml_root', help='yaml root', default='./val_data/EPIC100_state_positive_val.yaml')
    parser.add_argument('--num_workers', help='Total number of dataloader workers across all GPUs processes', type=int, default=16)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--iterations', default=10000, type=int)
    parser.add_argument('--steps', nargs="*", default=[1000,8000], type=int)
    parser.add_argument('--lr', help='Initial learning rate', default=1e-5, type=float)
    parser.add_argument('--num_frames', default=5, type=int)

    parser.add_argument('--gamma', help='LR := LR*gamma at every decay step', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)

    # Loading
    parser.add_argument('--load_network', default='', help='Path to pretrained network weight only')

    # Logging information
    parser.add_argument('--log_text_interval', default=100, type=int)
    parser.add_argument('--save_network_interval', default=500, type=int)

    parser.add_argument('--debug', help='Debug mode which logs information more often', action='store_true')

    # Multiprocessing parameters, not set by users
    parser.add_argument('--local_rank', default=0, type=int, help='Local rank of this process')
    parser.add_argument('--en_wandb', action='store_true')
    
    parser.add_argument('--cos_lr', action='store_true')
    parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

    parser.add_argument('--freeze', default=0, type=int, choices=[0,1])
    parser.add_argument('--repr_type', type=str, choices=['SlowFast', 'VideoMae'])
    parser.add_argument('--use_position_embedding', default=1, type=int, choices=[0,1])
    parser.add_argument('--openword_test', default=0, type=int, choices=[0,1])
    parser.add_argument('--config_file', default='./model/slowfast/SLOWFAST_8x8_R50.yaml', type=str)
    args = parser.parse_args()
    return {**vars(args), **{'amp': not args.no_amp}}

"""
Initial setup
"""
# Init distributed environment
distributed.init_process_group(backend="nccl")
print(f'CUDA Device count: {torch.cuda.device_count()}')

# Parse command line arguments
# 只针对于EPIC数据集
config = get_EPIC_parser()

torch.backends.cudnn.benchmark = True

local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print(f'I am rank {local_rank} in this world of size {world_size}!')

# Set seed to ensure the same initialization
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

config['num_gpus'] = world_size
if config['batch_size']//config['num_gpus']*config['num_gpus'] != config['batch_size']:
    raise ValueError('Batch size must be divisible by the number of GPUs.')
# 分配给每个GPU，每个GPU的bs、worker
config['batch_size'] //= config['num_gpus']
config['num_workers'] //= config['num_gpus']
print(f'We are assuming {config["num_gpus"]} GPUs.')

print(f'We are now starting stage EPIC')

if config['debug']:
    config['batch_size'] = 1
    config['num_frames'] = 5
    config['iterations'] = 5
    config['log_text_interval']  = 1
    config['save_network_interval']  = 2
    config['log_image_interval'] = 1

"""
Model related
"""
if local_rank == 0:    
    # exp_handler
    exp = ExpHandler(config=config, en_wandb=config['en_wandb'], resume=config['resume'])
else:
    exp = None


model = Trainer(config, logger=exp, 
            local_rank=local_rank, world_size=world_size).train()

# dummy_input = [torch.randn(2, 3, 8, 224, 224).cuda(), torch.randn(2, 3, 32, 224, 224).cuda()]
# dummy_input = torch.randn(2,16,3,224,224).cuda()
# output = model.model.module.Encoder(dummy_input).squeeze()
# print(f"output shape: {output.shape}")
# ddd

"""
Dataloader related
"""
# To re-seed the randomness everytime we start a worker
def worker_init_fn(worker_id): 
    worker_seed = torch.initial_seed()%(2**31) + worker_id + local_rank*100
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def construct_loader(dataset):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
    train_loader = DataLoader(dataset, config['batch_size'], sampler=train_sampler, num_workers=config['num_workers'],
                            worker_init_fn=worker_init_fn, drop_last=True)
    return train_sampler, train_loader

train_dataset = EPICDataset(data_root=config['epic_root'], yaml_root=config['yaml_root'], openword_test=config['openword_test'], 
                        num_frames=config['num_frames'], repr_type=config['repr_type'])

train_sampler, train_loader = construct_loader(train_dataset)
val_dataset = EPICtestDataset(data_root='./val_data', yaml_root='./val_data/EPIC100_state_positive_val.yaml', valset_yaml_root='./val_data/reordering_val.yaml', 
                        num_frames=5, repr_type=config['repr_type'])

val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

# Load pertrained model if needed
total_iter = 0

total_epoch = math.ceil(config['iterations']/len(train_loader))
current_epoch = total_iter // len(train_loader)
print(f'We approximately use {total_epoch} epochs.')

# Need this to select random bases in different workers
np.random.seed(np.random.randint(2**30-1) + local_rank*100)

try:
    while total_iter < config['iterations']:
        # Crucial for randomness! 
        train_sampler.set_epoch(current_epoch)
        current_epoch += 1
        print(f'Current epoch: {current_epoch}')

        for data in train_loader:
            # Train loop
            model.train()
            model.do_pass(data, total_iter, val_loader)
            total_iter += 1

            if total_iter > config['iterations']:
                break
            
            # if total_iter % 1000 == 0 and exp is not None:
            #     eval_metrics = validate(model.module, val_loader)
            #     exp.write(eval_metrics, total_iter)
finally:
    # not config['debug'] and total_iter>5000
    if model.logger is not None:
        model.save_network(total_iter)

del model

# 还没写呢

distributed.destroy_process_group()

