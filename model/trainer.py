import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import git
import datetime
from itertools import permutations
# TODO change to relative path
sys.path.append('/home/venom/projects/XMem/')
# from model.losses import LossComputer
# from network import XMem
from copy import deepcopy

import model.resnet as resnet
from model.network import FrameReorderNet
from model.losses import LossComputer
import matplotlib.pyplot as plt
from util.log_integrator import Integrator
from tqdm import tqdm
import scipy
from scipy import stats

# scores: [B, 5, 5]
def get_max_permutation(scores):
    # scores: B,5,5
    # 0 1 2 3 4
    all_perms = torch.tensor(list(permutations(range(5)))).to(scores.device) # [120, 5]
    perms_scores = torch.zeros((scores.shape[0], 120)).to(scores.device) # b,120
    for b in range(all_perms.shape[1]-1):        
        perms_scores[:] += scores[:, all_perms[:, b], all_perms[:, b+1]]
    # print(f"perms_scores: {perms_scores}")
    return torch.flip(all_perms[torch.argmax(perms_scores, dim=1)], [1]) # B,5

def spearman_acc(story, gt_order):
    try:
        story = story.cpu().numpy().tolist()
    except:
        story = story.tolist()
    try:
        gt_order = gt_order.cpu().numpy().tolist()
    except:
        gt_order = gt_order.tolist()
        
    story_rank = [story.index(i) for i in range(5)]
    gt_rank = [gt_order.index(i) for i in range(5)]
    
    return scipy.stats.spearmanr(story_rank, gt_rank)[0]

def absolute_distance(story, gt_order):
    try:
        story = story.cpu().numpy().tolist()
    except:
        story = story.tolist()
    
    try:
        gt_order = gt_order.cpu().numpy().tolist()
    except:
        gt_order = gt_order.tolist()
        
    story_rank = [story.index(i) for i in range(5)]
    gt_rank = [gt_order.index(i) for i in range(5)]

    return np.mean(np.abs(np.array(story_rank) - np.array(gt_rank)))

# def pairwise_acc(story, gt_order):
#     correct = 0
#     # gt order 原本比如是 3，2，4，0，1
#     # predict的story是 4，0，2，3，1
#     # 那么将3：0，2：1，4：2，0：3，1：4做这样一个替换
#     # story就变成了 2，3，1，0，4
#     # 然后gt_order就变为了0，1，2，3，4
#     for i in range(len(gt_order)):
#         index = story.index(gt_order[i])
#         story[index] = i
        
#     gt_order = list(range(len(gt_order)))
#     total = len(story) * (len(story)-1) // 2
#     for idx1 in range(len(story)):
#         for idx2 in range(idx1+1, len(story)):
#             if story[idx1] < story[idx2]:
#                 correct += 1
#     return correct/total

def pairwise_acc(story, gt_order):
    correct = 0
    try:
        story = story.cpu().numpy().tolist()
    except:
        story = story.tolist()
    try:
        gt_order = gt_order.cpu().numpy().tolist()
    except:
        gt_order = gt_order.tolist()
    total = len(story) * (len(story)-1) // 2
    for idx1 in range(len(story)):
        for idx2 in range(idx1+1, len(story)):
            if gt_order.index(story[idx1]) < gt_order.index(story[idx2]):
                correct += 1
    return correct/total

def validate(model, val_loader, config):
    all_scores = []
    all_gt = []
    open_word_acc = {'svsn':{'scores':[], 'gt':[]}, 
                'svnn':{'scores':[], 'gt':[]}, 
                'nvnn':{'scores':[], 'gt':[]}, 
                'nvsn':{'scores':[], 'gt':[]}}
    model.eval()
    # Start eval
    for ti, data in tqdm(enumerate(val_loader)):  
        with torch.no_grad():
            openword_type = data['open_word_type']
            
            frames = data['rgb'].cuda()
            gt_order = data['gt_order'].cuda()
            if config['repr_type'] == 'SlowFast':
                img_features = model.encode([data['slow_images'].cuda(), frames]).squeeze() # [B, num_frames, 2048/1024]
            elif config['repr_type'] == 'VideoMae':
                img_features = model.encode(frames) # [B, num_frames, 2048/1024]
            
            scores = torch.zeros(img_features.shape[0], img_features.shape[1], img_features.shape[1]).cuda() # [B, num_frames, num_frames]
            # scores[b, i,j]代表第b个batch i>j的概率
            # scores = torch.zeros((img_features.shape[0], img_features.shape[1])).cuda() # [B, num_frames],代表了每一帧的得分
            for idx1 in range(5):
                for idx2 in range(5):
                    if idx1 == idx2:
                        continue
                    else:
                        cat_feature = torch.cat([img_features[:, idx1], img_features[:, idx2]], dim = 1)
                        logits = model.classify(cat_feature) # [B,2] 
                        prob = torch.softmax(logits, dim = 1) # [B,2]
                        scores[:, idx1, idx2] = prob[:, 1] # [B]
                        
            perm = get_max_permutation(scores) # B, 5
            all_scores.append(perm)
            all_gt.append(gt_order)
            for i in range(perm.shape[0]):
                # print(openword_type[i])
                open_word_acc[openword_type[i]]['scores'].append(perm[i].cpu().numpy())
                open_word_acc[openword_type[i]]['gt'].append(gt_order[i].cpu().numpy())
    
    all_scores = torch.cat(all_scores, dim = 0).cpu()
    all_gt = torch.cat(all_gt, dim = 0).cpu().numpy()

    Spearman = np.mean([spearman_acc(all_scores[i], all_gt[i]) for i in range(len(all_scores))])
    Absoulte_Distance = np.mean([absolute_distance(all_scores[i], all_gt[i]) for i in range(len(all_scores))])
    Pairwise = np.mean([pairwise_acc(all_scores[i], all_gt[i]) for i in range(len(all_scores))])
    openword_dict = {}
    for key in open_word_acc.keys():
        key_spearman = np.mean([spearman_acc(open_word_acc[key]['scores'][i], open_word_acc[key]['gt'][i]) for i in range(len(open_word_acc[key]['scores']))])
        key_ab_distance = np.mean([absolute_distance(open_word_acc[key]['scores'][i], open_word_acc[key]['gt'][i]) for i in range(len(open_word_acc[key]['scores']))])               
        key_pairwise_acc = np.mean([pairwise_acc(open_word_acc[key]['scores'][i], open_word_acc[key]['gt'][i]) for i in range(len(open_word_acc[key]['scores']))])
        openword_dict.update({f'{key}/Spearman':key_spearman, 
                        f'{key}/Absoulte_Distance':key_ab_distance, 
                        f'{key}/Pairwise':key_pairwise_acc})
        
    return {**{'all/Spearman':Spearman, 
            'all/Absoulte_Distance':Absoulte_Distance, 
            'all/Pairwise':Pairwise}, **openword_dict}

class Trainer:
    def __init__(self, config, logger, local_rank, world_size):
        self.config = config
        self.logger = logger
        
        network = FrameReorderNet(config=config)
        self.model = nn.parallel.DistributedDataParallel(
                    network.cuda(), 
                    device_ids=[local_rank], output_device=local_rank, 
                    broadcast_buffers=False, find_unused_parameters=True)
        # 升级版的average_meter 
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        self.loss_computer = LossComputer(config)
        self.train()
        
        if logger is not None:
            self.last_time = time.time()
            self.logger.log(f'model_size:{str(sum([param.nelement() for param in self.model.parameters()]))}')
            self.save_path = logger._save_dir
        else:
            self.save_path = None
        
        if self.config['freeze']:
            print('Freezing the encoder in Trainer')
            for param in self.model.module.Encoder.parameters():
                param.requires_grad = False
        
        self.optimizer = optim.AdamW(filter(
            lambda p: p.requires_grad, self.model.parameters()), lr=config['lr'], weight_decay=config['weight_decay'])
        
        print('parameter not requires grad:')
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(name)
        print('------------------')

        
        if config['cos_lr']:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config['iterations'])
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, config['steps'], config['gamma'])
            
        if config['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.log_text_interval = config['log_text_interval']
        self.save_network_interval = config['save_network_interval']
        
    def do_pass(self, data, it, val_loader):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)
        frames = data['rgb'].cuda() # [B, num_frames, 3, H, W]
        gt_order = data['gt_order'].cuda() # [B, num_frames]
        if self.config['repr_type'] == 'SlowFast':
            img_features = self.model.module.encode([data['slow_images'].cuda(), frames]).squeeze() # [B, num_frames, 2048/1024]
        elif self.config['repr_type'] == 'VideoMae':
            img_features = self.model.module.encode(frames) # [B, num_frames, 2048/1024]
        
        all_logits = []
        all_target = []
        for idx1 in range(self.config['num_frames']):
            for idx2 in range(self.config['num_frames']):
                if idx1 == idx2:
                    continue
                else:
                    cat_feature = torch.cat([img_features[:, idx1], img_features[:, idx2]], dim = 1)
                    logits = self.model.module.classify(cat_feature) # [B,2]
                    target = (gt_order[:, idx1] > gt_order[:, idx2]).to(torch.int64) # [B]
                    
                    all_logits.append(logits)
                    all_target.append(target)

        # dd
        all_logits = torch.stack(all_logits, 0).flatten(start_dim=0, end_dim=1)
        all_target = torch.stack(all_target, 0).flatten(start_dim=0, end_dim=1)
        
        losses = self.loss_computer.compute(all_logits, all_target)
        
        # recording
        if self._do_log:
            self.integrator.add_dict(losses)
        
        if self._is_train:
            
            if (it) % self.log_text_interval == 0 and it != 0:
                train_metrics = self.train_integrator.finalize()
                if self.logger is not None:
                    self.model.eval()
                    eval_metrics = validate(self.model.module, val_loader, self.config)
                    self.model.train()
                    self.logger.write(prefix='reorder', train_metrics=train_metrics, eval_metrics=eval_metrics,**{'lr':self.scheduler.get_last_lr()[0],
                                    'time':(time.time()-self.last_time)/self.log_text_interval})
                    all_dicts = {**train_metrics, **{'lr':self.scheduler.get_last_lr()[0],
                                        'time':(time.time()-self.last_time)/self.log_text_interval}}
                    self.last_time = time.time()
                    for k, v in all_dicts.items():
                        msg = 'It {:6d} [{:5s}] [{:13}]: {:s}'.format(it, 'TRAIN', k, '{:.9s}'.format('{:0.9f}'.format(v)))
                        if self.logger is not None:
                            self.logger.log(msg)
                    for k, v in eval_metrics.items():
                        msg = 'It {:6d} [{:5s}] [{:13}]: {:s}'.format(it, 'EVAL', k, '{:.9s}'.format('{:0.9f}'.format(v)))
                        if self.logger is not None:
                            self.logger.log(msg)
                    print('-------------------')
                self.train_integrator.reset_except_hooks()

            if it % self.save_network_interval == 0 and it != 0:
                if self.logger is not None:
                    self.save_network(it)
        
        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.config['amp']:
            self.scaler.scale(losses['total_loss']).backward()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward() 
            
            self.optimizer.step()
        self.scheduler.step()
        
        # print(f"all_logits:{all_logits.shape}")
        # print(f"all_target:{all_target.shape}")
        # print(f"all_targets:{all_target}")

    
    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # 不使用BN和dropout
        self.model.train()
        return self

    def eval(self):
        self._is_train = False
        self._do_log = True
        self.model.eval()
        return self
    
    def save_network(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}/network_{it}.pth'
        torch.save(self.model.module.state_dict(), model_path)
        torch.save(self.model.module.state_dict(), f'{self.save_path}/latest_network.pth')
        print(f'Network saved to {model_path}.')