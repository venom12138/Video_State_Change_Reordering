import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import resnet
import clip
import scipy
from scipy import stats
from torchsummary import summary
from einops import rearrange
from dataset.EPIC_testdataset import EPICtestDataset

def spearman_acc(story):
    return scipy.stats.spearmanr(story, [0,1,2,3,4])[0]

def absolute_distance(story):
    return np.mean(np.abs(np.array(story) - np.array([0,1,2,3,4])))

def pairwise_acc(story):
    correct = 0
    total = len(story) * (len(story)-1) // 2
    for idx1 in range(len(story)):
        for idx2 in range(idx1+1, len(story)):
            if story[idx1] < story[idx2]:
                correct += 1
    return correct/total


if __name__ == '__main__':
    # test_dataset = EPICtestDataset(data_root='/home/venom/projects/XMem/val_data', 
    #                     yaml_root='/home/venom/projects/XMem/val_data/EPIC100_state_positive_val.yaml', 
    #                     valset_yaml_root='/home/venom/projects/State_Change_reordering/unused/reordering_val.yaml', 
    #                     num_frames=5, repr_type='ImageNet')
    # data = test_dataset[2]
    # print(f"rgb: {data['rgb'].shape}")
    # print(f"gt_order:{data['gt_order']}")
    # print(f"text: {data['text']}")
    noise = np.random.rand(98, 5)  # noise in [0, 1]
    idx = np.argsort(noise, axis=1)
    
    print('Spearman:')
    print(np.mean([spearman_acc(st) for st in idx]))

    print('Absoulte Distance:')
    print(np.mean([absolute_distance(st) for st in idx]))

    print('Pairwise:')
    print(np.mean([pairwise_acc(st) for st in idx]))
    print('---------------------')
    
    print('Spearman:')
    print(np.mean([spearman_acc(st) for st in [[0,1,2,3,4]]]))

    print('Absoulte Distance:')
    print(np.mean([absolute_distance(st) for st in [[0,1,2,3,4]]]))

    print('Pairwise:')
    print(np.mean([pairwise_acc(st) for st in [[0,1,2,3,4]]]))
    