import torch
import torch.nn as nn
import torch.nn.functional as F

from model import resnet
import clip

from torchsummary import summary
from einops import rearrange
from dataset.EPIC_testdataset import EPICtestDataset
from itertools import permutations
import numpy as np
import scipy
from scipy import stats
import yaml
from easydict import EasyDict

# x: B, nf, 3, H, W
# x = torch.randn(2, 4, 3, )
# x = torch.repeat_interleave(x,2,dim=1)
# print(x)
val_dataset = EPICtestDataset(data_root='./val_data', yaml_root='./val_data/EPIC100_state_positive_val.yaml', valset_yaml_root='./val_data/reordering_val.yaml', 
                            num_frames=5, repr_type='SlowFast')

data = val_dataset[2]
print(data['slow_images'].shape)
print(data['info'])
print(data['gt_order'])