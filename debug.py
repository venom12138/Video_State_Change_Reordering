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
x = torch.randn(2, 4, 3, )
x = torch.repeat_interleave(x,2,dim=1)
print(x)