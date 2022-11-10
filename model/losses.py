import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
import numpy as np

class LossComputer:
    def __init__(self, config):
        super().__init__()
        self.config = config

    def compute(self, logits, target):
        losses = {}
        losses['total_loss'] = F.cross_entropy(logits, target)
        return losses
