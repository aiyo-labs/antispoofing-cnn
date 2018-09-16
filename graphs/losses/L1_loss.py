"""
Loss for AntiSpoofing
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

class L1_loss(nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, input, target):
        loss = self.loss(input,target)
        return loss