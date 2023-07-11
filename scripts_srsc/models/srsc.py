"""VCOPN"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple

class SRSC(nn.Module):
    """Serial restoration of shuffled clips with PtrNet"""

    def __init__(self, base_network, feature_size, tuple_len, 
                hidden_dim, input_dim, mask=False, p=0.5):
        super(SRSC, self).__init__()

        if mask:
            from models.task_network_mask import TaskNet
            print("Model with mask")
        else:
            from models.task_network import TaskNet 
            print("Model without mask")

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len

        assert feature_size == input_dim, "output of base network does not match the input of TaskNet"
        self.srsc = TaskNet(input_dim, hidden_dim, p, tuple_len=tuple_len)
        

    def forward(self, X):
        # X: bs * seq_len * C * T * H * W
        # output: bs * seq_len * hidden_dim

        f = []  # clip features
        for i in range(self.tuple_len):
            clip = X[:, i, :, :, :, :]
            f.append(self.base_network(clip))
        features = torch.stack(f, dim=1)
        # bs * seq_len * feature_size 
        
        outputs, pointers = self.srsc(features)

        # bs * seq_len * seq_len, bs * seq_len * 1
        return outputs, pointers
