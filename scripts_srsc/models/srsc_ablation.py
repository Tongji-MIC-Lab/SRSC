"""VCOPN"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
from models.task_network_ablation import TaskNetAblateOAE, TaskNetAblateHAP

class SRSCAblation(nn.Module):
    """Serial restoration of shuffled clips with PtrNet"""

    def __init__(self, base_network, feature_size, tuple_len, 
                hidden_dim, input_dim, p=0.5, ablation=None):
        super(SRSCAblation, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len

        assert feature_size == input_dim, "output of base network does not match the input of TaskNet"
        self.ablation = ablation
        if ablation == 'OAE':
            self.srsc = TaskNetAblateOAE(input_dim, hidden_dim, p, tuple_len=tuple_len)
        elif ablation == 'HAP':
            self.srsc = TaskNetAblateHAP(input_dim, hidden_dim, p, tuple_len=tuple_len)
        else:
            raise ValueError("ERROR: NO SUCH ABLATION MODE.")


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