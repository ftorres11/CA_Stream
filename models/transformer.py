# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch imports
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

# In package imports
from models.utils import avg_forwardlike, recursive_convweights

# Package imports
import copy
import numpy as np
import pdb 

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# Snippets of code from ViT's pytorch implementation
# Found on:
# https://github.com/lucidrains/vit-pytorch/blob/c2aab05ebfb01b9740ba4dae5b515fce1140e97d/vit_pytorch/vit.py#L67



# ========================================================================
def _layer_init(layer):
    try:
        for m in layer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    except TypeError:
        pass

# ========================================================================
class CLS_Attention(nn.Module):
    def __init__(self, dim, heads: int = 8, dim_head: int = 64,
                 project_out: bool = False, dropout: float = 0.):

        super().__init__()
        self.heads = heads # Used for MSHA
        self.scale = dim_head ** -0.5 # Scaling Transf, 1/sqrt(dim)
        self.attend = nn.Softmax(dim=-1) # Softmax(QKt)
        self.dropout = nn.Dropout(dropout) # Dropout for regularization
    
        self.to_out = nn.Sequential(nn.Flatten(1, -1),
                                    nn.Linear(dim_head, dim),
                                    nn.Dropout(dropout))\
                                    if project_out else nn.Identity()


        # Add regularization as seen fit.
        try:
            for m in self.to_out:
                if isinstance(m, nn.Linear):
                    #nn.init.xavier_normal_(m.weight)
                    nn.init.xavier_uniform_(m.weight)
                    #nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        except TypeError:
            pass # When doing no projection at all

    def forward(self, feats, qcls):
        b, c, w, h = feats.shape
        feats = feats.view(b, c, w*h)
        qcls = qcls.view(b, 1, c)
        qcls = rearrange(qcls, 'b n (h d) -> b h n d', h=self.heads)
        feats = rearrange(feats, 'b (h d) n -> b h n d', h = self.heads)
        dots = torch.matmul(qcls, feats.transpose(-1, -2)) * self.scale # SA
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, feats) #+ qcls
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return dots.view(b, self.heads, w, h), out



