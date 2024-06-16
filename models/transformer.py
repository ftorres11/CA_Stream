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
#                                    nn.LayerNorm(dim))\


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


# ========================================================================
class CLS_KV(nn.Module):
    '''
    '''
    def __init__(self, dim, heads: int = 8, dim_head: int =64,
                 project_out: bool = False, dropout: float = 0.,
                 ch_scaling: int = 4):

        super().__init__()
        inner_dim = dim_head * heads
        self.ch_scaling = ch_scaling
        self.to_k = nn.Sequential(\
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1,
            p2=1), nn.Linear(ch_scaling, inner_dim))
        self.to_v = nn.Sequential(\
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1,
            p2=1), nn.Linear(ch_scaling, inner_dim))

        #nn.Identity())
        self.heads = heads
        self.scale = dim_head ** -0.5 # Scaling Transf, 1/sqrt(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout)) \
                                    if project_out else nn.LayerNorm(dim)
        
        _layer_init(self.to_k)
        _layer_init(self.to_v)
        _layer_init(self.to_out)

    def forward(self, feats, qcls):
        b, c, wf , hf = feats.shape
        # Independent Linears for K and V
        k = rearrange(self.to_k(feats), 'b n (h d) -> b h n d',
                      h = self.heads)
        v = rearrange(self.to_v(feats), 'b n (h d) -> b h n d',
                      h = self.heads)
        b, h, n, d = k.shape
        qcls = qcls.view(b, h, 1, d)
        dots = torch.matmul(qcls, k.transpose(-1, -2)) * self.scale
        # Dots are different because of how we build the patches
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v) #+ qcls
        out = rearrange(out, 'b n h d -> b n (h d)')
        out = nn.Flatten(1, -1)(out)
        out = self.to_out(out)
        return dots.view(b, 1, wf, hf), out


# ========================================================================
class Light_Shared(nn.Module):
    '''
    '''
    def __init__(self, dim_head, heads, project_out: bool = False,
                 dropout: float=0.):
        super().__init__()
        self.project = project_out
        self.heads = heads # Used for MSHA
        self.scale = dim_head ** -0.5 # Scaling Transf, 1/sqrt(dim)
        self.attend = nn.Softmax(dim=-1) # Softmax(QKt)
        self.dropout = nn.Dropout(dropout) # Dropout for regularization
        self.project = project_out

       
    def forward(self, feats, qcls, proj_layer):
        b, c, w, h = feats.shape
        feats = feats.view(b, c, w*h)
        qcls = qcls.view(b, 1, c)
        qcls = rearrange(qcls, 'b n (h d) -> b h n d', h=self.heads)
        feats = rearrange(feats, 'b (h d) n -> b h n d', h = self.heads)
        dots = torch.matmul(qcls, feats.transpose(-1, -2)) * self.scale # SA
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, feats)
        out = out.view(b, c, 1, 1)

        if self.project:
            out = F.interpolate(out, (w,h))
            out = proj_layer(out)
            out = out.mean(dim=(-1,-2), keepdim=True)
            
        return dots.view(b, self.heads, w, h), out


# ========================================================================
class Conv_Shared(nn.Module):
    '''
    '''
    def __init__(self, dim_head, heads, project_out: bool = False,
                 dropout: float=0.):
        super().__init__()
        self.project = project_out
        self.heads = heads # Used for MSHA
        self.scale = dim_head ** -0.5 # Scaling Transf, 1/sqrt(dim)
        self.attend = nn.Softmax(dim=-1) # Softmax(QKt)
        self.dropout = nn.Dropout(dropout) # Dropout for regularization
        self.project = project_out

       
    def forward(self, feats, qcls, proj_layer):
        b, c, w, h = feats.shape
        feats = feats.view(b, c, w*h)
        qcls = qcls.view(b, 1, c)
        qcls = rearrange(qcls, 'b n (h d) -> b h n d', h=self.heads)
        feats = rearrange(feats, 'b (h d) n -> b h n d', h = self.heads)
        dots = torch.matmul(qcls, feats.transpose(-1, -2)) * self.scale # SA
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, feats)
        out = out.view(b, c, 1, 1)

        if self.project:
            weights = []
            recursive_convweights(proj_layer, weights)
            out = avg_forwardlike(weights, out)
            
        return dots.view(b, self.heads, w, h), out


# ========================================================================
class CLS_FeedForward(nn.Module):
    '''
    Creates an MLP to use atop of a Self Attention Block
    '''
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        dim = int(dim)
        hidden_dim = int(hidden_dim)
        self.mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout))

    def forward(self, x):
        return self.mlp(x)

