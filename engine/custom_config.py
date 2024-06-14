# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch Imports
import torch
import torch.nn as nn
import torch.optim as optim

# In-package Imports

# Package Imports
import pdb
import copy

import json
import time
import arparse
import warnings
from functools import partial
warnings.filterwarnings("ignore")

# ========================================================================
class LinearWarmUP:
    def __init(self, optimizer, warmupsteps
# ========================================================================
class ResNetPytorchOptim:
    def __init__(params):
        self.optimizer = optim.SGD(params, lr=0.1, momentum=0.9,
                             weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.MultiStepLR(\
                             optimizer, milestones=[30, 60, 90], gamma=0.1)

# ========================================================================
class ViTmodOpt:
    def __init__(params):
        self.optimizer = optim.AdamW(params, lr=9e-3, weight_decay=0.3)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(\
                             optimizer, T_max=100)               
