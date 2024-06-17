# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa
# felipe.torres@lis-lab.fr - felitf.94@gmail.com
# Modifications and additions added to each function//class with their sources

# Torch imports
import torch
from torch import distributed as dist
from torch.optim.lr_scheduler import _LRScheduler

# In Package imports
import engine.ddp as ddp

# Package imports
import pdb
import numpy as np
import sys
epsilon = sys.float_info.epsilon

# ========================================================================
def weight_retriever(path, model):
    state_model = torch.load(path, map_location=lambda storage,
                             loc:storage)
    net_sdic = model.state_dict()
    net_names = list(net_sdic)
    names = state_model.keys()
    for key in names:
        try:
            if net_sdic[key].shape == state_model[key].shape:
                net_sdic[key] = state_model[key]
                if ddp.rank == 0:
                     print('Loaded key {}'. format(key))
        except:
            print('State dict key {} failed to be loaded'.format(key))
            continue
    model.load_state_dict(net_sdic)

# ========================================================================
def gradient_generation(logits, labels=None):
    one_hot = np.zeros(logits.size(), dtype=np.float32)
    if labels == None:
        labels = torch.argmax(logits, 1)

    for x in range(logits.shape[0]):
        one_hot[x][int(labels[x])] = 1
    one_hot = torch.from_numpy(one_hot).type_as(labels)
    one_hot = one_hot.type_as(labels)*logits
    one_hot = one_hot[torch.where(one_hot!=0)]
    one_hot.requires_grad_(True)
    return one_hot

# ========================================================================
def distribute_bn(model, world_size, reduce=False):
    # ensures every node has the same running bn stats
    for bn_name, bn_buf in model.named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # Average bn stats across the whole group
                torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            else:
                # Broadcasting bn stats from rank 0 to whole group
                dist.broadcast(bn_buf, 0)

# ========================================================================
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    def __init__(self, model, decay):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay*avg_model_param + (1-decay)*model_param
        super().__init__(model, ddp.local_rank, ema_avg, use_buffers=True)

# ========================================================================
class WarmUpLR(_LRScheduler):
    """
    Warm up Training learning rate scheduler:
    Args:
        optimizer: optimizer (i.e. SGD, Adam)
        total_iters: total iterations for warmup

    **Code taken from: https://github.com/weiaicunzai/pytorch-cifar100/**
    Fixes and optimizations on this version if needed
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Set the Learning rate to base_lr * m / total_iters
        """
        return [base_lr*self.last_epoch/(self.total_iters+epsilon)\
                for base_lr in self.base_lrs]

# ========================================================================
class ResNetPytorchOpt:
    def __init__(self, params, lr=0.1):
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9,
                             weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(\
                             self.optimizer, milestones=[30, 60, 90], 
                             gamma=0.1)
        self.ema = None
        self.warmup = None
        self.warm_iter = None
        self.scheduling = 'multistep'

        self.mixup = False
        self.label_smoothing = None
        self.clip_grad = None

# ========================================================================
class ViTmodOpt:
    def __init__(self, params, lr=9e-3):
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.3)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(\
                             self.optimizer, T_max=100)
        #self.warmup = None
        self.warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                         start_factor=9e-3, total_iters=10)
        self.warm_iter = 10
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer,
                schedulers=[self.warmup, self.scheduler],
                milestones=[self.warm_iter])

        #self.ema = None
        self.ema = ExponentialMovingAverage
        self.scheduling = 'cosineannealing'

        self.mixup = True
        self.mixup_alpha=0.2
        self.cutmix_alpha=1.0
        self.label_smoothing = 0.11
        self.clip_grad = False

# ========================================================================
class PascalOpt:
    def __init__(self, params, lr=1e-4):
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(\
                             self.optimizer, 12, eta_min=0)

        self.ema = None
        self.warmup = None
        self.warm_iter=None
        self.scheduling = 'multistep'

        self.mixup = False
        self.label_smoothing = None
        self.clip_grad = None

