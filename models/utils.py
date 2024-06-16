# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr


# Torch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# In package imports


# Package imports
import copy


# ========================================================================
def weights_loader(model, desired_weights):
    desired_keys = desired_weights.keys()
    model_state = model.state_dict()
    for key in model_state:
        if key in desired_keys:
            model_state[key] = desired_weights[key]
    model.load_state_dict(model_state)

# ========================================================================
def recursive_convweights(layer, weight_list):
    for idx, module in layer._modules.items():
        recursive_convweights(module, weight_list)
        if module.__class__.__name__ == 'Conv2d':
            weight_list.append(module.weight)

# ========================================================================
def avg_forwardlike(weight_list, features):
    identity = features

    for weight in weight_list:
        weight = weight.mean(dim=(-1,2), keepdim=True).detach()
        try:
            features = F.conv2d(features, weight)
        except RuntimeError:
            identity = F.conv2d(identity, weight)
            features += identity

    return features
