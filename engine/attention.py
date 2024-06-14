# -*- coding: utf-8  -*- 
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# In Package imports
from .utils import gradient_generation
from lib.data import (CIFAR_normalization, CIFAR_denormalization,
                      im_denormalization, im_normalization)

# Package imports
import os
osp = os.path
osj = osp.join

import pdb
import sys
epsilon = sys.float_info.epsilon

from einops import rearrange

import cv2
import copy
import numpy as np


# ========================================================================
# Basic configuration options
# ========================================================================
dict_reduction = {'mean': torch.mean,
                  'max': torch.max,
                  'min': torch.min}

dict_norms = {'imagenet': {'norm': im_normalization,
                           'denorm': im_denormalization},
              'CIFAR': {'norm': CIFAR_normalization,
                        'denorm': CIFAR_denormalization}}
              #Add more normalization options for different datasets
# ========================================================================
def gradcam_weights(wrapper, scores):
    batch_weights = []
    #batch_gradients = [] # For diagnostics
    for idx, score in enumerate(scores):
        wrapper.model.zero_grad()
        score.backward(retain_graph=True)
        gradient = wrapper.gradients[-1][idx].detach()

        if 'vit' in str(type(wrapper.model)):
            gradient = gradient[1:, :]
            pe, dim = gradient.shape
            wh = int(pe**.5)
            gradient = gradient.reshape(wh, wh, dim)
            gradient = gradient.transpose(1, 2).transpose(0, 1)

        elif 'convnext' in str(type(wrapper.model)).lower():
            gradient = gradient.permute(2, 0, 1)
        
        #batch_gradients.append(gradient) # For diagnostics
        weight = gradient.mean(dim=(-1, -2), keepdim=True)
        batch_weights.append(weight)
        wrapper.gradients = []
    # For diagnostics
    #return torch.stack(batch_weights), torch.stack(batch_gradients)
    return torch.stack(batch_weights)

# ========================================================================
def gradcampp_weights(wrapper, features, scores):
    batch_weights = []
    batch_alphas = []
    for idx, score in enumerate(scores):
        wrapper.model.zero_grad()
        score.backward(retain_graph=True)
        gradient = wrapper.gradients[-1][idx].detach()

        if 'vit' in str(type(wrapper.model)):
            gradient = gradient[1:, :]
            pe, dim = gradient.shape
            wh = int(pe**.5)
            gradient = gradient.reshape(wh, wh, dim)
            gradient = gradient.transpose(1, 2).transpose(0, 1)

        elif 'convnext' in str(type(wrapper.model)).lower():
            gradient = gradient.permute(2, 0, 1)

        grad_p2 = gradient**2
        grad_p3 = grad_p2*gradient
        target = features[idx]

        # Equation 19
        sum_act = target.sum(dim=(-1,-2), keepdim=True)
        aij = grad_p2/(grad_p2*2 + sum_act*grad_p3 +epsilon)
        aij = torch.where(gradient!=0, aij,
                          torch.tensor(0.).to(aij.device))
        batch_alphas.append(aij)
        weights = F.relu(gradient)*aij
        batch_weights.append(weights.sum(dim=(-1,-2), keepdim=True))
        wrapper.gradients = []
    return torch.stack(batch_weights)
    # For Diagnostics
    #return torch.stack(batch_weights), torch.stack(batch_alphas)

# ========================================================================
def scorecam_weights(wrapper, features, score, inputs, args):
    ''' Score-CAM reimplementation from Jacobgil 
    https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/score_cam.py
    with some fixes to account for normalization'''
    with torch.no_grad():
        device = inputs.get_device()
        activation = features
        # Upsampling and normalizing features
        upsample = torch.nn.UpsamplingBilinear2d(inputs.shape[-2:])
        upsampled = upsample(activation).to(device)

        B, C, _, _ = upsampled.size()
        max_feat = upsampled.amax(dim=(-1,-2), keepdim=True)
        min_feat = upsampled.amin(dim=(-1,-2), keepdim=True)
        upsampled = (upsampled-min_feat)/(max_feat-min_feat+epsilon)
        
        # Getting the index of the desired class on the logits tensor
        index = args.labels
        probs = []
        
        # Iterating over the mask
        for idx in range(0, C, 128):
            mask = upsampled[:, idx:idx+128, :].transpose(0, 1)
            denormd = dict_norms[args.imset]['denorm'](inputs)
            mask = denormd*mask
            mask = dict_norms[args.imset]['norm'](mask)

            logits_sc = wrapper.model(mask)
            probs.append(F.softmax(logits_sc, dim=-1)[:, index].data)
        probs = torch.stack(probs)
        weights = probs.view(1, C, 1 ,1)
        return weights
        
# ========================================================================
# Activations and Attention
def cam_processing(wrapper, features, score, inputs, args):
    # Compatibility transformations for novel models.
    if 'vit' in str(type(wrapper.model)).lower():
        bs, pd, dim = features.shape
        wh = int(pd**.5)
        features = features[:, 1:, :].reshape(features.size(0),
                       wh, wh, dim)
        features = features.transpose(2,3).transpose(1,2)

    elif 'convnext' in str(type(wrapper.model)).lower():
        features = features.permute(0, 3, 1, 2)

    if args.method == 'gradcam':
        #cam_coeff, batch_grad = gradcam_weights(wrapper, score) # Diags
        cam_coeff = gradcam_weights(wrapper, score) 
    elif args.method == 'gradcampp':
        cam_coeff = gradcampp_weights(wrapper, features, score)
        #cam_coeff, batch_grad = gradcampp_weights(wrapper, features, score) 
    elif args.method == 'scorecam':
        cam_coeff = scorecam_weights(wrapper, features, score, inputs,
                                     args)
    cam = features*cam_coeff
    cam = cam.sum(dim=1, keepdim=True).clamp(min=0)
    c_min, c_max = cam.amin(dim=(-1,-2), keepdim=True),\
                   cam.amax(dim=(-1,-2), keepdim=True)
    cam = (cam-c_min)/(c_max-c_min+epsilon)
    cam = cam.squeeze(1)
    #return cam, cam_coeff, batch_grad # Diagnostics
    return cam
