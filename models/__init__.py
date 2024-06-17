# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa
# Contact: felitf.94@gmail.com - felipe.torres@lis-lab.fr 

# ========================================================================
def backbone_freezer(model, cls_params):
    for name, param in model.named_parameters():
        if 'cls' in name:
            param.requires_grad = True
            cls_params.append(param)
        else:
            param.requires_grad = False

# ========================================================================
# ResNet imports
from .ResNet.resnet import dict_resnet
from .ResNet.resnet_final import dict_resnet_stream
from .ResNet.resnet_stream import dict_resnet_final

# ResNet Functions
def resnet_scrapper(model_name):
    model_variant = model_name.strip().split('_')[-1]
    if 'stream' in model_variant:
        return dict_resnet_stream[model_name]
    elif 'final' in model_variant:
        return dict_resnet_final[model_name]
    else:
        return dict_resnet[model_name]

def resnet_freezer(model, cls_params):
    for layer in model.children():
        if 'cls' not in layer.__class__.__name__.lower():
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True
            cls_params.append(layer.parameters())

def resnet_tuner(model):
    bbone_params = []
    cls_params = []
    for layer in model.children():
        if 'cls' not in layer.__class__.__name__.lower():
            bbone_params.append(layer.parameters())
        else:
            cls_params.append(layer.parameters())

    return bbone_params, cls_params

# ========================================================================
# EfficientNet
from .EfficientNet.efficientnet import dict_efficientnet
from .EfficientNet.efficientnet_stream import dict_efficientnet_stream
from .EfficientNet.efficientnet_final import dict_efficient_final

# EfficientNet Functions
def efficient_scrapper(model_name):
    model_variant = model_name.strip().split('_')[-1]
    if 'stream' in model_variant:
        return dict_efficient_stream[model_name]
    elif 'final' in model_variant:
        return dict_efficient_final[model_name]
    else:
        return dict_efficientnet[model_name]


# ========================================================================
# ConvNext
from .ConvNext.convnext import dict_convnext
from .ConvNext.covnext_stream import dict_convnext_stream
from .ConvNext.convnext_final import dict_convnext_final

# ConvNext Functions
def convnext_scrapper(model_name):
    if 'stream' in model_name:
        return dict_convnext_stream[model_name]
    elif 'final' in model_name:
        return dict_convnext_final[model_name]
    else:
        return dict_convnext[model_name]


