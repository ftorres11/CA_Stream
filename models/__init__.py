# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr
import pdb

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
from .ResNet.resnet_v1 import dict_resnet_v1
from .ResNet.resnet_v2 import dict_resnet_v2
from .ResNet.resnet_v3 import dict_resnet_v3
from .ResNet.resnet_v4 import dict_resnet_v4

from .ResNet.cls_v1 import dict_clsv1
from .ResNet.cls_v2 import dict_clsv2
from .ResNet.cls_v3 import dict_clsv3
from .ResNet.cls_v4 import dict_clsv4

dict_variants = {'v1': dict_resnet_v1, 'v2': dict_resnet_v2,
                 'v3': dict_resnet_v3, 'v4': dict_resnet_v4}

dict_cls = {'clsv1': dict_clsv1, 'clsv2': dict_clsv2,
            'clsv3': dict_clsv3, 'clsv4': dict_clsv4}

# ResNet Functions
def resnet_scrapper(model_name):
    model_variant = model_name.strip().split('_')[-1]
    if 'cls' in model_variant:
        return dict_cls[model_variant][model_name]
    elif 'v' in model_variant:
        return dict_variants[model_variant][model_name]
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
from .EfficientNet.cls_v1 import dict_efficient_clsv1
from .EfficientNet.v1 import dict_efficient_v1

# EfficientNet Functions
def efficient_scrapper(model_name):
    model_variant = model_name.strip().split('_')[-1]
    if 'cls' in model_variant:
        return dict_efficient_clsv1[model_name]
    elif 'v' in model_variant:
        return dict_efficient_v1[model_name]
    else:
        return dict_efficientnet[model_name]

# ========================================================================
# MobileNet
from .MobileNet.mobilenet_v2 import mobilenet_v2
from .MobileNet.mbv2_clsv1 import mobilenet_v2_clsv1
from .MobileNet.mbv2_v3 import mobilenet_v2_v3

# MobileNet Functions
def mobilenet_scrapper(model_name):
    model_variant = model_name.strip().split('_')[-1]
    if 'cls' in model_variant:
        return mobilenet_v2_clsv1
    elif 'v3' in model_variant:
        return mobilenet_v2_v3
    else:
        return mobilenet_v2

# ========================================================================
# ConvNext
from .ConvNext.convnext import dict_convnext
from .ConvNext.cls_v1 import dict_clsv1
from .ConvNext.v3 import dict_v3

# ConvNext Functions
def convnext_scrapper(model_name):
    if 'cls' in model_name:
        return dict_clsv1[model_name]
    elif 'v3' in model_name:
        return dict_v3[model_name]
    else:
        return dict_convnext[model_name]

# ========================================================================
# ViT
from .ViT.vit import dict_vit
from .ViT.vit_clsv1 import dict_vit_cls
from .ViT.vit_v3 import dict_vit_v3

# ViT Functions
def vit_scrapper(model_name):
    if 'cls' in model_name:
        return dict_vit_cls[model_name]
    elif 'v3' in model_name:
        return dict_vit_v3[model_name]
    else:
        return dict_vit[model_name]
