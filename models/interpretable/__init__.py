# -*- coding: utf-8 -*. 
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr
from .utils import Activation_Wrapper
import pdb

# Auctilliary function to get the correct layers for attention
def transformer_targetter(model, model_variant):
    name = str(type(model))
    # To do support for different architectures other than ResNet
    if 'ResNet' in name:
        if 'cls' in model_variant:
            try:
                target_list = [model.base, model.l1, model.l2, model.l3, model.l4]
            except AttributeError:
                print('Encountered shared approach//stream not going from'
                      'the start')
                target_list = [model.l2, model.l3, model.l4]
        else:
            target_list = [model.cls_surrogate]

    elif 'MobileNet' in name:
        target_list = [model.cls_stream[-1]]

    elif 'ConvNeXt' in name:
        target_list = [model.cls_stream[-1]]

    else:
        target_list = []

    return target_list

def cam_targetter(model):
    name = str(type(model))
    if 'resnet' in name.lower():
        target = [model.layer4[-1]]

    elif 'vit' in name.lower():
        target = [model.encoder.layers[-2]]

    elif 'convnext' in name.lower():
        target = [model.features[-1][-1].block[5]]

    else:
        target = [model.features[-1]]

    return target

def hybrid_targetter(model, args):
    name = str(type(model))       
    if 'ResNet' in name:
        # Fetching the last layers to get the weights of SA
        if 'cls' in args.model:
            target_list = [model.l3.to_out]
        elif 'v' in args.model:
            target_list = [model.cls_surrogate]
        gc_target = target_list + [model.layer4[-1]]

    return gc_target

