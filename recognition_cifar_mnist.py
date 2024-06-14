# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch imports
import torch
from torchvision import transforms
from torchsummary import summary 
# In Package Imports
from models.vgg import dict_vgg
from models.CIFAR import resnet_scrapper
from engine.evaluation import recognition_evaluator
from models.interpretable import (Activation_Wrapper, cam_targetter,
                                  transformer_targetter,
                                  hybrid_targetter)

from lib.utils import SplitLoader
from lib.data import CIFAR10_val, CIFAR100_val, MNIST_val

# Package Imports
import os
osp = os.path
osj = os.path.join

import argparse
import numpy as np
import pdb

import warnings
warnings.filterwarnings("ignore")


# ========================================================================
# Basic Configuration Options
dict_imsizes = {'MNIST': 28,
                'CIFAR10': 32,
                'CIFAR100': 32}

dict_validation = {'MNIST': MNIST_val,
                   'CIFAR10': CIFAR10_val,
                   'CIFAR100': CIFAR100_val}

# ========================================================================
def main():
    parser = argparse.ArgumentParser()

    # Activation procedure
    parser.add_argument('--state_root', default='*.pth', type=str,
                        help='Path of the model tested')
    parser.add_argument('--store_dir', default='Evaluation/',
                        help='Storing path for the experiments')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Using GPU acceleration?')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='Dataset to test the model in')

    # Data initialization procedures
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Amount of images per batch')
    parser.add_argument('--sorted_path', default='.', type=str,
                        help='Path for storing sorted Ids')

    # Model Settings
    parser.add_argument('--classes', default=10, type=int,
                        help='How many classes the model has')
    parser.add_argument('--model', default='Baseline', type=str,
                        help='Different kind of models to load')
    parser.add_argument('--project', action='store_true', default=False,
                        help='Projecting when out of SA?')
    parser.add_argument('--asclassifier', action='store_true',
                        default=False,
                        help='Use the linear projection as classifier?')
    parser.add_argument('--mixed', action='store_true', default=False,
                        help='Are we mixing CLS with AvgPool?')
    parser.add_argument('--concat', action='store_true', default=False,
                        help='Are we concatenating CLS with AvgPool?')
    parser.add_argument('--leaky', action='store_true', default=False,
                        help='Are we leaking attention?')
    parser.add_argument('--salient', action='store_true', default=False,
                        help='Store Saliency Maps?')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='Which kind of attention do we want?')
    parser.add_argument('--fixed_depth', action='store_true',
                        default=False,
                        help='Using fixed stream depth?')

    # Parsing Arguments
    args = parser.parse_args()
    args_dict = vars(args)
    args.imsize = dict_imsizes[args.dataset]
    args.multilabel = False
    args.path_salient = osj('SaliencyMaps', args.state_root.replace(\
                            'best_model.pth', ''), args.method)
    # ====================================================================
    # Checking the parsed arguments
    if not osp.exists(args.store_dir):
        os.makedirs(args.store_dir)

    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    if args.salient:
        if not osp.exists(args.path_salient):
            os.makedirs(args.path_salient)

    # ====================================================================
    # Dataloading
    experimental_dataset = SplitLoader(dict_validation[args.dataset])

    if 'resnet' in args.model:
        net = resnet_scrapper(args.model)(pretrained=False,
                                          num_classes=args.classes,
                                          mixed=args.mixed,
                                          concat=args.concat,
                                          fixed_dim=args.fixed_depth,
                                          project=args.project)
    if 'vgg' in args.model:
        net = dict_vgg[args.model](num_classes=args.classes)

    state_model = torch.load(args.state_root,
                             map_location=lambda storage,
                             loc: storage)
    
    # Fix for loading with DPP
    loaded_keys = state_model.keys()
    base_sdict = net.state_dict()
    base_keys = base_sdict.keys()

    # Loading weights
    for key in base_sdict:
        try:
            base_sdict[key] = state_model[key]
        except KeyError:
            mod_key = 'module.'+key
            base_sdict[key] = state_model[mod_key]
    
    net.load_state_dict(base_sdict)
    net = net.to(args.device)
    net.eval() 
    # ====================================================================
    # Targetting layers to retrieve attention
    if 'gradcam' in args.method:
        target = cam_targetter(net)
    elif 'attention' in args.method:
        target = transformer_targetter(net, args.model)
    elif 'hybrid_cam' in args.method:
        target = hybrid_targetter(net, args.model, args)

    wrapper = Activation_Wrapper(net, target)

    print('number of parameters: {}'.format(\
          sum([param.nelement() for param in net.parameters()])))
    # ====================================================================
    recognition_evaluator(wrapper, experimental_dataset, args)

if __name__ == '__main__':
    main()
