# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch imports
import torch
from torchvision import transforms
from torchsummary import summary

# In Package Imports
from models.vgg import dict_vgg
from models import convnext_scrapper, resnet_scrapper

from engine.evaluation import recognition_evaluator
from models.interpretable import Activation_Wrapper, cam_targetter

from lib.data import INet_Evaluator, imagenet_tester
from lib.utils import dataset_partition

# Package Imports
import os
osp = os.path
osj = os.path.join

from captum.attr import LRP

import argparse
import numpy as np

import pdb
import warnings
warnings.filterwarnings("ignore")

# ========================================================================
# Basic Configuration Data
transform = imagenet_tester(224)

# ========================================================================
def main():
    parser = argparse.ArgumentParser()

    # Activation procedure
    parser.add_argument('--state_root', default=None, type=str,
                        help='Path of the model tested')
    parser.add_argument('--store_dir', default='Evaluation/',
                        help='Storing path for the experiments')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Using GPU acceleration?')

    # Data initialization procedures
    parser.add_argument('--root_data', default=None, type=str,
                        help='Root for Pascal data')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Amount of images per batch')
    parser.add_argument('--sorted_path', default=None, type=str,
                        help='Path for sorted activations') # Remove    

    # Model Settings
    parser.add_argument('--classes', default=200, type=int,
                        help='How many classes the model has')
    parser.add_argument('--model', default='Baseline', type=str,
                        help='Different kind of models to load')
    parser.add_argument('--project', action='store_true', default=False,
                        help='Projecting when out of SA?')
    parser.add_argument('--salient', action='store_true', default=False,
                        help='Store Saliency Maps?')
    parser.add_argument('--fixed_depth', action='store_true', default=False,
                        help='Using fixed stream depth?')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='Which kind of attention do we want?')
    parser.add_argument('--soften', action='store_true', default=False,
                        help='Do we want to soften the saliency maps?')
    parser.add_argument('--heads', type=int, default=1,
                        help='Heads for multihead self attention')
    parser.add_argument('--reduction', type=str, default='mean',
                        help='Reduction for MHSA')
    parser.add_argument('--shared', action='store_true', default=False,
                        help='Are we sharing weights with the CNN to project?'
                        )
    parser.add_argument('--alpha', type=float, default=None,
                        help='Alpha coefficient for mixing raw att with gradcam')

    # Parsing Arguments
    args = parser.parse_args()
    args_dict = vars(args)
    args.multilabel = False
    suffix = 'softened' if args.soften else ''
    suf_alpha = '_a{}'.format(args.alpha) if args.alpha else ''
    sstate = args.state_root.split('/')[-1]
    args.imset = 'imagenet'
    args.path_salient = osj('SaliencyMaps', args.state_root.replace(\
                            sstate, ''), '{}{}{}'.format(\
                            args.method, suffix, suf_alpha))

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
    with open(args.sorted_path, 'r') as data:
        dataset = data.readlines()
    
    experimental_dataset = INet_Evaluator(args.root_data, dataset,
                                          transform)

    if 'convnext' in args.model:
        net = convnext_scrapper(args.model)()

    if 'efficientnet' in args.model:
        net = efficient_scrapper(args.model)()

    if 'mobilenet' in args.model:
        net = mobilenet_scrapper(args.model)()

    if 'resnet' in args.model:
        net = resnet_scrapper(args.model)(pretrained=False,
                                          num_classes=args.classes,
                                          fixed_dim=args.fixed_depth,
                                          project=args.project,
                                          heads=args.heads,
                                          shared=args.shared)

    if 'vit' in args.model:
        net = vit_scrapper(args.model)()

    if 'vgg' in args.model:
        net = dict_vgg[args.model](num_classes=args.classes,
                                   pretrained=True)

    # ====================================================================
    if args.state_root:
        state_model = torch.load(args.state_root,
                                 map_location=lambda storage,
                                 loc: storage)
        
        if 'model_sdict' in state_model.keys():
            state_model = state_model['model_sdict']

        loaded_keys = state_model.keys()
        base_sdict = net.state_dict()
        base_keys = base_sdict.keys()
        # Loading weights
        for key in base_sdict:
            try:
                base_sdict[key] = state_model[key]
                print('Loaded key {} correctly'.format(key))
            except KeyError:
                try:
                    mod_key = 'module.base_model.'+key
                    base_sdict[key] = state_model[mod_key]
                    print('Loaded key {} correctly'.format(key))
                except KeyError:
                    print('Error loading key {}'.format(key))
                    continue

        net.load_state_dict(base_sdict)

    net = net.to(args.device)
    net.eval()
    # ====================================================================
    # Targetting layers to retrieve attention
    if 'cam' in args.method:
        target = cam_targetter(net)
    else:
        target = transformer_targetter(net, args.model)
        
    wrapper = Activation_Wrapper(net, target)

    # Accounting for number of parameters
    params = 0
    for param in net.parameters():
        if param.requires_grad == True:
            params += param.nelement()
    print('Number of parameters: {}'.format(params))
    # ====================================================================
    recognition_evaluator(wrapper, experimental_dataset, args)

if __name__ == '__main__':
    main()
