# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch imports
import torch
from torchvision import transforms

# In Package Imports
from models.vgg import dict_vgg
from models import (convnext_scrapper, efficient_scrapper,
                    mobilenet_scrapper, resnet_scrapper,
                    vit_scrapper)
from engine.evaluation import interpretable_recognition

from lib.data import (INet_Evaluator, imagenet_tester,
                      im_normalization, im_denormalization,
                      Salient_Evaluator)

from lib.utils import dataset_partition

# Package Imports
import os
osp = os.path
osj = os.path.join

import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# ========================================================================
# Basic Configuration Data
dict_paths = {'imagenet': osj('data', 'revisited_imagenet_2012_val.csv'),
              'tiny': osj('data', 'val_tiny_imagenet.txt')
             }

dict_roots = {'imagenet': osj('/data1', 'data', 'corpus',
              'imagenet_2012_cr','validation', 'val'),
              'tiny': osj('/data1','data', 'corpus', 'tiny-imagenet-200',
              'val')
              }

im_sizes = {'imagenet': 224,
            'tiny': 64}

dict_transforms = {'imagenet': imagenet_tester(224),
                   'tiny': imagenet_tester(64)}

norms = {'normalization': im_normalization,
         'denormalization': im_denormalization}
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
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Amount of images per batch')
    parser.add_argument('--fraction', default=1e-3, type=float,
                        help='Fraction of the dataset to store/generate')
    parser.add_argument('--sorted_path', default=None, type=str,
                        help='Path for sorted activations') # Remove    

    # Model Settings
    parser.add_argument('--classes', default=1000, type=int,
                        help='How many classes the model has')
    parser.add_argument('--model', default='Baseline', type=str,
                        help='Different kind of models to load')
    parser.add_argument('--project', action='store_true', default=False,
                        help='Projecting when out of SA?')
    parser.add_argument('--fixed_depth', action='store_true', default=False,
                        help='Using fixed stream depth?')
    parser.add_argument('--imset', default='imagenet', type=str,
                        help='Which imagenet set are we using?')
    parser.add_argument('--pred', action='store_true', default=False,
                        help='Evaluation for gt or pred labels?')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='Which kind of attention do we want?')
    parser.add_argument('--soften', action='store_true', default=False,
                        help='Do we want to soften the saliency maps?')
    parser.add_argument('--shared', action='store_true', default=False,
                        help='Are we sharing weights with the backbone?')

    # Parsing Arguments
    args = parser.parse_args()
    args_dict = vars(args)
    args.multilabel = False
    root_val = dict_roots[args.imset]
    transform = dict_transforms[args.imset]
    args.imsize = im_sizes[args.imset]
    path_images = dict_paths[args.imset]
    suffix = 'softened' if args.soften else ''
    true = 'prediction' if args.pred else 'groundtruth'
    args.path_salient = osj('SaliencyMaps', true,
                            args.state_root.replace(\
                            'best_model.pth', '').replace(\
                            'final_epoch.pth', ''), '{}{}'.format(\
                            args.method, suffix))
    args.store_dir = osj(args.store_dir,
                         '{}{}'.format(args.method, suffix))
    # ====================================================================
    # Checking the parsed arguments
    if not osp.exists(args.store_dir):
        os.makedirs(args.store_dir)

    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    # ====================================================================
    # Dataloading
    if args.sorted_path:
        path_images = args.sorted_path

    with open(path_images, 'r') as read: 
        pure_data = read.readlines()

    experimental_dataset = INet_Evaluator(root_val, pure_data, transform)
    experimental_dataset = Salient_Evaluator(experimental_dataset,
                                             args.path_salient)

    # ====================================================================
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
                                          shared=args.shared,
                                          project=args.project)

    if 'vit' in args.model:
        net = vit_scrapper(args.model)()

    if 'vgg' in args.model:
        net = dict_vgg[args.model](num_classes=args.classes,
                                   pretrained=True)



    if args.state_root:
        state_model = torch.load(args.state_root,
                                 map_location=lambda storage,
                                 loc: storage)
        
        loaded_keys = state_model.keys()
        base_sdict = net.state_dict()
        base_keys = base_sdict.keys()
        # Loading weights
        for key in base_sdict:
            try:
                base_sdict[key] = state_model[key]
            except KeyError:
                try:
                    mod_key = 'module.'+key
                    base_sdict[key] = state_model[mod_key]
                except KeyError:
                    continue

        net.load_state_dict(base_sdict)

    net = net.to(args.device)
    net.eval()

    # ====================================================================
    # Accounting for number of parameters
    params = 0
    for param in net.parameters():
        if param.requires_grad == True:
            params += param.nelement()
    print('Number of parameters: {}'.format(params))
    # ====================================================================
    # Normalization parameters
    norm_dict = norms
    # ====================================================================
    interpretable_recognition(net, experimental_dataset, norm_dict, args)

if __name__ == '__main__':
    main()
