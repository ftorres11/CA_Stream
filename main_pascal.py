# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.dataloader import default_collate
from  torch.nn.parallel import DistributedDataParallel as DDP

import torchvision

# Timm imports
from timm.models import convert_sync_batchnorm
from timm.optim import Lamb

# In package imports
from models.utils import weights_loader
from models import (backbone_freezer, convnext_scrapper,
                    efficient_freezer, mobilenet_scrapper, 
                    resnet_scrapper, resnet_tuner, vit_scrapper) 
from lib.data import (dataset_wrapper, imagenet_trainer, imagenet_tester,
                      train_splitter, PascalClassifier)

import engine.ddp as ddp
from engine.utils import (distribute_bn, PascalOpt,
                          weight_retriever)
from engine.routines import (logger, simplified_trainer,
                             simplified_evaluator, 
                             routine_resumer)
from engine import transforms
# Package imports
import os
import pdb
import copy
osp = os.path
osj = osp.join

import json
import time
import argparse
import warnings
from functools import partial
warnings.filterwarnings("ignore")


# ========================================================================
# Basic Configuration Options
# Loss dictionary for different training losses
dict_losses = {#'BCELoss': BinaryCrossEntropy(target_threshold=0.2),
               'BCELoss': nn.BCELoss(),
               'BCELogitsLoss': nn.BCEWithLogitsLoss(),
               'CrossEntropy': nn.CrossEntropyLoss(),
               'KLDivLoss': nn.KLDivLoss()}

dict_recipes = {'resnet_orig': ResNetPytorchOpt,
                'vit_mod': ViTmodOpt}
# ========================================================================
def main():
    parser = argparse.ArgumentParser()
    # Training procedure settings
    parser.add_argument('--epochs', default=20, type=int,
                        help='Training iterations over the dataset')
    parser.add_argument('--criterion', default='CrossEntropy', type=str,
                        help='Loss Function to use during training')
    parser.add_argument('--print_interval', type=int, default=300,
                        help='Print results every X iterations')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from a certain checkpoint')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Use GPU acceleration for training?')
    parser.add_argument('--freeze', action='store_true', default=False,
                        help='Are we freezing the backbone?')
    parser.add_argument('--ftune', action='store_true', default=False,
                        help='Are we fine-tuning the Backbone?')

    # Data initialization procedure
    parser.add_argument('--root_data', default=None, type=str,
                        help='Root for Tiny ImageNet data')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='Training/Validation batch size')
    parser.add_argument('--imsize', default=64, type=int,
                        help='Input size for data')
    parser.add_argument('--crop_size', default=72, type=int,
                        help='Input size for cropping')
    parser.add_argument('--store_dir', default='Experiments/Baseline',
                        type=str, help='Where to store the experiment')
    parser.add_argument('--s_iterations', default=15, type=int,
                        help='Store intermediate results every X epochs')
    parser.add_argument('--process', default='plain', type=str,
                        help='How to process the data for loss')
    parser.add_argument('--fraction', default=1/6, type=float,
                        help='Fraction to use as  val split')
    parser.add_argument('--path_indexes', default=None, type=str,
                        help='Path to store indexes for train/val')
    parser.add_argument('--path_list', default=str, type=str,
                        help='Path to the list of imagenet data')
    parser.add_argument('--fixed', action='store_true', default=False,
                        help='Are we using a random seed?')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed to use')
    parser.add_argument('--full', action='store_true', default=False,
                        help='Using the full set to train?')

    # Model Settings
    parser.add_argument('--model', default='Baseline', type=str,
                        help='Which model are we training on')
    parser.add_argument('--classes', default=200, type=int,
                        help='Amount of classes the data has')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Sets to load a pretrained model')
    parser.add_argument('--pre_path', default='', type=str,
                        help='Pretrained Model Path')
    parser.add_argument('--interm_store', default=-1, type=int,
                        help='Steps until storing intermediate states')
    parser.add_argument('--project', action='store_true', default=False,
                        help='Projecting when out of SA?')
    parser.add_argument('--fixed_depth', action='store_true',
                        default=False,
                        help='Are we using a fixed stream depth?')
    parser.add_argument('--dropout', default=0., type=float,
                        help='Dropout for the transformer stream')
    parser.add_argument('--heads', type=int, default=1,
                        help='heads for multihead self attention')
    parser.add_argument('--shared', action='store_true', default=False,
                        help='Are we sharing weights with CNN to project?'
                        )
    parser.add_argument('--increased', action='store_true', default=False,
                        help='Are we increasing the learning rate of the '
                        'stream?')
    parser.add_argument('--recipe', type=str, default='resnet_orig',
                        help='Type of recipe to follow during training')

    # Parsing Arguments
    args = parser.parse_args()
    args.cpus_task = ddp.cpus_per_task
    # ====================================================================
    # Checking the parsed arguments
    if not osp.exists(args.store_dir) and ddp.rank == 0:
        os.makedirs(args.store_dir)

    try:
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=ddp.size, rank=ddp.rank)

    except:
        dist.init_process_group(backend='gloo', init_method='env://',
                                world_size=ddp.size, rank=ddp.rank)

    # Check for gpu acceleration
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(ddp.local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        args.device = torch.device('cuda:%d' % ddp.local_rank)
    else:
        args.device = torch.device('cpu')

    # Splitting Train into Train-Val
    with open(args.path_list, 'r') as data:
        dataset = json.load(json_file)
    # If the path of indexes exists, it loads the training and validation
    # split data; if not, it creates it under the name of sorted_set.txt
    if args.path_indexes:
        with open(args.path_indexes, 'r') as f:
            dict_splits = json.load(f)
        train_indexes = dict_splits['train']
        val_indexes = dict_splits['val']

    else:
        if ddp.rank == 0 and not args.full:
            print('Splitting data into Train & Val; total data {}'.format(\
                  len(dataset)))
        [train_indexes, val_indexes] = train_splitter(dataset,
                                                      args.fraction,
                                                      args.seed)
        if ddp.rank == 0 and not args.full:
            print('Split data into train {}, and val {}'.format(\
                  len(train_indexes), len(val_indexes)))

        dict_splits = {'train': train_indexes, 'val': val_indexes}
        #with open(osj(args.store_dir, 'sorted_set.txt'), 'w') as output:
        #    json.dump(dict_splits, output)

    # Generates the dataloader for Tiny ImageNet with the split indexes
    if args.full:
        train_data = PascalClassifier(args.root_data, dataset,
                                  [*range(len(dataset))],
                                  imagenet_trainer(args.imsize,
                                  args.crop_size))
        if ddp.rank == 0:
            print ('Using the entirety of the training set. '
                   'Total images {}'.format(len(dataset)))
    else:
        train_data = PascaClassifier(args.root_data, dataset,
                                     train_indexes,
                                     imagenet_trainer(args.imsize))
        val_data = PascalClassifier(args.root_data, dataset, val_indexes,
                                    imagenet_tester(args.crop_size))

    # Sets the number of batches given the length of the dataloader
    # If loading from batched data does not divide by the batch siz
    if args.batch_size:
        args.batches_train = len(train_data)//(args.batch_size)
        if not args.full:
            args.batches_val = len(val_data)//(args.batch_size)
    else:
        args.batches_train = len(train_data)
        if not args.full:
            args.batches_val = len(val_data)

    # ====================================================================
    # Loads the model with the correct architecture
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
                                          dropout=args.dropout,
                                          heads=args.heads,
                                          shared=args.shared)
    if 'vit' in args.model:
        net = vit_scrapper(args.model)()

    if 'vgg' in args.model:
        net = dict_vgg[args.model](pretrained=True,
                                   num_classes=args.classes)

    net = convert_sync_batchnorm(net)
    net = net.to(args.device)
    criterion = dict_losses[args.criterion]
    # ====================================================================
    last_epoch = 0
    args.min_val = 1e20
    
    # Generating the storing directories for the optim and intermediate
    # epochs
    if not osp.exists(args.store_dir):
        os.makedirs(args.store_dir)

    args.tr_iter = 0
    args.vl_iter = 0
    # ====================================================================
    # Choosing parameters to optimize:
    if args.ftune or args.increased:
        bbone_params, cls_params = resnet_tuner(net)
    elif args.freeze:
        params = []
        backbone_freezer(net, params)
        params = [{'params': param} for param in params]
    else:
        params = net.parameters()
    
    args.params = params
    # Choosing from a specific recipe
    recipe = dict_recipes[args.recipe](params)
    # Checking for Gradient Clipping
    args.clip_grad = recipe.clip_grad

    # Checking for Label Smoothing
    if recipe.label_smoothing != 0:
        args.labelsmooth=True
        try:
            criterion.label_smooth = recipe.label_smoothing
        except AttributeError:
            pass

    # Checking for Warm Up
    if recipe.warm_iter:
        args.warmup_iter = recipe.warm_iter
    
    # Checking for Exponential Moving Average
    # From https://github.com/pytorch/vision/blob/main/references/classification/train.py
    if recipe.ema:
        # model-ema steps = 32
        args.ema_steps = 32
        adjust = ddp.size*args.batch_size*args.ema_steps/args.epochs
        model_ema_decay = 0.99998
        alpha = 1-model_ema_decay
        alpha = min(1.0, alpha*adjust)
        args.ema = recipe.ema(net, decay=1.0 - alpha) 
    else:
        args.ema = None
    
    optimizer = recipe.optimizer
    # Loading weights
    if args.pretrained:
        weight_retriever(args.pre_path, net)

    # If resuming the training, load the previous optimizer, models,
    # epochs, write on a small logger
    args.idx = -1
    args.current_batch = 0

    # If loading from a pretrained model, load the weights
    if args.resume and ddp.rank == 0:
        checkpoint = routine_resumer(args.store_dir)

        mode = checkpoint['mode']
        state_model = checkpoint['model_sdict']
        state_optim = checkpoint['optim_sdict']
        last_epoch = checkpoint['epoch']
        current_batch = checkpoint['batch']
        args.min_val = checkpoint['loss']

        epoch_iter_tr = len(train_data)//(args.batch_size*ddp.size)
        epoch_iter_val = len(val_data)//(args.batch_size*ddp.size)

        if mode == 'batch':
            args.tr_iter = last_epoch*(epoch_iter_tr+current_batch+1)
            args.val_iter = epoch_iter_val*last_epoch-1
            args.current_batch = current_batch+1
            if ddp.local_rank == 0:
                print('Resuming training at epoch {}, batch {}'.format(\
                      epoch, current_batch))
                
        if mode == 'epoch':
            args.tr_iter = epoch_iter_tr*last_epoch
            args.val_iter = epoch_iter_val*last_epoch-1
            if ddp.local_rank == 0:
                print('Resuming training at the begining of epoch'
                      ' {}'.format(last_epoch))
        
        net.load_state_dict(state_model)
        optimizer.load_state_dict(state_optim)

    # Moving to GPU 
    net = DDP(net, device_ids=[ddp.local_rank])

    # Learning rate scheduler dictionary
    scheduler = recipe.scheduler
    log = logger(args.store_dir, last_epoch)
    args.logger = log

    if args.resume and recipe.scheduling == 'multistep':
        recipe.scheduler.step(last_epoch)

    # Last Data preparations: Mixup - Cutmix
    mixup_transforms = []
    collate_fn = None
    if recipe.mixup:
        mixup_transforms.append(transforms.RandomMixup(args.classes,
            p=1.0, alpha=recipe.mixup_alpha))
        mixup_transforms.append(transforms.RandomCutmix(args.classes,
            p=1.0, alpha=recipe.cutmix_alpha))
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    # Wrapping inside dataloader
    train_data, train_sampler = dataset_wrapper(train_data, ddp.size,
                                                ddp.rank, collate_fn,
                                                args)
    if not args.full:
        val_data, val_sampler = dataset_wrapper(val_data, ddp.size,
                                                ddp.rank, args)

    # ====================================================================
    # Training loop
    for epoch in range(last_epoch+1, args.epochs+1):
        args.epoch = epoch
        if ddp.rank == 0:
            print('====== Starting Epoch: {}/{} --> {} ======'.format(epoch,
                  args.epochs, time.strftime("%H:%M:%S")))
        for param_group in optimizer.param_groups:
            if ddp.rank == 0:
                print('Learning Rate: {:.5f}'.format(param_group['lr']))

        # Trainining
        if train_sampler:
            train_sampler.set_epoch(epoch)
        loss_train, acc_train = simplified_trainer(train_data, net,
                                                   criterion, optimizer,
                                                   args)

        distribute_bn(net, ddp.size, True)
        if not args.full:
            if val_sampler:
                val_sampler.set_epoch(epoch)
            loss_val, acc_val = simplified_evaluator(val_data,
                                                     net, criterion, args)

        scheduler.step()
        # Saving the model with the lowest loss
        if not args.full:
            if loss_val < args.min_val:
                args.min_val = loss_val
                named = osj(args.store_dir, 'best_model.pth')
                if ddp.rank == 0:
                    print('Best model saved')
                    torch.save(net.state_dict(), named)

        # Saving every given number of epochs
        if epoch % args.s_iterations == 0 and ddp.rank == 0:
            model_check = osj(args.store_dir, 'checkpoint.pth')

            checkpoint_dict = {'mode': 'epoch',
                               'model_sdict': net.state_dict(),
                               'optim_sdict': optimizer.state_dict(),
                               'epoch': epoch,
                               'batch': 0,
                               'loss': args.min_val}

            torch.save(checkpoint_dict, model_check)
            print('Checkpoint saved')

        if ddp.rank == 0:
            args.logger.update('Epoch Loss: Training', loss_train, epoch)
            args.logger.update('Epoch Accuracy: Training', acc_train, epoch)

            if not args.full:
                args.logger.update('Epoch Loss: Validation', loss_val, epoch)
                args.logger.update('Epoch Accuracy: Validation', acc_val, epoch)

            args.logger.flush()
    
    if ddp.rank == 0:
        args.logger.close()
        model_final = osj(args.store_dir, 'final_epoch.pth')
        torch.save(net.module.state_dict(), model_final)
        print('Final model saved')

if __name__ == '__main__':
    main()
