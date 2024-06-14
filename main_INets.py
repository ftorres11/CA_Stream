# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from  torch.nn.parallel import DistributedDataParallel as DPP

# Timm imports
from timm import utils
from timm.data import Mixup

#from timm.loss import BinaryCrossEntropy
from timm.optim import Lamb
from timm.scheduler import CosineLRScheduler

# In package imports
from models.utils import weights_loader
from models import resnet_scrapper
from lib.data import (dataset_wrapper, imagenet_trainer, imagenet_tester,
                      train_splitter, INet_Trainer)

import engine.dpp as dpp
from engine.utils import WarmUpLR
from engine.routines import (logger, simplified_trainer,
                             simplified_evaluator, 
                             routine_resumer)

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
warnings.filterwarnings("ignore")


# ========================================================================
# Basic Configuration Options
# Loss dictionary for different training losses
dict_losses = {#'BCELoss': BinaryCrossEntropy,
               'BCELoss': nn.BCELoss(), 
               'BCELogitsLoss': nn.BCEWithLogitsLoss(),
               'CrossEntropy': nn.CrossEntropyLoss(),
               'KLDivLoss': nn.KLDivLoss()}

# Optimizer dictionary
dict_optimizers = {'Adam': optim.Adam,
                   'SGD': optim.SGD,
                   'Lamb': Lamb}


# ========================================================================
def main():
    parser = argparse.ArgumentParser()
    # Training procedure settings
    parser.add_argument('--epochs', default=20, type=int,
                        help='Training iterations over the dataset')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Initial learning rate')
    parser.add_argument('--optimizer', default='Adam', type=str,
                        help='Optimizer for the training procedure')
    parser.add_argument('--criterion', default='CrossEntropy', type=str,
                        help='Loss Function to use during training')
    parser.add_argument('--print_interval', type=int, default=300,
                        help='Print results every X iterations')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from a certain checkpoint')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Use GPU acceleration for training?')
    parser.add_argument('--scheduler', default='plateau', type=str,
                        help='Which learning rate scheduler to use')
    parser.add_argument('--warm', default=0, type=int,
                        help='Iterations for warming up')
    parser.add_argument('--mixup', action='store_true',  default=False,
                        help='Allowing BCE with Mixup/CutMix?')

    # Data initialization procedure
    parser.add_argument('--root_data', default=None, type=str,
                        help='Root for Tiny ImageNet data')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='Training/Validation batch size')
    parser.add_argument('--imsize', default=64, type=int,
                        help='Input size for data')
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

    # Model Settings
    parser.add_argument('--model', default='Baseline', type=str,
                        help='Which model are we training on')
    parser.add_argument('--classes', default=200, type=int,
                        help='Amount of classes the data has')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Sets to load a pretrained model')
    parser.add_argument('--pre_path', default='', type=str,
                        help='Pretrained Model Path')
    parser.add_argument('--multilabel', action='store_true', default=False,
                        help='Multilabel Setting?')
    parser.add_argument('--interm_store', default=-1, type=int,
                        help='Steps until storing intermediate states')
    parser.add_argument('--project', action='store_true', default=False,
                        help='Projecting when out of SA?')
    parser.add_argument('--mixed', action='store_true', default=False,
                        help='Mixing CLS token and avg pooled features?')
    parser.add_argument('--concat', action='store_true', default=False,
                        help='Concatenate CLS tokens to avg pooled feats')
    parser.add_argument('--fixed_depth', action='store_true',
                        default=False,
                        help='Are we using a fixed stream depth?')

    # Parsing Arguments
    args = parser.parse_args()
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=dpp.size, rank=dpp.rank)
    # ====================================================================
    # Checking the parsed arguments
    if not osp.exists(args.store_dir) and dpp.rank == 0:
        os.makedirs(args.store_dir)

    # Check for gpu acceleration
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(dpp.local_rank)
    else:
        args.device = torch.device('cpu')

    # Splitting Train into Train-Val
    with open(args.path_list, 'r') as data:
        dataset = data.readlines()
    # If the path of indexes exists, it loads the training and validation
    # split data; if not, it creates it under the name of sorted_set.txt
    if args.path_indexes:
        with open(args.path_indexes, 'r') as f:
            dict_splits = json.load(f)
        train_indexes = dict_splits['train']
        val_indexes = dict_splits['val']

    else:
        if dpp.local_rank == 0:
            print('Splitting data into Train & Val; total data {}'.format(\
                  len(dataset)))
        [train_indexes, val_indexes] = train_splitter(dataset,
                                                      args.fraction)
        if dpp.local_rank == 0:
            print('Split data into train {}, and val {}'.format(\
                  len(train_indexes), len(val_indexes)))

        dict_splits = {'train': train_indexes, 'val': val_indexes}
        with open(osj(args.store_dir, 'sorted_set.txt'), 'w') as output:
            json.dump(dict_splits, output)

    # Generates the dataloader for Tiny ImageNet with the split indexes
    train_data = INet_Trainer(args.root_data, dataset, train_indexes,
                              imagenet_trainer(args.imsize))
    val_data = INet_Trainer(args.root_data, dataset, val_indexes,
                            imagenet_tester(args.imsize))

    # Sets the number of batches given the length of the dataloader
    # If loading from batched data does not divide by the batch siz
    if args.batch_size:
        args.batches_train = len(train_data)//(args.batch_size)
        args.batches_val = len(val_data)//(args.batch_size)
    else:
        args.batches_train = len(train_data)
        args.batches_val = len(val_data)

    # Loads the model with the correct architecture
    if 'resnet' in args.model:
        net = resnet_scrapper(args.model)(pretrained=False,
                                          num_classes=args.classes,
                                          mixed=args.mixed,
                                          concat=args.concat,
                                          fixed_dim=args.fixed_depth)
                                  
    if 'vgg' in args.model:
        net = dict_vgg[args.model](pretrained=True,
                                   num_classes=args.classes)
        #net.classifier = nn.Linear(512, args.classes)

    criterion = dict_losses[args.criterion]
    # ====================================================================
    last_epoch = 0
    args.min_val = 1e20
    
    # Generating the storing directories for the optim and intermediate
    # epochs
    if not osp.exists(args.store_dir):
        os.makedirs(args.store_dir)
    if not osp.exists(osj(args.store_dir, 'optim')):
        os.makedirs(osj(args.store_dir, 'optim'))
        
    args.tr_iter = 0
    args.vl_iter = 0
    net = net.to(torch.device("cuda"))
    net = DPP(net, device_ids=[dpp.local_rank])

    # Choosing parameters to optimize:
    params = net.parameters()

    # Choosing the proper optimizer
    if args.optimizer == 'Adam':
        optimizer = dict_optimizers[args.optimizer](params,
                                                    args.lr)
    if args.optimizer == 'SGD':
        optimizer = dict_optimizers[args.optimizer](params,
                                                    args.lr, momentum=.9,
                                                    weight_decay=5e-4)
    if args.optimizer == 'Lamb':
        optimizer = dict_optimizers[args.optimizer](params,
                                                    args.lr,
                                                    weight_decay=0.02)
    
    # If loading from a pretrained model, load the weights
    if args.pretrained:
        state_model = torch.load(args.pre_path,
                                 map_location=lambda storage,
                                 loc: storage)
        net_sdic = net.state_dict()
        names = state_model.keys()

        for key in names:
            if net_sdic[key].shape == state_model[key].shape:
                net_sdic[key] = state_model[key]

        net.load_state_dict(net_sdic)

    # If resuming the training, load the previous optimizer, models,
    # epochs, write on a small logger
    args.idx = -1
    args.current_batch = 0
    if args.resume:
        routine_summary = routine_resumer(args.store_dir)
        l_model = routine_summary[0]
        l_optim = routine_summary[1]
        last_epoch = routine_summary[2]
        last_epoch = last_epoch-1

        epoch_iter_tr = len(train_data)//(args.batch_size)
        epoch_iter_val = len(val_data)//(args.batch_size)

        try:
            current_batch = routine_summary[3]
            total_batches = routine_summary[4]
            args.tr_iter = last_epoch*(total_batches+current_batch+1)
            args.val_iter = epoch_iter_val*(last_epoch-1)
            args.current_batch = current_batch+1
            if dpp.local_rank == 0:
                print('Resuming training at epoch {}, batch {}'.format(\
                      last_epoch, current_batch))
        except IndexError:
            args.tr_iter = epoch_iter_tr*last_epoch
            args.val_iter = epoch_iter_val*(last_epoch-1)
            if dpp.local_rank == 0:
                print('Resuming training at the begining of epoch'
                      ' {}'.format(last_epoch))

        state_model = torch.load(l_model,
                                 map_location=lambda storage,
                                 loc:storage)
        state_optim = torch.load(l_optim,
                                 map_location=lambda storage,
                                 loc:storage)

        net.load_state_dict(state_model)
        optimizer.load_state_dict(state_optim)

    # Learning rate scheduler dictionary
    dict_schedulers={'plateau': optim.lr_scheduler.ReduceLROnPlateau(\
                   optimizer, factor=.8, patience=4, cooldown=5,
                   min_lr=1e-10),
                   'multistep': optim.lr_scheduler.MultiStepLR(optimizer,
                   milestones=[60, 120, 160], gamma=0.2),
                   'cosine': CosineLRScheduler(optimizer,
                                               t_initial=args.epochs,
                                               warmup_t=args.warm,
                                               k_decay=8e-3)
                       }
    log = logger(args.store_dir, last_epoch)
    args.logger = log
    scheduling = copy.deepcopy(args.scheduler)
    args.scheduler = dict_schedulers[args.scheduler]

    # Wrapping inside dataloader
    train_data = dataset_wrapper(train_data, dpp.size, dpp.rank)
    val_data = dataset_wrapper(val_data, dpp.size, dpp.rank)

    # ====================================================================
    # Extra. Mixup - Cutmix
    if args.mixup:
        mixup_args = dict(
                mixup_alpha=0.1,
                cytmix_alpha=1.0,
                cutmix_minmax=None,
                prob=1.0,
                switch_prob=0.5,
                mode='batch',
                label_smoothing=False,
                num_classes=args.classes)

        args.mixup_fn = Mixup(**mixup_args)
    
    # Training loop
    for epoch in range(last_epoch+1, args.epochs+1):
        args.epoch = epoch
        if dpp.local_rank == 0:
            print('====== Starting Epoch: {}/{} --> {} ======'.format(epoch,
                  args.epochs, time.strftime("%H:%M:%S")))
        for param_group in optimizer.param_groups:
            if dpp.local_rank == 0:
                print('Learning Rate: {:.5f}'.format(param_group['lr']))

        # Trainining
        loss_train, acc_train = simplified_trainer(train_data, net,
                                                   criterion, optimizer,
                                                   args)

        loss_val, acc_val = simplified_evaluator(val_data,
                                                 net, criterion, args)

        if scheduling == 'cosine':
            args.scheduler.step(epoch)
        else:
            if epoch > args.warm:
                if 'multistep' in scheduling:
                    args.scheduler.step(loss_train)
                else:
                    args.scheduler.step(epoch)

        # Saving the model with the lowest loss
        if loss_val < args.min_val:
            args.min_val = loss_val
            named = osj(args.store_dir, 'best_model.pth')

            torch.save(net.state_dict(), named)
            if dpp.local_rank == 0:
                print('Best model saved')

        # Saving every given number of epochs
        if epoch % args.s_iterations == 0 and dpp.local_rank == 0:
            model_check = osj(args.store_dir,
                              'epoch_{}.pth'.format(epoch))
            optim_check = osj(args.store_dir, 'optim',
                              'optim_{}.pth'.format(epoch))

            torch.save(net.state_dict(), model_check)
            torch.save(optimizer.state_dict(), optim_check)
            print('Checkpoint saved')

        if dpp.rank == 0:
            args.logger.update('Epoch Loss: Training', loss_train, epoch)
            args.logger.update('Epoch Loss: Validation', loss_val, epoch)
            
            args.logger.update('Epoch Accuracy: Training', acc_train, epoch)
            args.logger.update('Epoch Accuracy: Validation', acc_val, epoch)
            args.logger.flush()
    
    if dpp.rank == 0:
        args.logger.close()

if __name__ == '__main__':
    main()
