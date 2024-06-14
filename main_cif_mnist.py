# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# In package imports
from models.utils import weights_loader
from models.CIFAR import resnet_scrapper
from lib.data import (dataset_wrapper, dict_datasets, split_MNIST,
                      train_splitter)

import engine.ddp as dpp
from engine.utils import WarmUpLR
from engine.routines import (logger, simplified_trainer,
                             simplified_evaluator, 
                             routine_resumer)
# Package imports
import os
osp = os.path
osj = osp.join
import copy
import pdb

import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")


# ========================================================================
# Basic Configuration Options
# Optimizer dictionary
dict_optimizers = {'Adam': optim.Adam,
                   'SGD': optim.SGD}

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
    parser.add_argument('--print_interval', type=int, default=300,
                        help='Print results every X iterations')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from a certain checkpoint')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Use GPU acceleration for training?')
    parser.add_argument('--scheduler', default='plateau', type=str,
                        help='Learning rate scheduler to use')
    parser.add_argument('--warm', default=0, type=int,
                        help='Iterations for warming up')

    # Data initialization procedure
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Training/Validation batch size')
    parser.add_argument('--workers', default=2, type=int,
                        help='Number of workers for dataloading')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='Dataset to train the model in')
    parser.add_argument('--store_dir', default='Experiments/Baseline',
                        type=str, help='Where to store the experiment')
    parser.add_argument('--s_iterations', default=15, type=int,
                        help='Store intermediate results every X epochs')
    parser.add_argument('--fraction', default=1/6, type=float,
                        help='Fraction to use as  val split')
    parser.add_argument('--path_indexes', default=None, type=str,
                        help='Path to store indexes for train/val')
    parser.add_argument('--fixed', action='store_true', default=False,
                        help='Are we using a fixed random seed?')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random Seed to use') 

    # Model Settings
    parser.add_argument('--model', default='Baseline', type=str,
                        help='Which model are we training on')
    parser.add_argument('--classes', default=10, type=int,
                        help='Amount of classes the data has')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Sets to load a pretrained model')
    parser.add_argument('--pre_path', default='', type=str,
                        help='Pretrained Model Path')
    parser.add_argument('--interm_store', default=-1, type=int,
                        help='Steps until storing intermediate states')
    parser.add_argument('--freezing', action='store_true', default=False,
                        help='Are we freezing the model?')
    parser.add_argument('--project', action='store_true', default=False,
                        help='Projecting when out of SA?')
    parser.add_argument('--mixed', action='store_true', default=False,
                        help='Mixing CLS token and avg pooled features?')
    parser.add_argument('--concat', action='store_true', default=False,
                        help='Concatenate features?')
    parser.add_argument('--fixed_depth', action='store_true',
                        default=False,
                        help='Are we using a fixed stream depth?')

    # Parsing Arguments
    args = parser.parse_args()
    # ====================================================================
    args.multilabel = False
    args.mixup=False
    args.freeze = False
    args.labelsmooth = False
    args.clip_grad = False
    args.ema = False

    try:
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=dpp.size, rank=dpp.rank)
    except ValueError:
        dist.init_process_group(backend='gloo', init_method='env://',
                                world_size=dpp.size, rank=dpp.rank)

    # Checking the parsed arguments
    if not osp.exists(args.store_dir) and dpp.rank == 0:
        os.makedirs(args.store_dir)

    # Check for gpu acceleration
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(dpp.local_rank)
    else:
        args.device = torch.device('cpu')

    # Splitting Train into Train-Val
    dataset = dict_datasets[args.dataset]
    [train_indexes, val_indexes] = train_splitter(dataset, args.fraction)
    
    # If the path of indexes exists, it loads the training and validation
    # split data; if not, it creates it under the name of sorted_set.txt
    if args.path_indexes:
        with open(args.path_indexes, 'r') as f:
            dict_splits = json.load(f)
        train_indexes = dict_splits['train']
        val_indexes = dict_splits['val']

    else:
        dict_splits = {'train': train_indexes, 'val': val_indexes}
        with open(osj(args.store_dir, 'sorted_set.txt'), 'w') as output:
            json.dump(dict_splits, output)
    
    # Generates the dataloader for MNIST/CIFAR with the split indexes
    train_data = split_MNIST(dataset, train_indexes)
    val_data = split_MNIST(dataset, val_indexes)

    # Loads the model with the correct architecture
    if 'resnet' in args.model:
        net = resnet_scrapper(args.model)(pretrained=False,
                                         num_classes=args.classes,
                                         mixed=args.mixed,
                                         concat=args.concat,
                                         fixed_dim=args.fixed_depth,
                                         project=args.project)
    if 'vgg' in args.model:
        net = dict_vgg[args.model](pretrained=False,
                                   num_classes=args.classes)
    
    if args.batch_size:
        args.batches_train = len(train_data)//(args.batch_size)
        args.batches_val = len(val_data)//(args.batch_size)
    else:
        args.batches_train = len(train_data)
        args.batches_val = len(val_data)
    
    net = net.to(torch.device("cuda"))
    criterion = nn.CrossEntropyLoss()
    
    # ====================================================================
    last_epoch = 0
    min_val = 1e20
    
    # Generating the storing directories for the optim and intermediate
    # epochs
    if not osp.exists(args.store_dir) and dpp.rank==0:
        os.makedirs(args.store_dir)
    if not osp.exists(osj(args.store_dir, 'optim')) and dpp.rank==0:
        os.makedirs(osj(args.store_dir, 'optim'))
        
    args.tr_iter = 0
    args.vl_iter = 0
    net = net.to(torch.device("cuda"))
    net = DDP(net, device_ids=[dpp.local_rank])

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
    
    # Learning rate scheduler dictionary
    dict_schedulers={'plateau': optim.lr_scheduler.ReduceLROnPlateau(\
                 optimizer, factor=.8, patience=10, cooldown=5,
                 min_lr=1e-10),
                 'multistep': optim.lr_scheduler.MultiStepLR(optimizer,
                 milestones=[60, 120, 160], gamma=0.2)}

    
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
            print('Resuming training at epoch {}, batch {}'.format(\
                  last_epoch, current_batch))
        except IndexError:
            args.tr_iter = epoch_iter_tr*last_epoch
            args.val_iter = epoch_iter_val*(last_epoch-1)
            print('Resuming training at the begining of epoch {}'.format(\
                  last_epoch))
                  
        state_model = torch.load(l_model,
                                 map_location=lambda storage,
                                 loc:storage)
        state_optim = torch.load(l_optim,
                                 map_location=lambda storage,
                                 loc:storage)

        weights_loader(net, state_model)
        weights_loader(optimizer, state_optim)
           
    # If loading from a pretrained model, load the weights
    if args.pretrained:
        state_model = torch.load(args.pre_path,
                                 map_location=lambda storage,
                                 loc: storage)

        weights_loader(net, state_model)

    args.scheduling = copy.deepcopy(args.scheduler)
    args.scheduler = dict_schedulers[args.scheduler]
    args.warmup = WarmUpLR(optimizer,
                      args.batches_train * args.warm)

    if dpp.rank == 0:
        log = logger(args.store_dir, last_epoch)
        args.logger = log

    train_data = dataset_wrapper(train_data, dpp.size, dpp.rank, args)
    val_data = dataset_wrapper(val_data, dpp.size, dpp.rank, args)
    
    # ====================================================================
    # Training loop
    for epoch in range(last_epoch+1, args.epochs+1):

        args.epoch = epoch
        if dpp.rank == 0:
            print('====== Starting Epoch: {}/{} --> {} ======'.format(\
                  epoch, args.epochs, time.strftime("%H:%M:%S")))
        for param_group in optimizer.param_groups:
            if dpp.rank == 0:
                print('Learning Rate: {:.5f}'.format(param_group['lr']))

        # Trainining
        loss_train, acc_train = simplified_trainer(train_data,
                                                  net, criterion,
                                                  optimizer, args)

        loss_val, acc_val = simplified_evaluator(val_data,
                                                 net, criterion, args)

        if epoch > args.warm:
            if 'multistep' in args.scheduling:
                args.scheduler.step(epoch)
            elif 'plateau' in args.scheduling:
                args.scheduler.step(loss_train)


        # Saving the model with the lowest loss
        if loss_val < min_val and dpp.rank == 0:
            min_val = loss_val
            named = osj(args.store_dir, 'best_model.pth')

            torch.save(net.state_dict(), named)
            print('Best model saved')

        # Saving every given number of epochs
        if epoch % args.s_iterations == 0 and dpp.rank == 0:
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
