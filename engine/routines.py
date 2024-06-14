# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch imports
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler 

# Timm imports
from timm.models import model_parameters

# In Package imports
import engine.ddp as ddp

# Package imports
import os
osp = os.path
osj = osp.join
import re
import pdb

import copy
import glob
import time
import sys
import numpy as np
from contextlib import suppress
from sklearn.metrics import average_precision_score
import datetime
from functools import partial

epsilon = sys.float_info.epsilon


# ========================================================================
# Classes
# ========================================================================
class AverageMeter(object):
    # Computes and stores average and current values
    def __init__(self):
        self.reset()

    def reset(self):
    # Resets values to 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
    # Updates by one (or n) step(s) the contents
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum/self.count


# ========================================================================
class logger(SummaryWriter):
    def __init__(self, logging_path=None, start=0):
    # Tensorboard logger for running Experiments
        super().__init__()
        if logging_path == None:
            log_path = '.'

        if not isinstance(logging_path, str) and log_path != None:
            raise TypeError('Provide an actual path for the logger')
        # Sets the Logger name according to the time (H:M:S) the 
        # experiment is launched
        current = datetime.datetime.now()
        hour = current.hour
        minute = current.minute
        second = current.second
        self.logger = SummaryWriter(log_dir=osj(logging_path,
                                    'start_{}h{}m{}s'.format(hour,
                                    minute, second)),
                                    purge_step=start)

    def update(self, field, value, interval):
    # Updates the logger with the field-values-names passed
        self.logger.add_scalar(field, value, interval)

# ========================================================================
# Functions
# ========================================================================
def routine_resumer(store_dir):
    # Fetches the last optimizer, model, loss, epoch/batch to resume the
    # routine.
    # Fetches data from the directory where the experiment is located,
    # either the last batch where a checkpoint was made, or the epoch that
    # was last passed as checkpoint
    checkpoint = torch.load(osj(store_dir, 'checkpoint.pth'),
                            map_location=lambda storage,
                            loc:storage)
    
    return checkpoint

# ========================================================================
def stats_disabler(model):
    for name, module in model.named_parameters():
        if 'cls' in name.lower():
            continue
        elif 'norm' in name.lower() and 'cls' not in name.lower():
            module.eval()

# ========================================================================
def simplified_trainer(data, model, criterion, optimizer, args):
    model.train()
    # Disabling batch statistics
    if args.freeze:
        stats_disabler(model)

    args.training = True
    iterations = int(args.batches_train/ddp.size)
    # Average Meters initialization
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()
    rank = dist.get_rank()
    amp_dtype = torch.float16
    amp_autocast = partial(torch.autocast,
                           device_type=torch.device('cuda').type,
                           dtype=amp_dtype)
    if amp_autocast:
        loss_scaler = GradScaler()

    # Sets the initial time for the epoch loop
    time_init = time.time()
    second_order = hasattr(optimizer, 'is_second_order')\
                           and optimizer.is_second_order

    for idx, batch in enumerate(data, 0):
        # Resuming training if fixed batches were specified
        if args.resume and args.current_batch != idx:
            if idx % args.print_interval ==  0 and rank == 0:
                print('Seeking batch {}, current {}'.format(\
                      args.current_batch, idx))
            if idx+1 == args.current_batch and rank == 0:
                print('Resuming training at batch: {}'.format(\
                      idx+1))
            continue

        # Preparing data
        images, labels = batch
        labels = labels.to(ddp.local_rank, non_blocking=True)
        images = images.to(ddp.local_rank, non_blocking=True)

        optimizer.zero_grad() # Cleaning Gradients
        with amp_autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        _, indices = torch.max(outputs, 1)

        if args.labelsmooth or 'bce' in str(type(criterion)).lower():
            try:
                if criterion.label_smooth > 0:
                    _, labels = torch.topk(labels, 1,  dim=-1)
            except TypeError:
                pass

        accuracy = ((indices==labels)*1).sum()/len(indices)

        epoch_acc.update(accuracy)

        if loss_scaler:
            loss_scaler.scale(loss).backward()
            if args.clip_grad:
                loss_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            loss_scaler.step(optimizer)
            loss_scaler.update()

        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad:
                nn.utils.clip_grad_norm(args.params, args.clip_grad)
            optimizer.step()

        if args.ema:
            if idx % args.ema_steps == 0:
                args.ema.update_parameters(model)
                if args.epoch < args.warmup_iter:
                    args.ema.n_averaged.fill_(0)

        # Weight update
        optimizer.zero_grad()

        # Updating meters
        torch.cuda.synchronize()
        
        # Updating the logger
        if rank == 0:
            args.logger.update('Iteration Loss - Training', loss, 
                               args.tr_iter)
            epoch_loss.update(loss.cpu().data)
        # Updates the batch accuracy. Can't do batchwise mAP
        if rank == 0:
            args.logger.update('Tr Iteration Acc', accuracy,
                    args.tr_iter)
            
        # Stores intermediate results
        if idx % args.interm_store == 0 and args.interm_store != -1:
            if rank == 0:
                checkpoint_path = osj(args.store_dir, 'checkpoint.pth')

                checkpoint_dict = {'mode': 'batch',
                                   'model_sdict': model.state_dict(),
                                   'optim_sdict': optimizer.state_dict(),
                                   'epoch': args.epoch,
                                   'batch': idx,
                                   'loss': args.min_val}

                torch.save(checkpoint_dict, checkpoint_path)
                print('Saved intermediate model, batch {} of {}, epoch {}'.\
                      format(idx, iterations, args.epoch))
        
        # Printing batch statistics
        if idx % args.print_interval == 0 and rank == 0:
            print('[{}/{} ({:.0f}%)\tLoss:{:.4f}\tAcc:{:.4f}'
                  '\tTime: {}]'.format((idx+1), iterations,
                                       100.*(idx+1)/iterations,
                                       loss,
                                       accuracy,
                                       time.strftime('%H:%M:%S')))
        args.tr_iter += 1
        args.current_batch = -1
        args.resume = False

    # Updating mAP value on the logger
    if rank == 0:
        print('Epoch Loss: {:.4f}\t Epoch Acc: {:.4f}'.format(\
              epoch_loss.avg, epoch_acc.avg))
        total_time = time.time() - time_init
        print('Elapsed Epoch time: {:.0f}m {:.0f}s'.format(total_time//60,
          total_time%60))
    torch.cuda.empty_cache()
    return epoch_loss.avg, epoch_acc.avg

# ========================================================================
def simplified_evaluator(data, model, criterion, args):
    criterion = torch.nn.CrossEntropyLoss()
    iterations = int(args.batches_val/ddp.size)
    model.eval()
    args.training = False
    # Average Meters initialization
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()

    # Sets the initial time for the epoch loop
    time_init = time.time()
    rank = dist.get_rank()

    amp_dtype = torch.float16
    amp_autocast = partial(torch.autocast,
                           device_type=torch.device('cuda').type,
                           dtype=amp_dtype)
    #amp = suppress

    with torch.no_grad():
        for idx, batch in enumerate(data, 0):
            # Preparing data
            images, labels = batch
            labels = labels.to(ddp.local_rank, non_blocking=True)
            images = images.to(ddp.local_rank, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, indices = torch.max(outputs, 1)
            if 'binarycrossentropy' in str(type(criterion)).lower():
                _, labels = torch.topk(labels, 1,  dim=-1)
            accuracy = ((indices==labels)*1).sum()/len(indices)
            epoch_acc.update(accuracy)
            
            # Updating the logger
            if rank == 0:
                args.logger.update('Iteration Loss - Val', loss, 
                                   args.vl_iter)
                epoch_loss.update(loss.cpu().data)
            # Updates the batch accuracy. Can't do batchwise mAP
            if rank == 0:
                args.logger.update('Val Iteration Acc', accuracy,
                    args.vl_iter)

            del images, labels, outputs, batch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            args.vl_iter += 1
        
             # Printing batch statistics
            if idx % args.print_interval == 0 and rank == 0:
                print('[{}/{} ({:.0f}%)\tLoss:{:.4f}\tAcc:{:.4f}'
                      '\tTime: {}]'.format((idx+1), iterations,
                                           100.*(idx+1)/iterations,
                                           loss,
                                           accuracy,
                                           time.strftime('%H:%M:%S')))

    # Updating mAP value on the logger   
    torch.cuda.empty_cache()
    if rank == 0:
        print('Epoch Loss: {:.4f}\t Epoch Acc:{:.4f}'.format(\
              epoch_loss.avg, epoch_acc.avg))
        total_time = time.time() - time_init
        print('Elapsed Epoch time: {:.0f}m {:.0f}s'.format(\
              total_time//60, total_time%60))
    torch.cuda.empty_cache()   
    return epoch_loss.avg, epoch_acc.avg
