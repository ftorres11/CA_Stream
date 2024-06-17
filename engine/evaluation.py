# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch imports
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

# In Package imports
from .routines import AverageMeter
from .attention import cam_processing#, raw_attention,
                        #rollout_attention, ronan_attention,
                        #gradcam_cls, gradcam_gradcam,
                        #hybrid_ham)

from .utils import gradient_generation
from .interpretable.ins_del import CausalMetric, gkern, auc
from .interpretable.perturbation import PerturbationMetric

# Package imports
import os
osp = os.path
osj = osp.join
import pdb

from sklearn.metrics import average_precision_score

import sys
epsilon = sys.float_info.epsilon

import cv2
import numpy as np

import copy
import time


# ========================================================================
dict_methods = {'cam': cam_processing,
                'raw_attention': raw_attention,
                'rollout_attention': rollout_attention}


# ========================================================================
# Supporting Functions
# ========================================================================
def activation_extractor(wrapper, images, labels, args):
    approach = args.method
    # Zero-grad models
    bsz, _, wd_d, hght_d = images.shape
    wrapper.model.zero_grad()
    features, logits = wrapper(images)
    features = features[-1].detach()
    #logits, cl_r = logits # Diagnostics
    if isinstance(logits, tuple):
        score = gradient_generation(logits[-1], labels)
    else:
        if args.pred:
            labels = torch.argmax(logits, dim=-1)
        score = gradient_generation(logits, labels)
    args.labels = labels

    activations = dict_methods['cam'](wrapper, features, score, images,
                                          args)

    #return logits, activations, cl_r
    return logits, activations

# ========================================================================
def multilabel_activation(wrapper, images, labels, args):
    approach = args.method
    # Zero-grad models
    bsz, _, wd_d, hght_d = images.shape
    wrapper.model.zero_grad()
    positives = torch.nonzero(labels.clamp(min=0))
    logits = wrapper.model(images)
    salient = torch.empty(0).to(torch.cuda.current_device())
    for _, position in enumerate(positives):
        wrapper.model.zero_grad()
        im_id, label = position
        to_forward = images[im_id].unsqueeze(0)
        feat_i, log_i = wrapper(to_forward)
        feat_i = feat_i[-1].detach()
        if isinstance(logits, tuple):
            score = gradient_generation(log_i[-1], lab_i)
        else:
            score = [log_i[0, label]]
        args.labels = label
        im_act = dict_methods['cam'](wrapper, feat_i, score,
                                     to_forward, args)
        salient = torch.cat((salient, im_act), 0)  

    return logits, salient

# ========================================================================
def mAP_evaluation(scores, labels, args):
    # Making a  vector of zeros with n classes
    mAP_vector = np.zeros(args.classes, dtype=float)
    for idx in range(args.classes):
        mAP_vector[idx] = average_precision_score(\
                           labels[:, idx].reshape(-1),
                           scores[:, idx].reshape(-1))

    # Computing experiment-wise mAP
    experiment_mAP = np.mean(mAP_vector)
    return mAP_vector, experiment_mAP
       
# ========================================================================
def gradcampp_recognition(original, explanation, labels, args):
    labels = labels.cpu()
    original = original.cpu().detach().squeeze()
    explanation = explanation.cpu().detach().squeeze()
    if not args.multilabel:
        length = torch.from_numpy(np.arange(0, labels.shape[0], 1))
    
    # Average Drop
    ad_b = torch.clamp(original-explanation, min=0)
    ad_b = ad_b/(original+epsilon).numpy()
    # Average Gain
    ag_b = torch.clamp(explanation-original, min=0)
    ag_b = ag_b/(1-original+epsilon).numpy()
    # Increase in Confidence
    ic_b = torch.sign(explanation-original).clamp(min=0)
    
    # Vectorized
    if args.multilabel:
        ad = ad_b[labels].detach() # Vectorized Average Drop
        ic = ic_b[labels].detach() # Vectorized Increase in Confidence
        ag = ag_b[labels].detach() # Vectorized Average Gain
        ad = (np.asarray(ad)).mean()*100 # Mean Average Drop
        ic = (np.asarray(ic)).mean()*100 # Mean Increase in Confidence
        ag = (np.asarray(ag)).mean()*100 # Mean Average Gain

    else:
        ad = ad_b[length, labels].detach() # Vectorized Average Drop
        ic = ic_b[length, labels].detach() # Vectorized Increase in Confidence
        ag = ag_b[length, labels].detach() # Vectorized Average Gain
        ad = (np.asarray(ad)).mean()*100 # Mean Average Drop
        ic = (np.asarray(ic)).mean()*100 # Mean Increase in Confidence
        ag = (np.asarray(ag)).mean()*100 # Mean Average Gain

    return ad, ic, ag


# ========================================================================
def saliency_extractor(wrapper, images, labels, args):
    approach = args.method
    features, logits = wrapper(images)
    # Getting base information, keep wrapper inside as for CAM backprop
    # is needed
    #if 'cls' not in args.model:
    #    approach = 'raw_attention'
    attention = dict_methods[approach](features, images)
    return logits, attention

# ========================================================================
def saliency_storer(saliencies, args):
    bsz, _, _ = saliencies.shape
    for idx in range(bsz):
        smap = saliencies[idx, :, :].cpu().numpy()
        np.save(osj(args.path_salient, '{}.npy'.format(args.idx[idx])),
                smap)

# ========================================================================
def multisalient_storer(saliencies, labels, args):
    positives = torch.nonzero(labels.clamp(min=0))
    for position in positives:
        im_id, label = position
        smap = saliencies[im_id, :, :].cpu().numpy()
        str_lab = args.names[label]
        im_name = args.idx[im_id]
        np.save(osj(args.path_salient, '{}_{}.npy'.format(im_name,
                str_lab)), smap)
    
# ========================================================================
# Diagnostic function to store gradient, probs, repr, 
# saliency map, weighting coefficient.
def diagnostic_storer(logits, activations, cl_r, args):
    bsz = logits.shape[0]
    _, coeff, grad = activations

    for idx in range(bsz):
        coef_x = coeff[idx].cpu().numpy()
        grad_x = grad[idx].cpu().numpy()
        log_x = logits[idx].detach().cpu().numpy()
        cl_x = cl_r[idx].cpu().numpy()
        dict_info = {'w': coef_x, 'grad': grad_x, 'logits': log_x,
                     'repr': cl_x}
        np.save(osj(args.path_salient, '{}.npy'.format(args.idx[idx])),
                dict_info)


# ========================================================================
def multiple_evaluator(wrapper, images, labels, mul_label, mul_score,
                       args):
    # Flushing gradients
    wrapper.model.zero_grad()
    # Getting input's shape
    _, ch, wh, hg = images.shape
    # Preparing tensor of positives
    if args.salient:
        logits, salient = multilabel_activation(wrapper, images,
                                                labels, args)

        logits = logits.detach().cpu()
        multisalient_storer(salient, labels, args)
    else:
        logits = wrapper.model(images).detach().cpu()

    mul_score = torch.cat((mul_score, torch.sigmoid(logits)), 0)
    mul_label = torch.cat((mul_label, labels.cpu()), 0).clamp(min=0)
    return mul_label, mul_score

# ========================================================================
def single_evaluator(wrapper, images, labels, acc_metric, missclassified,
                     args):

    # Flushing gradients
    wrapper.model.zero_grad()

    if args.salient:
        if 'cam' in args.method:
            logits, salient  = activation_extractor(wrapper, images,
                                                    labels, args)

        else:
            logits, salient = saliency_extractor(wrapper, images, labels,
                                                 args)
        saliency_storer(salient, args)

    else:
        logits = wrapper.model(images)

    if isinstance(logits, tuple):
        prediction = torch.argmax(logits[-1], 1)
    else: 
        prediction = torch.argmax(logits, 1)

    correct = (labels == prediction)*1
    acc = correct.cpu().numpy().mean()
    
    # Appending missclassified values and updating metrics
    failing = args.idx[torch.where(labels.cpu()!=\
                       prediction.cpu())].tolist()
    missclassified += failing
    acc_metric.update(acc)

# ========================================================================
def multi_recognition(model, batch, norms, metrics, insertion, deletion,
                      positive, negative, args):
    # Sigmoid for AD-IC-AG & Ins - Del
    sigmoid = nn.Sigmoid()
    # Retrieving information from the batch
    images, labels, names = batch
    images = images.to(torch.cuda.current_device())

    positives = torch.nonzero(labels.clamp(min=0))
    logits = model(images)
    base_probs = sigmoid(logits)
    ad_m, ic_m, ag_m, ins_m, del_m, pos_m, neg_m = metrics
    for _, position in enumerate(positives):
        # Getting positive labels only for evaluation
        im_id, label = position
        prefix = names[im_id]
        name = args.names[label]
        to_forward = images[im_id].unsqueeze(0)
        probs_id = base_probs[im_id].unsqueeze(dim=0) # Probability 
        # ================================================================
        # Saliency Map
        smap = np.load(osj(args.path_salient, '{}_{}.npy'.format(prefix,
                       name)))
        smap = torch.from_numpy(smap).unsqueeze(0)
        smap = smap.unsqueeze(0).to(torch.cuda.current_device())
        smap = F.interpolate(smap, size=to_forward.shape[2:],
                             mode='bilinear')
        # ================================================================a
        # Processing - AD, AG- AI
        denormalized = norms['denormalization'](to_forward)
        normalized = norms['normalization'](denormalized*smap)
        logs_id = model(normalized)
        sid_probs = sigmoid(logs_id)
        ad_i, ic_i, ag_i = gradcampp_recognition(probs_id, sid_probs,
                                                 label, args)
        # ================================================================
        # Processing - Ins/Del, +Pert/-Pert
        smap = smap.squeeze(0).detach()
        smap = smap.cpu().numpy()
        ins_i = auc(insertion.single_run(to_forward, smap))
        del_b = auc(deletion.single_run(to_forward, smap))

        pos_i = positive.evaluate(to_forward, smap, label)
        neg_i = negative.evaluate(to_forward, smap, label)
        
        # ================================================================
        # Updating meters
        ad_m.update(ad_i)
        ic_m.update(ic_i)
        ag_m.update(ag_i)
        ins_m.update(ins_i)
        del_m.update(del_b)
        pos_m.update(pos_i)
        neg_m.update(neg_i)

# ========================================================================
def single_recognition(model, batch, norms, metrics, insertion, deletion,
                       positive, negative, args):
    # Softmax for AD-IC & INS-DEL
    softmax = nn.Softmax(dim=1)
    # Retrieving information from the batch
    images, labels, smaps = batch
    images = images.to(torch.cuda.current_device())
    labels = labels.to(torch.cuda.current_device())
    smaps = smaps.unsqueeze(1).to(torch.cuda.current_device())
    smaps = F.interpolate(smaps, size=images.shape[2:], mode='bilinear') 
    # Retrieving average meters for metrics
    ad_m, ic_m, ag_m, ins_m, del_m, pos_m, neg_m = metrics
    # Beginning of interpretable recognition routine
    logits = model(images)
    if args.pred:
        _, labels = torch.max(logits, 1)
    if isinstance(logits, tuple):
        logits = logits[-1]
    base_probs = softmax(logits)
    # Now forward with saliency
    model.zero_grad()
    denormalized = norms['denormalization'](images)
    normalized = norms['normalization'](denormalized*smaps)
    c_logits = model(normalized)
    if isinstance(c_logits, tuple):
        c_logits = c_logits[-1]
    salient_probs = softmax(c_logits)
    ad_b, ic_b, ag_b = gradcampp_recognition(base_probs, salient_probs,
                                             labels, args)
    # Insertion-deletion
    # Saliency has to be shaped: Batch x Width x Height
    smaps = smaps.squeeze(1).detach() # Previously Batch x 1 x Width x Height
    smaps = smaps.cpu().numpy()
    ins_b = auc(insertion.evaluate(images, smaps, images.shape[0],
                                   labels).mean(1))
    del_b = auc(deletion.evaluate(images, smaps, images.shape[0],
                                  labels).mean(1))
    ins_b = ins_b.mean(axis=0).mean()
    del_b = del_b.mean(axis=0).mean()

    # Perturbation - game
    pos_b = 0
    neg_b = 0
    # Updating Metrics
    ad_m.update(ad_b) # Average Drop
    ic_m.update(ic_b) # Increase in Confidence
    ag_m.update(ag_b)
    ins_m.update(ins_b) # Insertion
    del_m.update(del_b) # Deletion
    pos_m.update(pos_b) # Positive perturbation
    neg_m.update(neg_b) # Negative perturbation

# ========================================================================
# Evaluation code
# ========================================================================
def recognition_evaluator(wrapper, loader, args):
    data = DataLoader(loader, batch_size=args.batch_size, num_workers=0,
                      shuffle=False)

    # Initializing Average Metrics
    # Initializing lists of Average Drops and Misclassified examples
    if not args.multilabel:
        accuracy_metric = AverageMeter()
        missclassified = []
        path_miss = osj(args.store_dir, 'missclassified.csv')
        obj_miss = open(path_miss, 'w')

    else:
        mul_label = torch.empty(0)
        mul_score = torch.empty(0)

    # Preparing files to write metrics and misclassified examples
    path_metrics = osj(args.store_dir, 'metrics.csv')
    obj_metrics = open(path_metrics, 'w')
    obj_metrics.write('{}\n'.format('Acc'))

    iterations = len(loader)//args.batch_size
    args.counter = 0
    args.ten = len(data)*0.1
    # Iterating through the data
    for idx, batch in enumerate(data, 0):
        images, labels, idx_values = batch
        labels = labels.to(args.device)
        images = Variable(images).to(args.device)
        args.idx = idx_values
    
        if args.multilabel:
            mul_label, mul_score = multiple_evaluator(wrapper, images,
                                       labels, mul_label,
                                       mul_score, args)
        else:
            single_evaluator(wrapper, images, labels, accuracy_metric,
                             missclassified, args)

    # Writing Metrics
    if args.multilabel:
        mAP_vector, mAP = mAP_evaluation(mul_score.numpy(),
                                         mul_label.numpy(), args)
        obj_metrics.write('{:.5f}\n'.format(mAP))
        obj_metrics.close()
    else:
        obj_metrics.write('{:.5f}\n'.format(accuracy_metric.avg))
        obj_metrics.close()

        for fail in missclassified:
            obj_miss.write('{}\n'.format(fail))
        obj_miss.close()

# ========================================================================
def interpretable_recognition(model, loader, norm_dict, args):
    data = DataLoader(loader, batch_size=args.batch_size, num_workers=4,
                      shuffle=False)

    # Initializing metrics
    ad_metric = AverageMeter()
    ic_metric = AverageMeter()
    ag_metric = AverageMeter()
    ins_metric = AverageMeter()
    del_metric = AverageMeter()
    posp_metric = AverageMeter()
    negp_metric = AverageMeter()
    metrics = [ad_metric, ic_metric, ag_metric, ins_metric,
               del_metric, posp_metric, negp_metric]

    # Preparing files to write metrics
    path_metrics = osj(args.store_dir, 'int_recognition_metrics.csv')
    obj_metrics = open(path_metrics, 'w')
    # Writing the headers
    obj_metrics.write('{},{},{},{},{},{}\n'.format('AD', 'IC', 'AG',
                      'Ins', 'Del', '+Pert', '-Pert'))
    # Preparing for insertion/deletion
    klen = 5
    ksig = 5
    kern = gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern.to(args.device),
                                          padding=2)

    # Initializing Insertion/Deletion Workers
    insertion = CausalMetric(model, 'ins', args.imsize*8,
                             blur, args.imsize, args.classes)
    deletion = CausalMetric(model, 'del', args.imsize*8,
                            torch.zeros_like, args.imsize, args.classes)

    # Initializing (+)/(-) Perturbation Metric Workers
    positive = PerturbationMetric(model, 'positive', 0.1, norm_dict)
    negative = PerturbationMetric(model, 'negative', 0.1, norm_dict) 

    # ====================================================================
    for _, batch in enumerate(data, 0):
        images, labels, smaps = batch
        labels = labels.to(args.device)
        images = images.to(args.device)

        if args.multilabel:
            multi_recognition(model, batch, norm_dict, metrics,
                               insertion, deletion, positive, negative,
                               args)

        else:
            single_recognition(model, batch, norm_dict, metrics,
                               insertion, deletion, positive, negative,
                               args)

    # ====================================================================
    # Writing Metrics
    obj_metrics.write('{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(\
                      ad_metric.avg, ic_metric.avg, ag_metric.avg,
                      ins_metric.avg, del_metric.avg, posp_metric.avg,
                      negp_metric.avg))
    obj_metrics.close()
