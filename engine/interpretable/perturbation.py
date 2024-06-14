# -*- coding: utf-8 -*- 
# Author Felipe Torres Figueroa, felipe.torres

# Torch imports
import torch
# In Package imports

# Package imports
import pdb
import numpy as np


# ========================================================================
# Classes
# ========================================================================
class PerturbationMetric():
    def __init__(self, model, mode, step, norm_dict):
        """ Creates a positive/negative perturbation metric instance based
        on Transformer Interpretability Beyond Attention Visualization
        https://arxiv.org/abs/2012.09838 for evaluation.
        Style and approach closer to Insertion-Deletion code
        """
        assert mode in ['positive', 'negative']
        self.model = model
        self.mode = mode
        self.step = np.arange(step, 1, step)
        self.norms = norm_dict

    def evaluate(self, batch_tensor, explanation_batch, labels):
        """Runs metric on batch of images-saliency pairs"""
        explanation_batch = torch.from_numpy(explanation_batch)
        base_size = batch_tensor.shape[-1]*batch_tensor.shape[-2]
        scores = torch.zeros((len(self.step)))

        if self.mode == 'negative':
            vis = -explanation_batch
        elif self.mode == 'positive':
            vis = explanation_batch

        # Here starts adaptation from
        # https://github.com/hila-chefer/Transformer-Explainability/blob/f085594997553b46ce026ad5478ea01267ed9ed7/baselines/ViT/pertubation_eval_from_hdf5.py
        org_shape = batch_tensor.shape
        vis = vis.reshape(org_shape[0], -1)

        for i in range(len(self.step)):
            _data = self.norms['denormalization'](batch_tensor.clone())
            _, idx = torch.topk(vis, int(base_size * self.step[i]), dim=-1)
            idx = idx.unsqueeze(1).repeat(1, org_shape[1], 1)
            idx = idx.to(_data.device)
            _data = _data.reshape(org_shape[0], org_shape[1], -1)
            _data = _data.scatter_(-1, idx, 0)
            _data = _data.reshape(*org_shape)

            _norm_data = self.norms['normalization'](_data)
            logits = self.model(_norm_data)
            if isinstance(logits, tuple):
                logits = logits[-1]
            idxs = torch.argmax(logits, 1)
            accuracy = ((idxs == labels)*1).sum()\
                       /batch_tensor.size()[0]
            scores[i] = accuracy

        return scores.mean()           
