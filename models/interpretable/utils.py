# -*- coding: utf-8 -*- 
# Author: Felipe Torres Figueroa - felipe.torres@lis-lab.fr

# Torch imports
import torch

# In-package imports


# Package imports
import pdb

# ========================================================================
class Activation_Wrapper:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []

        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))

            self.handles.append(target_layer.register_forward_hook(\
                    self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        # Adding support for multiple outputs
        if isinstance(activation, tuple):
            att, cls = output
            if self.reshape_transform is not None:
                att = self.reshape_transform(att)
            self.activations.append(att.detach())

        else:
            if self.reshape_transform is not None:
                activation = self.reshape_transform(activation)
            self.activations.append(activation.detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not \
                output.requires_grad:
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.activations, self.model(x)


    def release(self):
        for handle in self.handles:
            handle.remove()

