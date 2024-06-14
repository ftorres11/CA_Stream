# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch imports
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, distributed

# In package imports
from .utils import seed_worker

# Package imports
import os
import pdb
import pickle
import numpy as np
from PIL import Image

osp = os.path
osj = osp.join


# ========================================================================
# Normalization and de-normalization
MNIST_normalization = transforms.Normalize((.1307,), (.3081,))
MNIST_denormalization = transforms.Normalize((-.1307/.3081,), (1/.3081))

CIFAR_normalization = transforms.Normalize((.4914, .4822, .4465),
                                           (.2023, .1994, .2010))
CIFAR_denormalization = transforms.Normalize(\
                            (-.4914/.2023, -.4822/.1994, -.4465/.2010),
                            (1/.2023, 1/.1994, 1/.2010))

im_normalization = transforms.Normalize(mean=[.485, .456, .406],
                                        std=[.229, .224, .225])
im_denormalization = transforms.Normalize(\
                         mean=[-.485/.229, -.456/.224, -.406/.225],
                         std=[1/.229, 1/.224, 1/.225])


# ========================================================================
CIFAR_transform = transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((.4914, .4822, .4465),
                                            (.2023, .1994, .2010))])

CIFAR_training = transforms.Compose([transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.RandomRotation(15),
                       transforms.ToTensor(),
                       transforms.Normalize((.4914, .4822, .4465),
                                            (.2023, .1994, .2010))])

MNIST_transform = transforms.Compose([transforms.ToTensor(),
                      transforms.Normalize((.1307,), (.3081,))])

tensor_transform = transforms.Compose([transforms.ToTensor()])

im_normalization = transforms.Normalize(mean=[.485, .456, .406],
                                         std=[.229, .224, .225])

# ========================================================================
CIFAR10_train = datasets.CIFAR10(root='data', train=True,
                                 transform=CIFAR_training,
                                 download=False)
CIFAR10_val = datasets.CIFAR10(root='data', train=False,
                               transform=CIFAR_transform,
                               download=False)

CIFAR100_train = datasets.CIFAR100(root='data', train=True,
                                   transform=CIFAR_training,
                                   download=False)
CIFAR100_val = datasets.CIFAR100(root='data', train=False,
                                 transform=CIFAR_transform,
                                 download=False)

MNIST_train = datasets.MNIST(root='data', train=True,
                             transform=MNIST_transform,
                             download=False)
MNIST_val = datasets.MNIST(root='data', train=False,
                           transform=MNIST_transform,
                           download=False)

# ========================================================================
dict_datasets = {'CIFAR10': CIFAR10_train , 'CIFAR100': CIFAR100_train,
                 'MNIST': MNIST_train}

dict_validation = {'CIFAR10': CIFAR10_val, 'CIFAR100': CIFAR100_val,
                   'MNIST': MNIST_val}

# ========================================================================
def train_splitter(dataset, fraction):
    data_length = len(dataset)
    real_size = int(data_length*fraction)
    permuted_val = [int(x) for x in np.random.permutation(real_size)]
    permuted_train = []
    for x in range(data_length):
        if x not in permuted_val:
            permuted_train.append(x)
    return permuted_train, permuted_val

# ========================================================================
def imagenet_trainer(size):
    RRC = transforms.RandomResizedCrop
    inet_trans = transforms.Compose([transforms.Resize((size, size)),
                                     RRC((size,size)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.ConvertImageDtype(\
                                     torch.float),
                                     im_normalization])
    return inet_trans

# ========================================================================
def imagenet_tester(size):
    inet_trans = transforms.Compose([transforms.Resize((size, size)),
                                     transforms.ToTensor(),
                                     transforms.ConvertImageDtype(\
                                     torch.float),
                                     im_normalization])
    return  inet_trans

# ========================================================================
def dataset_wrapper(loader, size, rank, args):
    sampler = distributed.DistributedSampler(loader,
                  num_replicas=size, rank=rank)

    # Runs data loading/sampling with a fixed seed if desired
    if args.fixed:
        gen = torch.Generator()
        gen.manual_seed(args.seed)
        data = DataLoader(loader, batch_size=args.batch_size,
                          num_workers=0,
                          sampler=sampler,
                          worker_init_fn=seed_worker,
                          generator=gen,
                          shuffle=False)
    else:
        data = DataLoader(loader, batch_size=args.batch_size,
                num_workers=0,
                sampler=sampler)
        
    return data

# ========================================================================
# Classes
# ========================================================================
class split_MNIST(Dataset):
    def __init__(self, dataset, id_list):
        self.id_list = id_list
        self.dataset = dataset
    def __getitem__(self, x):
        image = self.dataset[self.id_list[x]][0]
        label = self.dataset[self.id_list[x]][1]
        return image, label

    def __len__(self):
        return len(self.id_list)

# ========================================================================
class INet_Trainer(Dataset):
    def __init__(self, root_data, contents, iterating_list,
                 transform=None):
        self.root = root_data
        self.contents = [contents[x] for x in iterating_list]
        self.transform = transform
        if transform:
            self.target_size = transform.transforms[0].size
    
    def __getitem__(self, idx):
        name, label = self.contents[idx].strip().split(' ')
        image = Image.open(osj(self.root, name)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(label)

    def __len__(self):
        return len(self.contents)

# ========================================================================
class INet_Evaluator(Dataset):
    def __init__(self, root_path, listed_data, transform=None):
        self.root = root_path
        self.data = listed_data
        self.ids = []
        for idx, _ in enumerate(listed_data):
            self.ids.append(idx)
        self.transform = transform

    def __getitem__(self, x):
        if ',' in self.data[x]:
            name, label = self.data[x].strip().split(',')
        else:
            name, label = self.data[x].strip().split(' ')

        image = Image.open(osj(self.root, name)).convert('RGB')
        idx = self.ids[x]
        if self.transform:
            image = self.transform(image)
        return image, int(label), idx
    def __len__(self):
        return len(self.data)

# ========================================================================
class INet_wNames(Dataset):
    def __init__(self, root_path, listed_data, transform=None):
        self.root = root_path
        self.data = listed_data
        self.ids = []
        for idx, _ in enumerate(listed_data):
            self.ids.append(idx)
        self.transform = transform

    def __getitem__(self, x):
        name, label = self.data[x].strip().split(',')
        image = Image.open(osj(self.root, name)).convert('RGB')
        idx = int(name.split('_')[-1].replace('.JPEG',''))
        if self.transform:
            image = self.transform(image)
        return image, int(label), idx
    def __len__(self):
        return len(self.data)

# ========================================================================
class Salient_Evaluator(Dataset):
    def __init__(self, parent_data, root_saliency):
        self.path = root_saliency
        self.data = parent_data

    def __getitem__(self, x):
        image, label, indx = self.data[x]
        saliency_map = np.load(osj(self.path, '{}.npy'.format(indx)))
        return image, label, saliency_map#, indx

    def __len__(self):
        return len(self.data)

# ========================================================================
class PascalClassifier(Dataset):
    def __init__(self, root_data, dict_json, transform=None):
        self.root = root_data
        self.names = dict_json.keys()
        self.data = dict_json
        self.transform = transform

    def __getitem__(self, x):
        name = self.names[x]
        labels = self.data[name]
        image = Image.open(osj(self.root, '{}.jpg'.format(name)))
        if self.transform:
            image = self.transform(image)
        return image, np.asarray(labels), name

    def __len__(self):
        return(len(self.names))

# ========================================================================
class Pascal_Evaluator(Dataset):
    def __init__(self, root_data, dict_json, transform):
        self.root = root_data
        self.names = [*dict_json]
        self.data = dict_json
        self.transform = transform

    def __getitem__(self, x):
        name = self.names[x]
        labels = self.data[name]
        image = Image.open(osj(self.root, '{}.jpg'.format(name)))
        if self.transform:
            image = self.transform(image)
        return image, np.asarray(labels), name

    def __len__(self):
        return(len(self.names))

# ========================================================================
class DiagnosticClass(Dataset):
    def __init__(self, root_salient, root_diagnostics, listed_data):
        self.root_salient = root_salient
        self.root_diagnostics = root_diagnostics
        self.data = listed_data

    def __getitem__(self, x):
        name, label = self.data[x].strip().split(',')
        name = name.replace('.JPEG', '')
        salient = np.load(osj(self.root_salient, '{}.npy'.format(x)))
        dict_diag = np.load(osj(self.root_diagnostics,
                            '{}.npy'.format(x)), allow_pickle=True)
        dict_diag = dict_diag.item()
        coeff = dict_diag['w']; gradient = dict_diag['grad'];
        logs = dict_diag['logits']; reprt = dict_diag['repr']
        return salient, int(label), coeff, gradient, logs, reprt, name

    def __len__(self):
        return len(self.data)


