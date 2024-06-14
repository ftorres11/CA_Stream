# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch Imports
from torch.utils.data import Dataset

# Package Imports
import os
osp = os.path
osj = osp.join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import xml.etree.ElementTree as ET


# ========================================================================
# Dataloading utils
# ========================================================================
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ========================================================================
def read_content(xml_file: str):
    """Reads bbox annotations of Pascal VOC datasets. Code from:
    https://stackoverflow.com/questions/53317592/reading-pascal-voc-annotations-in-python
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes

# ========================================================================
def dataset_partition(listed_data, fraction):
    # Listed data should be a list, fraction a float
    data_length = len(listed_data)

    if fraction > 1:
        raise ValueError('Cannot use a fraction bigger than 1')

    elif fraction == 1:
        effective_size = data_length
        listed_fraction = np.arange(0, effective_size)

        return listed_fraction, effective_size

    else:
        effective_size = int(np.around(data_length*fraction))
        complete_data = np.arange(0, data_length)
        random_samples = np.random.permutation(data_length)
        listed_fraction = []

        for idx in range(effective_size):
            listed_fraction.append(complete_data[random_samples[idx]])

        return listed_fraction, effective_size

# ========================================================================
class SplitLoader(Dataset):
    def __init__(self, parent_data):
        self.ids = range(0, len(parent_data))
        self.data = parent_data

    def __getitem__(self, x):
        image, label = self.data[x]
        idx = self.ids[x]

        return image, label, idx

    def __len__(self):
        return len(self.ids)
