# -*- coding: utf-8 -*-
# Author: Felipe Torres Figueroa, felipe.torres@lis-lab.fr

# Torch imports
import torch

# In-package imports

# Package imports
import os
import cv2
import numpy as np
import scipy.ndimage
from typing import Union
from dataclasses import dataclass


import sys
epsilon = sys.float_info.epsilon

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
# ========================================================================
"""
Code adapted from:
https://github.com/zphang/saliency_investigation/blob/master/casme/model_basics.py
"""

@dataclass
class BoxCoords:
    # Exclusive

    xmin: Union[int, np.ndarray]
    xmax: Union[int, np.ndarray]
    ymin: Union[int, np.ndarray]
    ymax: Union[int, np.ndarray]

    @property
    def xslice(self):
        return slice(self.xmin, self.xmax)

    @property
    def yslice(self):
        return slice(self.ymin, self.ymax)

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @property
    def area(self):
        return self.width * self.height

    def clamp(self, vmin, vmax):
        return self.__class__(
            xmin=np.clip(self.xmin, vmin, vmax),
            xmax=np.clip(self.xmax, vmin, vmax),
            ymin=np.clip(self.ymin, vmin, vmax),
            ymax=np.clip(self.ymax, vmin, vmax),
        )

    @classmethod
    def from_dict(cls, d):
        return cls(
            xmin=d["xmin"],
            xmax=d["xmax"],
            ymin=d["ymin"],
            ymax=d["ymax"],
        )

    def to_dict(self):
        return {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
        }

    
def get_bounding_box(m):
    x = m.any(0)
    y = m.any(1)
    box_coords = BoxCoords(xmin=np.argmax(x), xmax=np.argmax(np.cumsum(x)),
                           ymin=np.argmax(y), ymax=np.argmax(np.cumsum(y)),
                           )
    with torch.no_grad():
        box_mask = torch.zeros(224, 224).to(device)
        box_mask[box_coords.yslice, box_coords.xslice] = 1
        
    return box_mask, box_coords

def get_largest_connected(m):
    mask, num_labels = scipy.ndimage.label(m)
    largest_label = np.argmax(np.bincount(
        mask.reshape(-1), weights=m.reshape(-1)))
    largest_connected = (mask == largest_label)

    return largest_connected

def get_rectangular_mask(m):
    return get_bounding_box(get_largest_connected(m))


def binarize_mask(mask):
    with torch.no_grad():
        batch_size = mask.size(0)
        avg = mask.view(batch_size, -1).mean(dim=1)
        binarized_mask = mask.gt(avg.view(batch_size, 1, 1, 1)).float()
        return binarized_mask.to(device)

def classification_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# ========================================================================
"""Now for evaluating 
https://github.com/zphang/saliency_investigation/blob/cdd03e92c4b8b26c43a60cb50f9d97ea2c7f9400/casme/tasks/imagenet/score_bboxes.py
"""
def compute_f1(m, gt_box, gt_box_size):
    with torch.no_grad():
        inside = (m*gt_box).sum()
        precision = inside / (m.sum() + 1e-6)
        recall = inside / gt_box_size
        
        return (2 * precision * recall)/(precision + recall + 1e-6)

def compute_iou(m, gt_box, gt_box_size):
    with torch.no_grad():
        intersection = (m*gt_box).sum()
        return intersection / (m.sum() + gt_box_size - intersection)
    
def get_loc_scores(bbox, continuous_mask, rectangular_mask):
    if bbox.area == 0:
        return 0, 0

    truncated_bbox = bbox.clamp(0, 224)

    gt_box = np.zeros((224, 224))
    gt_box[truncated_bbox.yslice, truncated_bbox.xslice] = 1

    f1 = compute_f1(continuous_mask, gt_box, bbox.area)
    iou = compute_iou(rectangular_mask, gt_box, bbox.area)

    return f1, 1*(iou > 0.5)


def get_path_stub(path):
    return os.path.basename(path).split('.')[0]

def get_image_bboxes(bboxes_dict, path):
    ls = []
    for bbox in bboxes_dict[get_path_stub(path)]:
        ls.append(BoxCoords.from_dict(bbox))
    return ls


def tensor_to_image_arr(x):
    if len(x.shape) == 4:        
        return x.permute([0, 2, 3, 1]).cpu().numpy()
    elif len(x.shape) == 3:
        return x.permute([1, 2, 0]).cpu().numpy()
    else:
        raise RuntimeError(x.shape)

def compute_saliency_metric(input_, target, bbox_coords, classifier):
    resized_sliced_input_ls = []
    area_ls = []
    for i, bbox in enumerate(bbox_coords):
        sliced_input_single = tensor_to_image_arr(input_[i, :, bbox.yslice, bbox.xslice])
        if bbox.area>0 and sliced_input_single.size > 0:
            resized_sliced_input_single = cv2.resize(
                sliced_input_single,
                (224, 224),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            resized_sliced_input_single = np.zeros([224, 224, 3])
        area_ls.append(bbox.area)
        resized_sliced_input_ls.append(resized_sliced_input_single)
    resized_input = torch.tensor(np.moveaxis(np.array(resized_sliced_input_ls), 3, 1)).float()
    with torch.no_grad():
        cropped_upscaled_yhat = classifier(resized_input.to(device))
    term_1 = (torch.tensor(area_ls).float() / (224 * 224)).clamp(0.05, 1).log().numpy()
    term_2 = torch.softmax(cropped_upscaled_yhat, dim=-1).detach().cpu()[
        torch.arange(cropped_upscaled_yhat.size(0)), target].log().numpy()
    saliency_metric = term_1 - term_2
    # Note: not reduced

    # Slightly redundant, but doing to validate we're computing things correctly
    acc1, = classification_accuracy(cropped_upscaled_yhat, target.to(device), topk=(1,))

    return saliency_metric, term_1, term_2, acc1.item()

# ========================================================================
""" From:
https://github.com/clovaai/wsolevaluation/blob/e00842f8e9d86588d45f8e3e30c237abb364bba4/util.py#L48
""" 
def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != np.float:
        raise TypeError("Scoremap must be of np.float type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))

def check_box_convention(boxes, convention):
    """
    Args:
        boxes: numpy.ndarray(dtype=np.int or np.float, shape=(num_boxes, 4))
        convention: string. One of ['x0y0x1y1', 'xywh'].
    Raises:
        RuntimeError if box does not meet the convention.
    """
    if (boxes < 0).any():
        raise RuntimeError("Box coordinates must be non-negative.")

    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, 0)
    elif len(boxes.shape) != 2:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if boxes.shape[1] != 4:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if convention == 'x0y0x1y1':
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif convention == 'xywh':
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    else:
        raise ValueError("Unknown convention {}.".format(convention))

    if (widths < 0).any() or (heights < 0).any():
        raise RuntimeError("Boxes do not follow the {} convention."
                           .format(convention))

# ========================================================================
"""
From:
https://github.com/clovaai/wsolevaluation/blob/e00842f8e9d86588d45f8e3e30c237abb364bba4/evaluation.py
"""
def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                  multi_contour_eval=False):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation
    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list

def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious
# ========================================================================
# Original-ish//Hanwei's code
def energy_point_game(bbox, saliency_map):
    saliency_map = saliency_map.squeeze()
    w, h = saliency_map.shape
    empty = np.zeros((w, h))
    for box in bbox:
        empty[box.xslice, box.yslice] = 1
    mask_bbox = saliency_map * empty
    energy_bbox =  mask_bbox.sum()
    energy_whole = saliency_map.sum()
    proportion = energy_bbox / (energy_whole + epsilon)
    return proportion


# ========================================================================
def accumulate(scoremap, bboxes, threshold_cam, threshold_iou, correct, 
               cnt_):
    num_correct = correct
    box_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(\
                 scoremap=scoremap, scoremap_threshold_list=threshold_cam,
                 multi_contour_eval=False)
    boxes_at_thresholds = np.concatenate(box_at_thresholds, axis=0)
    multiple_iou = calculate_multiple_iou(np.array(boxes_at_thresholds),
                                          np.array(bboxes))
    idx = 0
    sliced_multiple_iou = []
    for nr_box in number_of_box_list:
        sliced_multiple_iou.append(max(multiple_iou.max(1)[idx:idx+nr_box]))
        idx += nr_box

    for threshold in threshold_iou:
        correct_threshold_indices = np.where(np.asarray(sliced_multiple_iou)\
            >=(threshold/100))[0]
        num_correct[threshold][correct_threshold_indices] +=1
    cnt_ +=1
    return num_correct, cnt_

# ========================================================================
def compute(threshold_iou, correct, cnt):
    max_box_acc = []
    for threshold in threshold_iou:
        localization_acc = correct[threshold] * 100. /float(cnt)
        max_box_acc.append(localization_acc.max())
    return max_box_acc

