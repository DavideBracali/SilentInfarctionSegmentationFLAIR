#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 19:49:24 2025

@author: david
"""

import numpy as np
import SimpleITK as sitk
import os
import warnings


from utils import DimensionError
from utils import check_3d
from utils import get_info
from utils import get_array_from_image
from utils import plot_image
from utils import orient_image
from utils import resample_to_reference



def get_mask_from_segmentation(segm, labels):
    """
    Returns a binary mask from a FreeSurfer segmentation image.
    
    Parameters
    ----------
    segm (SimpleITK.Image): Any segmentation image where each pixel is represented by a numeric label.
    labels (numeric or iterable): Iterable of labels to include in the mask.
    
    Returns
    -------
    mask (SimpleITK.Image): Binary mask.
    """
    # if labels is not a list, make it a list
    if isinstance(labels, (int,float)):      
        labels = [labels]            
    if isinstance(labels, dict):
        labels = list(labels.values())
    
    mask = segm == labels[0]         # first label
    for i in range(1,len(labels)):   # other labels (if present)
        mask = sitk.Or(mask, segm == labels[i])     # union
    
    if not np.any(get_array_from_image(mask)):       # empty mask
        warnings.warn(f"Labels {labels} are not present in the segmentation image. Returned mask will be empty.")
    
    return mask


def get_mask_from_pve(pve, thr=1e-12):
    """
    Returns a binary mask from a partial volume estimation (PVE) map.

    Parameters
    ----------
        pve (SimpleITK.Image): The partial volume estimation map.
        thr (float): Voxels >= thr will be set to 1, otherwise to 0.
    
    Returns
    -------
        mask (SimpleITK.Image): Binary mask.
    """

    mask = pve >= thr

    return mask


def evaluate(mask, gt):
    """
    Returns evaluation parameters from two binary images.

    Parameters
    ----------
        mask (SimpleITK.image): The binary image to evaluate.
        gt (SimpleITK.image): The binary image containing the ground truth.
    
    Returns
    -------
        metrics (dict): A dict containing true/false positive/negative fractions as floats.
        dice (float): DICE coefficient.
        accuracy (float): Fraction of correctly classified voxels.
    """

    if get_info(mask) != get_info(gt):
        mask = resample_to_reference(mask, gt)
    
    mask_arr = get_array_from_image(mask)
    gt_arr = get_array_from_image(gt)

    if not (set(np.unique(mask_arr)).issubset({0,1})
         and set(np.unique(gt_arr)).issubset({0,1})):
        raise ValueError(f"Mask and ground truth must be binary images containing only 0 and 1")
    
    tp = np.sum((mask_arr == 1) & (gt_arr == 1))
    tn = np.sum((mask_arr == 0) & (gt_arr == 0))
    fp = np.sum((mask_arr == 1) & (gt_arr == 0))
    fn = np.sum((mask_arr == 0) & (gt_arr == 1)) 

    pos = tp + fn
    neg = tn + fp

    metrics = {
        "TPR": tp / pos if pos > 0 else 0.0,
        "TNR": tn / neg if neg > 0 else 0.0,
        "FPR": fp / neg if neg > 0 else 0.0,
        "FNR": fn / pos if pos > 0 else 0.0,
    }
    metrics = {k: float(v) for k,v in metrics.items()}

        
    accuracy = (tp + tn) / (pos + neg)

    if tp + fp + fn == 0:
        dice = 1.0 if np.array_equal(mask_arr, gt_arr) else 0.0
    else:
        dice = 2*tp / (2*tp + fp + fn)

    return metrics, dice, accuracy