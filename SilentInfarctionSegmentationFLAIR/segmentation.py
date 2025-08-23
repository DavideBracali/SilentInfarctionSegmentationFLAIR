#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 19:49:24 2025

@author: david
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import warnings


from SilentInfarctionSegmentationFLAIR.utils import DimensionError
from SilentInfarctionSegmentationFLAIR.utils import check_3d
from SilentInfarctionSegmentationFLAIR.utils import get_info
from SilentInfarctionSegmentationFLAIR.utils import get_array_from_image
from SilentInfarctionSegmentationFLAIR.utils import plot_image
from SilentInfarctionSegmentationFLAIR.utils import orient_image
from SilentInfarctionSegmentationFLAIR.utils import resample_to_reference



def get_mask_from_segmentation(segm, labels=1):
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


def apply_threshold(image, thr, show=True, ax=None):
    """
    Applies a binary lower threshold to a SimpleITK image.
    Voxels with gray level >= thr will be set to 1, otherwise to 0.
    Optionally plots a vertical green dashed line.

    Parameters
    ----------
        - image (SimpleITK.image): The image to threshold.
        - thr (float): The gray level threshold to apply. 
    
    Returns
    -------
        - thr_image (SimpleITK.image): The thresholded binary image.
    """
    max_gl = get_array_from_image(image).max()
    # set upper threshold higher than maximum gl because default value is 255
    if thr >= max_gl:
        upper_thr = float(thr + 1)
        warnings.warn(f"Lower threshold ({thr}) is higher than maximum gray level ({max_gl}).")
    else:
        upper_thr = float(get_array_from_image(image).max() + 1)

    thr_image = sitk.BinaryThreshold(image, lowerThreshold=thr,
                        upperThreshold=upper_thr)

    if show:
        if ax is None:
            ax = plt.gca()
        ax.axvline(thr, linestyle='--', color='lime',
                linewidth=2, label=f"Threshold ({thr:.1f})")

    return thr_image



def evaluate_voxel_wise(mask, gt):
    """
    Returns voxel-wise evaluation parameters from two binary images.

    Parameters
    ----------
        mask (SimpleITK.image): The binary image to evaluate.
        gt (SimpleITK.image): The binary image containing the ground truth.
    
    Returns
    -------
        metrics (dict): A dict containing true/false positive fractions
            and the DICE coefficient (floats).
    """
    mask_arr = get_array_from_image(mask)
    gt_arr = get_array_from_image(gt)

    # true/false positives/negatives    
    tp = np.sum((mask_arr > 0) & (gt_arr > 0))
    tn = np.sum((mask_arr == 0) & (gt_arr == 0))
    fp = np.sum((mask_arr > 0) & (gt_arr == 0))
    fn = np.sum((mask_arr == 0) & (gt_arr > 0)) 

    # DICE
    if 2*tp + fp + fn == 0:   # do not divide by 0
        dice = 1.0 if np.array_equal(mask_arr, gt_arr) else 0.0
    else:
        dice = 2*tp / (2*tp + fp + fn)

    pos = tp + fn
    neg = tn + fp

    metrics = {
        "vw-TPF": tp / pos if pos > 0 else 0.0,
        "vw-FPF": fp / neg if neg > 0 else 0.0,
        "vw-DSC": dice
    }
    metrics = {k: float(v) for k,v in metrics.items()}


    return metrics