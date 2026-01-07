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

from SilentInfarctionSegmentationFLAIR.utils import get_array_from_image
from SilentInfarctionSegmentationFLAIR.utils import get_image_from_array


def get_mask_from_segmentation(segm, labels=1):
    """
    Generate a binary mask from a segmentation image.

    Parameters
    ----------
    segm : SimpleITK.Image
        Segmentation image where each voxel contains an integer label.
    labels : int, float, iterable, or dict
        Labels to include in the mask. If a dict is provided, its values
        are used.

    Returns
    -------
    mask : SimpleITK.Image
        Binary mask where selected labels are set to 1.
    """
    if isinstance(labels, (int, float)):
        labels = [labels]
    if isinstance(labels, dict):
        labels = list(labels.values())

    mask = segm == labels[0]
    for i in range(1, len(labels)):
        mask = sitk.Or(mask, segm == labels[i])

    if not np.any(get_array_from_image(mask)):
        warnings.warn(
            f"Labels {labels} are not present in the segmentation image. "
            "Returned mask will be empty."
        )

    return mask


def get_mask_from_pve(pve, thr=1e-12):
    """
    Generate a binary mask from a partial volume estimation (PVE) map.

    Parameters
    ----------
    pve : SimpleITK.Image
        Partial volume estimation map.
    thr : float, optional
        Threshold value. Voxels >= thr are set to 1. Default is 1e-12.

    Returns
    -------
    mask : SimpleITK.Image
        Binary mask.
    """
    return pve >= thr


def apply_threshold(image, thr, ax=None):
    """
    Apply a binary lower threshold to an image.

    Voxels with intensity >= ``thr`` are set to 1, otherwise to 0.
    Optionally draws a vertical threshold line on a provided axis.

    Parameters
    ----------
    image : SimpleITK.Image
        Input image.
    thr : float
        Lower threshold value.
    ax : matplotlib.axes.Axes, optional
        Axis on which to draw the threshold line.

    Returns
    -------
    thr_image : SimpleITK.Image
        Binary thresholded image.
    """
    arr = get_array_from_image(image)
    max_gl = np.max(arr)

    if thr > max_gl:
        warnings.warn(
            f"Lower threshold ({thr}) is higher than maximum gray level "
            f"({max_gl})."
        )
        empty_arr = np.zeros(arr.shape)
        return get_image_from_array(empty_arr, image)

    upper_thr = float(max_gl)
    thr_mask = sitk.BinaryThreshold(
        image,
        lowerThreshold=thr,
        upperThreshold=upper_thr
    )

    if ax is not None:
        ax.axvline(
            thr,
            linestyle="--",
            color="lime",
            linewidth=2,
            label=f"Threshold ({thr:.1f})"
        )

    return thr_mask


def evaluate_voxel_wise(mask, gt):
    """
    Compute voxel-wise evaluation metrics between prediction and ground truth.

    Metrics include:
    - True Positive Fraction (TPF)
    - False Positive Fraction (FPF)
    - Dice Similarity Coefficient (DSC)
    - Matthews Correlation Coefficient (MCC)

    Parameters
    ----------
    mask : SimpleITK.Image
        Binary prediction mask.
    gt : SimpleITK.Image
        Binary ground-truth mask.

    Returns
    -------
    metrics : dict
        Dictionary containing voxel-wise metrics.
    """
    mask_arr = get_array_from_image(mask)
    gt_arr = get_array_from_image(gt)

    tp = np.sum((mask_arr > 0) & (gt_arr > 0))
    tn = np.sum((mask_arr == 0) & (gt_arr == 0))
    fp = np.sum((mask_arr > 0) & (gt_arr == 0))
    fn = np.sum((mask_arr == 0) & (gt_arr > 0))

    if 2 * tp + fp + fn == 0:
        dice = 1.0 if np.array_equal(mask_arr, gt_arr) else 0.0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)

    pos = tp + fn
    neg = tn + fp

    metrics = {
        "vw-TPF": tp / pos if pos > 0 else 0.0,
        "vw-FPF": fp / neg if neg > 0 else 0.0,
        "vw-DSC": dice,
        "vw-PPV": tp / (tp + fp) if (tp + fp) > 0 else 0.0
    }

    return {k: float(v) for k, v in metrics.items()}
