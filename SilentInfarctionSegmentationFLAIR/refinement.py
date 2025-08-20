#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 13:41:50

@author: david
"""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import generate_binary_structure, label, find_objects

from SilentInfarctionSegmentationFLAIR.utils import get_info
from SilentInfarctionSegmentationFLAIR.utils import get_array_from_image
from SilentInfarctionSegmentationFLAIR.utils import get_image_from_array


def connected_components(image, connectivity=26):
    """
    Separates a binary image into its connected components.

    Parameters
    ----------
        - image (SimpleITK.image): Binary input image
            (0=background, 1=foreground).
        - connectivity (bool): Connectivity type
            (6=face, 18=face+edge, 26=face+edge+corner).

    Returns
    -------
        - ccs (SimpleITK.image): Labeled image where each connected
            component has a unique integer value.
        - n_components (int): Number of connected components found
            (excluding background).
    """
    if connectivity not in (6, 18, 26):
        raise ValueError("'connectivity' must be 6, 18 or 26")

    arr = get_array_from_image(image)

    if connectivity == 6:
        structure = generate_binary_structure(3, 1)  # faces
    elif connectivity == 18:
        structure = generate_binary_structure(3, 2)  # faces + edges
    elif connectivity == 26:
        structure = generate_binary_structure(3, 3)  # faces + edges + corners

    ccs_arr, n_components = label(arr, structure=structure)

    ccs = get_image_from_array(ccs_arr)
    ccs.CopyInformation(image)

    return ccs, n_components
    

def find_diameters(ccs):
    """
    Computes the minimum and maximum length among any axis (diameter)
    of the connected components.

    Parameters
    ----------
        - ccs (SimpleITK.image): Connected components labeled image.

    Returns
    -------
        - diameters (dict): Dictionary of minimum and maximum diameters,
            where each key is a label (int) and the value is a tuple
            of floats of the type (min_diameter, max_diameter).
    """
    ccs_arr = get_array_from_image(ccs)
    spacing = get_info(ccs)["spacing"]
    diameters = {}

    slices = find_objects(ccs_arr)      # returns bounding boxes
    for label_idx, s in enumerate(slices, start=1): # 0 is background

        if s is None:
            continue

        diams = [(sl.stop - sl.start)*sp for sl, sp in zip(s, spacing)]
        min_d, max_d = min(diams), max(diams)
        diameters[label_idx] = (min_d, max_d)

    return diameters


def diameter_filter(ccs, lower_thr=None, upper_thr=None):
    """
    Filters out connected components whose length among any axis (diameter)
    is outside of the specified range.

    Parameters
    ----------
        - ccs (SimpleITK.image): Connected components labeled image.
        - lower_thr (float): Minimum connected component length on any axis.
        - upper_thr (float): Maximum connected component length on any axis.

    Returns
    -------
        - ccs_filtered (SimpleITK.image): Labeled image of connected components
            after filtering.
        - n_components (int): Number of connected components after filtering.
        - removed (dict): Dictionary of removed labels, where each key is a label (int)
            and each value is a tuple (min_diameter, max_diameter) (float).
    """
    ccs_arr = get_array_from_image(ccs)
    spacing = get_info(ccs)["spacing"]
    diameters = find_diameters(ccs)
    removed = {}

    if lower_thr is None and upper_thr is None:
        return ccs, len(np.unique(ccs_arr))-1, {}
    if lower_thr is None:
        lower_thr = 0
    if upper_thr is None:
        upper_thr = (np.array(ccs_arr.shape) * spacing).max()

    for label_idx, diams in diameters.items():
        min_d, max_d = diams
        # keep track of the labels to remove
        if min_d < lower_thr or max_d > upper_thr:
            removed[label_idx] = (min_d, max_d)

    # remove components
    to_remove = list(removed.keys())
    mask = np.isin(ccs_arr, to_remove)
    ccs_arr[mask] = 0

    ccs_filtered = get_image_from_array(ccs_arr)
    ccs_filtered.CopyInformation(ccs)
    n_components = len(diameters) - len(removed)

    return ccs_filtered, n_components, removed


def label_filter(segm, labels_to_remove=[], keywords_to_remove=[], labels_dict=None):
    """
    Sets the specified labels to 0 in a segmentation image.
    The labels to remove can be specified by number or by keyword.

    Parameters
    ----------
        - segm (SimpleITK.Image): Labeled segmentation image where each
            voxel is an integer label.
        - labels_to_remove (list): List of the integer labels to remove.
        - keywords_to_remove (list): List of strings containing the keywords
            of the labels to remove. Every label containing any of the specified
            keywords will be set to 0.
        - labels_dict (dict): Dictionary containing the relationship between the
            numeric labels and their description. Must be provided if
            keywords_to_remove is not an empty list, otherwise it will be ignored.
    Returns
    -------
        - segm_filtered (SimpleITK.Image): Labeled segmentation image after filtering.
        - removed (dict): Dictionary of removed voxels, where each key is a numeric
            label (int), and each value contains the number of removed voxels
    """
    if not keywords_to_remove == []:
        if labels_dict is None:
            raise ValueError("If 'keywords_to_remove' is specified then 'labels_dict' must be specified as well")
        else:
            # add to the list of labels to remove
            labels_to_remove.extend([k for k, v in labels_dict.items()
                                if any(kw.lower() in v.lower() for kw in keywords_to_remove)]) 
    
    segm_arr = get_array_from_image(segm)
    segm_arr_filtered = segm_arr.copy()
    
    if not labels_to_remove == []:
        segm_arr_filtered[np.isin(segm_arr, labels_to_remove)] = 0      # set to 0
    
    segm_filtered = get_image_from_array(segm_arr_filtered, segm)

    removed = {}
    for label in labels_to_remove:      # build dictionary
        count = np.sum(segm_arr == label)
        if count > 0:
            removed[label] = int(count)
    
    return segm_filtered, removed
    

def evaluate_region_wise(mask, gt):
    """
    Returns region-wise evaluation parameters from two binary images.
    The definitions of True Positive Fraction, False Positive Fraction
    and DICE coefficient are taken from Cabezas et al. 'Automatic multiple
    sclerosis lesion detection inbrain MRI by FLAIR thresholding' (2014).
            
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

    if not (set(np.unique(mask_arr)).issubset({0,1})
         and set(np.unique(gt_arr)).issubset({0,1})):
        raise ValueError(f"Mask and ground truth must be binary images containing only 0 and 1")

    ccs_mask, n_mask = connected_components(mask)
    ccs_gt, n_gt = connected_components(gt)
    
    tp = 0
    for label_idx in range(1, n_gt+1): 
        gt_lesion = get_array_from_image(ccs_gt) == label_idx
        if np.any(np.logical_and(gt_lesion, mask_arr)):
            tp += 1         # gt lesion is detected by mask


    fp = 0
    for label_idx in range(1, n_mask+1): 
        mask_lesion = get_array_from_image(ccs_mask) == label_idx
        if not np.any(np.logical_and(mask_lesion, gt_arr)):
            fp += 1         # mask lesion is not positive in gt


    metrics = {
        "TPF": tp / n_gt if n_gt > 0 else 0,
        "FPF": fp / n_mask if n_mask > 0 else 0,
        "DSC": (2 * tp) / (n_mask + n_gt) if (n_mask + n_gt) > 0 else 0,
    }
    metrics = {k: float(v) for k,v in metrics.items()}

    return metrics