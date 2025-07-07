#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 19:49:24 2025

@author: david
"""

import numpy as np
import SimpleITK as sitk
import os


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
    segm (SimpleITK.Image): FreeSurfer segmentation image.
    labels (numeric or iterable): Iterable of labels to include in the mask.
    
    Returns
    -------
    mask (SimpleITK.Image): Binary mask.
    """
    # if labels is not a list, make it a list
    if isinstance(labels, (int,float)):      
        labels = [labels]            
    elif isinstance(labels, dict):
        labels = list(labels.values())
    else:
        mask = segm == labels[0]         # first label
        for i in range(1,len(labels)):   # other labels (if present)
            mask = sitk.Or(mask, segm == labels[i])     # union
    
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