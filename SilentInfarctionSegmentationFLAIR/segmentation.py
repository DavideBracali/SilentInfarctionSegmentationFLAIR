#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 19:49:24 2025

@author: david
"""

import numpy as np
import SimpleITK as sitk


from utils import DimensionError
from utils import check_3d
from utils import get_info
from utils import get_array_from_image
from utils import plot_image
from utils import orient_image
from utils import resample_to_reference

gm_labels = [3, 8, 10, 11, 12, 13, 17, 18, 19, 42, 47, 49, 50, 51, 52, 53, 54, 55]
segm = sitk.ReadImage("SilentInfarctionSegmentationFLAIR/Examples/aseg.auto_noCCseg.nii") #°_° to remove


def get_mask_from_segmentation(segm, labels=gm_labels):
    """
    Returns a binary mask from a FreeSurfer segmentation image.
    
    Parameters
    ----------
    segm (SimpleITK.Image): FreeSurfer segmentation image.
    labels (list): List of integer labels to include in the mask.
    
    Returns
    -------
    mask (SimpleITK.Image): Binary mask.
    """
    segm_array = get_array_from_image(segm)
