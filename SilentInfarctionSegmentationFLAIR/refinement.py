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
        - image (SimpleITK.image): Binary input image (0=background, 1=foreground).
        - connectivity (bool): Connectivity type (6=face, 18=face+edge, 26=face+edge+corner).

    Returns
    -------
        - ccs (SimpleITK.image): Labeled image where each connected component has a unique integer value.
        - n_components (int): Number of connected components found (excluding background).
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
    Computes the minimum and maximum length among any axis (diameter) of the connected components.

    Parameters
    ----------
        - ccs (SimpleITK.image): Connected components labeled image.

    Returns
    -------
        - diameters (dict): Dictionary of minimum and maximum diameters, where each key is a label (int)
            and the value is a tuple (min_diameter, max_diameter) (float).
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
    Filters out connected components whose length among any axis (diameter) is outside of the specified range.

    Parameters
    ----------
        - ccs (SimpleITK.image): Connected components labeled image.
        - lower_thr (float): Minimum connected component length on any axis.
        - upper_thr (float): Maximum connected component length on any axis.

    Returns
    -------
        - ccs_filtered (SimpleITK.image): Labeled image of connected components after filtering.
        - n_components (int): Number of connected components after filtering.
        - removed (dict): Dictionary of removed labels, where each key is a label (int)
            and the value is a tuple (min_diameter, max_diameter) (float).
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