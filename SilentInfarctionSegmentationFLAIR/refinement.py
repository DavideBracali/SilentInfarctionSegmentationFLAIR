#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 13:41:50

@author: david
"""

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi
import pandas as pd
import time

from SilentInfarctionSegmentationFLAIR.utils import get_info
from SilentInfarctionSegmentationFLAIR.utils import get_array_from_image
from SilentInfarctionSegmentationFLAIR.utils import get_image_from_array
from SilentInfarctionSegmentationFLAIR.utils import progress_bar


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
        structure = ndi.generate_binary_structure(3, 1)  # faces
    elif connectivity == 18:
        structure = ndi.generate_binary_structure(3, 2)  # faces + edges
    elif connectivity == 26:
        structure = ndi.generate_binary_structure(3, 3)  # faces + edges + corners

    ccs_arr, n_components = ndi.label(arr, structure=structure)

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

    slices = ndi.find_objects(ccs_arr)      # returns bounding boxes
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
        - points (pandas.Series): Index corresponds to connected component label,
            value is set to 0 for filtered components, otherwise to 1.
        - ccs_filtered (SimpleITK.image): Labeled image of connected components
            after filtering.
        - n_components (int): Number of connected components after filtering.
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
    remove_mask = np.isin(ccs_arr, to_remove)
    ccs_arr[remove_mask] = 0

    ccs_filtered = get_image_from_array(ccs_arr)
    ccs_filtered.CopyInformation(ccs)
    n_components = len(diameters) - len(removed)

    # assign 1 point to ccs that are not removed
    points = pd.Series({label: 0 if label in removed else 1
                        for label in diameters.keys()})

    return points, n_components, ccs_filtered


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
        - removed (dict): Dictionary of removed voxels, where each key is a text
            label (str), and each value contains the number of removed voxels
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
            if labels_dict is not None and label in labels_dict.keys():
                label_name = labels_dict[label]
            else:
                label_name = str(label)

            removed[label_name] = int(count)
            
    return segm_filtered, removed
    
def pve_filter(ccs, n_components, pves):
    """
    Assigns points to connected components based on predominant partial volume estimates (PVE)
    of white matter (WM), gray matter (GM), and cerebrospinal fluid (CSF).

    Parameters
    ----------
        - ccs (SimpleITK.image): Connected components labeled image.
        - n_components (int): Number of connected components in `ccs`.
        - pves (list of SimpleITK.image): List of PVE images in the format
            [pve_wm, pve_gm, pve_csf]

    Returns
    -------
        - points (pandas.Series): Index corresponds to connected component label.
            Values are assigned as follows:
                - +2 → WM is predominant
                - +1 → GM is predominant
                -  0 → CSF is predominant
                - -2 → all PVEs are zero
        - (n_wm, n_gm, n_csf, n_zeros) (tuple): Counts of connected components
            assigned to each category.
        - pve_sums (pandas.DataFrame): Mean PVE values of WM, GM, and CSF
            in the surrounding region for each component.
    """
    pve_wm = pves[0]
    pve_gm = pves[1]
    pve_csf = pves[2]
    
    filt = sitk.LabelStatisticsImageFilter()
    filt.Execute(pve_wm, ccs)
    pve_wm_means = [filt.GetMean(l) for l in range(1, n_components + 1)]

    filt = sitk.LabelStatisticsImageFilter()
    filt.Execute(pve_gm, ccs)
    pve_gm_means = [filt.GetMean(l) for l in range(1, n_components + 1)]

    filt = sitk.LabelStatisticsImageFilter()
    filt.Execute(pve_csf, ccs)
    pve_csf_means = [filt.GetMean(l) for l in range(1, n_components + 1)]

    pve_sums = pd.DataFrame({
        "pve_wm": pve_wm_means,
        "pve_gm": pve_gm_means,
        "pve_csf": pve_csf_means
    }, index=range(1, n_components + 1))

    max_columns = pve_sums.idxmax(axis=1)
    points = pd.Series(index=pve_sums.index, dtype=int)
    n_zeros = 0
    n_csf = 0
    n_gm = 0
    n_wm = 0
    for idx in pve_sums.index:
        if (pve_sums.loc[idx].sum() == 0):
            points[idx] = -2      # -2 points if all pves are 0
            n_zeros += 1
        elif max_columns[idx] == "pve_wm":
            points[idx] = 2       # +2 points if wm is predominant
            n_wm +=1
        elif max_columns[idx] == "pve_gm":
            points[idx] = 1       # +1 points if gm is predominant
            n_gm += 1
        else: 
            points[idx] = 0       # 0 points if csf is predominant
            n_csf += 1
    
    return points, (n_wm, n_gm, n_csf, n_zeros), pve_sums


def nearly_isotropic_kernel(spacing, desired_radius):

    """
    Computes a nearly isotropic kernel radius in voxel units, given the image spacing.
    Tries to obtain a kernel radius as close as possible to the desired one.

    Parameters
    ----------
        - spacing (tuple or list of float): The voxel spacing of the image,
            e.g. (x_spacing, y_spacing, z_spacing).
        - desired_radius (float, optional): Desired physical radius (in mm) of the kernel.

    Returns
    -------
        - kernel (list of int): The kernel radius for each dimension, adjusted to voxel units.
    """
    
    kernel = []

    for s in spacing:
        kernel.append(max(1, round(desired_radius / s)))
    return kernel


def surrounding_filter(ccs, n_components, pves, dilation_radius = 1):
    
    """
    Assigns points to connected components based on the predominant partial volume estimates (PVE)
    of surrounding tissue (WM, GM, CSF) in the neighborhood of each lesion.

    Parameters
    ----------
        - ccs (SimpleITK.Image): Connected components labeled image.
        - n_components (int): Number of connected components in `ccs`.
        - pves (list of SimpleITK.Image): List of PVE images in the format
            [pve_wm, pve_gm, pve_csf].
        - dilation_radius (float, optional): Physical radius (in mm) for dilation
            used to define the surrounding region. Default is 1.

    Returns
    -------
        - points (pandas.Series): Index corresponds to connected component label.
            Values are assigned as follows:
                - +2 → WM is predominant in the surrounding region
                - +1 → GM is predominant in the surrounding region
                -  0 → CSF is predominant in the surrounding region
                - -2 → all PVEs are zero in the surrounding region
        - (n_wm, n_gm, n_csf, n_zeros) (tuple): Counts of connected components
            assigned to each category.
        - pve_sums (pandas.DataFrame): Mean PVE values of WM, GM, and CSF
            in the surrounding region for each component.
    """
    pve_wm = pves[0]
    pve_gm = pves[1]
    pve_csf = pves[2]

    # surrounding of all lesions
    lesion_mask = ccs > 0
    kernel = nearly_isotropic_kernel(ccs.GetSpacing(), dilation_radius)
    dilated = sitk.BinaryDilate(lesion_mask, kernelRadius=kernel, kernelType=sitk.sitkBox)
    surround_mask = dilated & ~lesion_mask

    # assign each voxel to the closest region
    filter = sitk.DanielssonDistanceMapImageFilter()
    filter.SetInputIsBinary(False)
    filter.SetUseImageSpacing(True)
    filter.Execute(ccs)
    voronoi_map = filter.GetVoronoiMap()        # closest ccs for each voxel

    # assign each surrounding to the closest lesion
    surround_ccs = sitk.Mask(voronoi_map, sitk.Cast(surround_mask, sitk.sitkUInt8))

    # compute statistics
    filt = sitk.LabelStatisticsImageFilter()
    filt.Execute(pve_wm, surround_ccs)
    pve_wm_means = [filt.GetMean(l) for l in range(1, n_components + 1)]

    filt = sitk.LabelStatisticsImageFilter()
    filt.Execute(pve_gm, surround_ccs)
    pve_gm_means = [filt.GetMean(l) for l in range(1, n_components + 1)]

    filt = sitk.LabelStatisticsImageFilter()
    filt.Execute(pve_csf, surround_ccs)
    pve_csf_means = [filt.GetMean(l) for l in range(1, n_components + 1)]

    pve_sums = pd.DataFrame({
        "pve_wm": pve_wm_means,
        "pve_gm": pve_gm_means,
        "pve_csf": pve_csf_means
    }, index=range(1, n_components + 1))

    # assign points
    points = pd.Series(index=pve_sums.index, dtype=int)
    n_zeros = n_csf = n_gm = n_wm = 0
    max_columns = pve_sums.idxmax(axis=1)

    for idx in pve_sums.index:
        if pve_sums.loc[idx].sum() == 0:
            points[idx] = -2
            n_zeros += 1
        elif max_columns[idx] == "pve_wm":
            points[idx] = 2
            n_wm += 1
        elif max_columns[idx] == "pve_gm":
            points[idx] = 1
            n_gm += 1
        else:
            points[idx] = 0
            n_csf += 1

    return points, (n_wm, n_gm, n_csf, n_zeros), pve_sums


def gaussian_transform(image, mean, std, return_float = False, normalized = False):
    """
    Apply a Gaussian weighting to the voxel intensities of a SimpleITK image.

    Each voxel intensity `I` is transformed according to:
        G(I) = exp(-0.5 * ((I - mean) / std)²) * I
    If `return_float=True` and `normalized=True`, the result is further scaled by:
        1 / (std * sqrt(2π))

    Parameters
    ----------
    image : SimpleITK.Image
        Input image with non-negative voxel intensities.
    mean : float
        Mean of the Gaussian distribution (center of enhancement).
    std : float
        Standard deviation of the Gaussian (controls the enhancement width).
    return_float : bool, optional (default=False)
        If True, return a float-valued image between 0 and 1.
        If False, the result is cast back to the original voxel type.
    normalized : bool, optional (default=False)
        If True and `return_float=True`, apply the normalization factor
        1 / (std * sqrt(2π)).

    Returns
    -------
    SimpleITK.Image
        The Gaussian-weighted image. Spacing, origin and direction are
        preserved from the input image.

    Raises
    ------
    ValueError
        If the input image contains negative voxel intensities.
    """
    arr = get_array_from_image(image)

    if np.any(arr < 0):
        raise ValueError("The input image contains negative values.")

    if return_float:
        if normalized:
            norm = 1 / (std * np.sqrt(2*np.pi))
        else:
            norm = 1
        # return transformed image of floats between 0 and 1 
        gaussian_arr = norm * np.exp(-0.5 * ((arr - mean) / std)**2)
        return get_image_from_array(gaussian_arr, image, cast_to_reference=False)

    else:
        gaussian_arr = np.exp(-0.5 * ((arr - mean) / std)**2) * arr
        # return transformed image with the same voxel type 
        return get_image_from_array(gaussian_arr, image, cast_to_reference=True)



def extend_lesions(ccs, n_components, image, n_std=1, dilation_radius=1):
    
    # surrounding of all lesions
    lesion_mask = ccs > 0
    kernel = nearly_isotropic_kernel(ccs.GetSpacing(), dilation_radius)
    dilated = sitk.BinaryDilate(lesion_mask, kernelRadius=kernel, kernelType=sitk.sitkBox)
    surround_mask = dilated & ~lesion_mask

    # assign each voxel to the closest region
    filter = sitk.DanielssonDistanceMapImageFilter()
    filter.SetInputIsBinary(False)
    filter.SetUseImageSpacing(True)
    filter.Execute(ccs)
    voronoi_map = filter.GetVoronoiMap()        # closest ccs for each voxel

    # assign each surrounding to the closest lesion
    surround_ccs = sitk.Mask(voronoi_map, sitk.Cast(surround_mask, sitk.sitkUInt8))

    # compute statistics
    filt = sitk.LabelStatisticsImageFilter()
    filt.Execute(image, ccs)
    means = np.array([filt.GetMean(l)
                      for l in range(1, n_components + 1)])
    stds = np.array([filt.GetSigma(l)
                     for l in range(1, n_components + 1)])
    thrs = means - n_std*stds

    surround_arr = get_array_from_image(surround_ccs)
    thrs_with_zero = np.concatenate([[0], thrs])  # thrs_with_zero[0] = 0
    # if surround_arr[x,y,z] = i --> threshold_arr[x,y,z] = thrs_with_zero[i]
    thrs_arr = np.take(thrs_with_zero, surround_arr)

    image_arr = get_array_from_image(sitk.Mask(image, surround_mask))
    extended_lesion_arr = (image_arr > thrs_arr).astype(np.uint8)
    extended_lesion = get_image_from_array(extended_lesion_arr, lesion_mask)
    extended_lesion = sitk.Cast(lesion_mask, sitk.sitkUInt8) | extended_lesion

    return extended_lesion






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
    gt_arr = get_array_from_image(gt)
    mask_arr = get_array_from_image(mask)

    ccs_mask, n_mask = connected_components(mask)
    ccs_mask_arr = get_array_from_image(ccs_mask)
    ccs_gt, n_gt = connected_components(gt)
    ccs_gt_arr = get_array_from_image(ccs_gt)
    
    # (labeled) gt voxels that are detected
    gt_in_mask = ccs_gt_arr[mask_arr > 0]
    # number of gt lesions that are detected  
    tp = len(np.unique(gt_in_mask[gt_in_mask > 0]))               

    # (labeled) mask voxels that are positive in gt
    mask_in_gt = ccs_mask_arr[gt_arr > 0]
    # (labeled) mask voxels that do not intersect any gt voxel
    mask_not_touching_gt = ccs_mask_arr[~np.isin(ccs_mask_arr, mask_in_gt)]
    # number of mask lesions that do not intersect any gt voxel
    fp = len(np.unique(mask_not_touching_gt[mask_not_touching_gt > 0]))
    

    metrics = {
        "rw-TPF": tp / n_gt if n_gt > 0 else 0,
        "rw-FPF": fp / n_mask if n_mask > 0 else 0,
    }
    
    metrics = {k: float(v) for k,v in metrics.items()}

    return metrics