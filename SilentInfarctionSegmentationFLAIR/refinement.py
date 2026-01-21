#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 13:41:50

@author: david
"""

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
import pandas as pd

from SilentInfarctionSegmentationFLAIR.utils import (
    get_info,
    get_array_from_image,
    get_image_from_array
)


def connected_components(image, connectivity=26):
    """
    Compute connected components in a binary image.

    Parameters
    ----------
    image : SimpleITK.Image
        Binary input image (0 = background, 1 = foreground).
    connectivity : int, optional
        Connectivity type: 6 (faces), 18 (faces + edges),
        26 (faces + edges + corners). Default is 26.

    Returns
    -------
    ccs : SimpleITK.Image
        Labeled image where each connected component has a unique
        integer label.
    n_components : int
        Number of connected components (excluding background).
    """
    if connectivity not in (6, 18, 26):
        raise ValueError("'connectivity' must be 6, 18 or 26")

    arr = get_array_from_image(image)

    if connectivity == 6:
        structure = ndi.generate_binary_structure(3, 1)
    elif connectivity == 18:
        structure = ndi.generate_binary_structure(3, 2)
    else:
        structure = ndi.generate_binary_structure(3, 3)

    ccs_arr, n_components = ndi.label(arr, structure=structure)

    ccs = get_image_from_array(ccs_arr)
    ccs.CopyInformation(image)

    return ccs, n_components


def find_diameters(ccs):
    """
    Compute minimum and maximum physical diameter of each connected
    component.

    Parameters
    ----------
    ccs : SimpleITK.Image
        Labeled connected components image.

    Returns
    -------
    diameters : dict of tuple
        Dictionary mapping each label to a tuple
        ``(min_diameter, max_diameter)`` in millimeters.
    """
    ccs_arr = get_array_from_image(ccs)
    spacing = get_info(ccs)["spacing"]
    diameters = {}

    slices = ndi.find_objects(ccs_arr)

    for label_idx, s in enumerate(slices, start=1):
        if s is None:
            continue

        diams = [(sl.stop - sl.start) * sp for sl, sp in zip(s, spacing)]
        diameters[label_idx] = (min(diams), max(diams))

    return diameters

def diameter_filter(ccs, lower_thr=None, upper_thr=None):
    """
    Filter connected components based on their physical diameters.

    A component is removed if its minimum diameter is below ``lower_thr`` or
    its maximum diameter is above ``upper_thr``.

    Parameters
    ----------
    ccs : SimpleITK.Image
        Labeled connected components image.
    lower_thr : float or None, optional
        Minimum allowed diameter (mm). If None, no lower bound is applied.
    upper_thr : float or None, optional
        Maximum allowed diameter (mm). If None, no upper bound is applied.

    Returns
    -------
    points : pandas.Series
        Series indexed by component label. Value is 1 if the component is
        kept, 0 if removed.
    n_components : int
        Number of components remaining after filtering.
    ccs_filtered : SimpleITK.Image
        Labeled image after removing filtered components.
    """
    ccs_arr = get_array_from_image(ccs)
    spacing = get_info(ccs)["spacing"]
    diameters = find_diameters(ccs)
    removed = {}

    if lower_thr is None and upper_thr is None:
        return ccs, len(np.unique(ccs_arr)) - 1, {}

    if lower_thr is None:
        lower_thr = 0
    if upper_thr is None:
        upper_thr = (np.array(ccs_arr.shape) * spacing).max()

    for label_idx, diams in diameters.items():
        min_d, max_d = diams
        if min_d < lower_thr or max_d > upper_thr:
            removed[label_idx] = (min_d, max_d)

    to_remove = list(removed.keys())
    remove_mask = np.isin(ccs_arr, to_remove)
    ccs_arr[remove_mask] = 0

    ccs_filtered = get_image_from_array(ccs_arr)
    ccs_filtered.CopyInformation(ccs)

    n_components = len(diameters) - len(removed)

    points = pd.Series(
        {label: 0 if label in removed else 1 for label in diameters.keys()}
    )

    return points, n_components, ccs_filtered

def label_filter(segm, labels_to_remove=None, keywords_to_remove=None,
                 labels_dict=None):
    """
    Remove specific labels from a segmentation image.

    Labels can be removed either by specifying their numeric values or by
    providing keywords that match their textual descriptions.

    Parameters
    ----------
    segm : SimpleITK.Image
        Labeled segmentation image.
    labels_to_remove : list of int, optional
        List of numeric labels to remove. Default is None.
    keywords_to_remove : list of str, optional
        Keywords used to match label names in ``labels_dict``.
    labels_dict : dict, optional
        Mapping from numeric labels to textual descriptions. Required if
        ``keywords_to_remove`` is provided.

    Returns
    -------
    segm_filtered : SimpleITK.Image
        Segmentation image after removing selected labels.
    removed : dict
        Dictionary mapping removed label names to the number of voxels
        removed.
    """
    if labels_to_remove is None:
        labels_to_remove = []
    if keywords_to_remove is None:
        keywords_to_remove = []

    if keywords_to_remove:
        if labels_dict is None:
            raise ValueError(
                "If 'keywords_to_remove' is specified, 'labels_dict' "
                "must also be provided."
            )
        matches = [
            k for k, v in labels_dict.items()
            if any(kw.lower() in v.lower() for kw in keywords_to_remove)
        ]
        labels_to_remove.extend(matches)

    segm_arr = get_array_from_image(segm)
    segm_arr_filtered = segm_arr.copy()

    if labels_to_remove:
        segm_arr_filtered[np.isin(segm_arr, labels_to_remove)] = 0

    segm_filtered = get_image_from_array(segm_arr_filtered, segm)

    removed = {}
    for label in labels_to_remove:
        count = int(np.sum(segm_arr == label))
        if count > 0:
            if labels_dict is not None and label in labels_dict:
                name = labels_dict[label]
            else:
                name = str(label)
            removed[name] = count

    return segm_filtered, removed

def pve_filter(ccs, n_components, pves):
    """
    Assign points to connected components based on predominant partial
    volume estimates (PVE).

    Each component receives a score depending on whether white matter (WM),
    gray matter (GM), or cerebrospinal fluid (CSF) is predominant in its
    region.

    Parameters
    ----------
    ccs : SimpleITK.Image
        Labeled connected components image.
    n_components : int
        Number of connected components in ``ccs``.
    pves : list of SimpleITK.Image
        List of PVE images in the order [WM, GM, CSF].

    Returns
    -------
    points : pandas.Series
        Score assigned to each component:
        - +2 → WM predominant
        - +1 → GM predominant
        -  0 → CSF predominant
        - -2 → all PVEs equal to zero
    counts : tuple of int
        Tuple ``(n_wm, n_gm, n_csf, n_zeros)`` with the number of
        components in each category.
    pve_sums : pandas.DataFrame
        Mean PVE values for WM, GM, and CSF for each component.
    """
    pve_wm, pve_gm, pve_csf = pves

    filt = sitk.LabelStatisticsImageFilter()
    filt.Execute(pve_wm, ccs)
    pve_wm_means = [filt.GetMean(l) for l in range(1, n_components + 1)]

    filt = sitk.LabelStatisticsImageFilter()
    filt.Execute(pve_gm, ccs)
    pve_gm_means = [filt.GetMean(l) for l in range(1, n_components + 1)]

    filt = sitk.LabelStatisticsImageFilter()
    filt.Execute(pve_csf, ccs)
    pve_csf_means = [filt.GetMean(l) for l in range(1, n_components + 1)]

    pve_sums = pd.DataFrame(
        {
            "pve_wm": pve_wm_means,
            "pve_gm": pve_gm_means,
            "pve_csf": pve_csf_means,
        },
        index=range(1, n_components + 1),
    )

    max_columns = pve_sums.idxmax(axis=1)
    points = pd.Series(index=pve_sums.index, dtype=int)

    n_wm = n_gm = n_csf = n_zeros = 0

    for idx in pve_sums.index:
        row = pve_sums.loc[idx]
        if row.sum() == 0:
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


def nearly_isotropic_kernel(spacing, desired_radius=1):
    """
    Compute a nearly isotropic kernel radius in voxel units.

    Converts a desired physical radius (in mm) into voxel units for each
    dimension, based on the image spacing.

    Parameters
    ----------
    spacing : sequence of float
        Voxel spacing (e.g., (sx, sy, sz)).
    desired_radius : float, optional
        Desired physical radius in millimeters. Default is 1.

    Returns
    -------
    kernel : list of int
        Kernel radius in voxels for each dimension.
    """
    kernel = []
    for s in spacing:
        kernel.append(max(1, round(desired_radius / s)))
    return kernel

def surrounding_filter(ccs, n_components, pves, dilation_radius=1):
    """
    Assign points to connected components based on surrounding tissue
    composition.

    For each lesion, the surrounding region is defined by morphological
    dilation. Each voxel in this region is assigned to the closest lesion
    using a Voronoi-like partition. Mean PVE values of WM, GM, and CSF are
    then computed for each lesion's surrounding area.

    Parameters
    ----------
    ccs : SimpleITK.Image
        Labeled connected components image.
    n_components : int
        Number of connected components in ``ccs``.
    pves : list of SimpleITK.Image
        List of PVE images in the order [WM, GM, CSF].
    dilation_radius : float, optional
        Physical radius (mm) used for morphological dilation. Default is 1.

    Returns
    -------
    points : pandas.Series
        Score assigned to each component:
        - +2 → WM predominant in surrounding region
        - +1 → GM predominant
        -  0 → CSF predominant
        - -2 → all PVEs equal to zero
    counts : tuple of int
        Tuple ``(n_wm, n_gm, n_csf, n_zeros)`` with the number of components
        in each category.
    pve_sums : pandas.DataFrame
        Mean PVE values for WM, GM, and CSF in the surrounding region.
    """
    pve_wm, pve_gm, pve_csf = pves

    lesion_mask = ccs > 0
    kernel = nearly_isotropic_kernel(ccs.GetSpacing(), dilation_radius)

    dilated = sitk.BinaryDilate(
        lesion_mask,
        kernelRadius=kernel,
        kernelType=sitk.sitkBox
    )
    surround_mask = dilated & ~lesion_mask

    ccs_arr = sitk.GetArrayFromImage(ccs)
    surround_arr = sitk.GetArrayFromImage(surround_mask)

    seed_coords = np.argwhere(ccs_arr > 0)
    seed_labels = ccs_arr[ccs_arr > 0]

    target_coords = np.argwhere(surround_arr > 0)
    tree = cKDTree(seed_coords)
    _, indices = tree.query(target_coords)

    voronoi_arr = np.zeros_like(ccs_arr, dtype=ccs_arr.dtype)
    voronoi_arr[
        target_coords[:, 0],
        target_coords[:, 1],
        target_coords[:, 2]
    ] = seed_labels[indices]

    voronoi_map = sitk.GetImageFromArray(voronoi_arr)
    voronoi_map.CopyInformation(ccs)

    surround_ccs = sitk.Mask(
        voronoi_map,
        sitk.Cast(surround_mask, sitk.sitkUInt8)
    )

    filt = sitk.LabelStatisticsImageFilter()
    filt.Execute(pve_wm, surround_ccs)
    pve_wm_means = [filt.GetMean(l) for l in range(1, n_components + 1)]

    filt = sitk.LabelStatisticsImageFilter()
    filt.Execute(pve_gm, surround_ccs)
    pve_gm_means = [filt.GetMean(l) for l in range(1, n_components + 1)]

    filt = sitk.LabelStatisticsImageFilter()
    filt.Execute(pve_csf, surround_ccs)
    pve_csf_means = [filt.GetMean(l) for l in range(1, n_components + 1)]

    pve_sums = pd.DataFrame(
        {
            "pve_wm": pve_wm_means,
            "pve_gm": pve_gm_means,
            "pve_csf": pve_csf_means,
        },
        index=range(1, n_components + 1),
    )

    points = pd.Series(index=pve_sums.index, dtype=int)
    n_wm = n_gm = n_csf = n_zeros = 0
    max_columns = pve_sums.idxmax(axis=1)

    for idx in pve_sums.index:
        row = pve_sums.loc[idx]
        if row.sum() == 0:
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

def extend_lesions(ccs, n_components, image, n_std=1, dilation_radius=1):
    """
    Extend lesion regions by adding surrounding voxels whose intensity
    exceeds a lesion‑specific threshold.

    The surrounding region is defined by morphological dilation. Each voxel
    in this region is assigned to the closest lesion using a Voronoi-like
    partition. For each lesion, a threshold is computed as:

    ``threshold_i = mean_i - n_std * std_i``

    Voxels in the surrounding region whose intensity exceeds this threshold
    are added to the lesion.

    Parameters
    ----------
    ccs : SimpleITK.Image
        Labeled connected components image (lesions).
    n_components : int
        Number of connected components in ``ccs``.
    image : SimpleITK.Image
        Original intensity image used to compute lesion statistics.
    n_std : float, optional
        Number of standard deviations subtracted from the mean to compute
        the threshold. Lower values produce more aggressive extension.
        Default is 1.
    dilation_radius : float, optional
        Physical radius (mm) used for morphological dilation. Default is 1.

    Returns
    -------
    extended_lesion : SimpleITK.Image
        Binary image containing the original and extended lesions.

    Notes
    -----
    - Each lesion receives its own adaptive threshold.
    - Voronoi assignment ensures each surrounding voxel is associated with
      the closest lesion.
    """
    lesion_mask = ccs > 0
    kernel = nearly_isotropic_kernel(ccs.GetSpacing(), dilation_radius)

    dilated = sitk.BinaryDilate(
        lesion_mask,
        kernelRadius=kernel,
        kernelType=sitk.sitkBox
    )
    surround_mask = dilated & ~lesion_mask

    ccs_arr = sitk.GetArrayFromImage(ccs)
    surround_arr = sitk.GetArrayFromImage(surround_mask)

    seed_coords = np.argwhere(ccs_arr > 0)
    seed_labels = ccs_arr[ccs_arr > 0]

    target_coords = np.argwhere(surround_arr > 0)
    tree = cKDTree(seed_coords)
    _, indices = tree.query(target_coords)

    voronoi_arr = np.zeros_like(ccs_arr, dtype=ccs_arr.dtype)
    voronoi_arr[
        target_coords[:, 0],
        target_coords[:, 1],
        target_coords[:, 2]
    ] = seed_labels[indices]

    voronoi_map = sitk.GetImageFromArray(voronoi_arr)
    voronoi_map.CopyInformation(ccs)

    surround_ccs = sitk.Mask(
        voronoi_map,
        sitk.Cast(surround_mask, sitk.sitkUInt8)
    )

    filt = sitk.LabelStatisticsImageFilter()
    filt.Execute(image, ccs)

    means = np.array([filt.GetMean(l) for l in range(1, n_components + 1)])
    stds = np.array([filt.GetSigma(l) for l in range(1, n_components + 1)])
    thrs = means - n_std * stds

    surround_arr = get_array_from_image(surround_ccs)
    thrs_with_zero = np.concatenate([[0], thrs])
    thrs_arr = np.take(thrs_with_zero, surround_arr)

    image_arr = get_array_from_image(sitk.Mask(image, surround_mask))
    extended_lesion_arr = (image_arr > thrs_arr).astype(np.uint8)

    extended_lesion = get_image_from_array(
        extended_lesion_arr,
        lesion_mask
    )
    extended_lesion = (
        sitk.Cast(lesion_mask, sitk.sitkUInt8) | extended_lesion
    )

    return extended_lesion

def evaluate_region_wise(mask, gt):
    """
    Compute region‑wise detection metrics between a predicted mask and
    ground‑truth lesions.

    Metrics follow the definitions in:
    Cabezas et al., *Automatic multiple sclerosis lesion detection in brain
    MRI by FLAIR thresholding* (2014).

    A lesion is considered detected if at least one voxel of the predicted
    mask overlaps with it.

    Parameters
    ----------
    mask : SimpleITK.Image
        Binary prediction mask to evaluate.
    gt : SimpleITK.Image
        Binary ground‑truth lesion mask.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - ``rw-TPF`` : True Positive Fraction (detected GT lesions / total GT)
        - ``rw-FPF`` : False Positive Fraction (spurious predicted lesions /
                       total predicted lesions)
    """
    gt_arr = get_array_from_image(gt)
    mask_arr = get_array_from_image(mask)

    ccs_mask, n_mask = connected_components(mask)
    ccs_mask_arr = get_array_from_image(ccs_mask)

    ccs_gt, n_gt = connected_components(gt)
    ccs_gt_arr = get_array_from_image(ccs_gt)

    # GT lesions that intersect the predicted mask
    gt_in_mask = ccs_gt_arr[mask_arr > 0]
    tp = len(np.unique(gt_in_mask[gt_in_mask > 0]))

    # Predicted lesions that do NOT intersect any GT lesion
    mask_in_gt = ccs_mask_arr[gt_arr > 0]
    mask_not_touching_gt = ccs_mask_arr[
        ~np.isin(ccs_mask_arr, mask_in_gt)
    ]
    fp = len(np.unique(mask_not_touching_gt[mask_not_touching_gt > 0]))

    metrics = {
        "rw-TPF": tp / n_gt if n_gt > 0 else 0.0,
        "rw-FPF": fp / n_mask if n_mask > 0 else 0.0,
    }

    return {k: float(v) for k, v in metrics.items()}
