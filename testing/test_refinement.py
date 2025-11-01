#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 15:33:12

@author: david
"""

import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, assume
import hypothesis.extra.numpy as stnp

import sys
import os
import numpy as np
import SimpleITK as sitk
import matplotlib
import pandas as pd

matplotlib.use('Agg')
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                            '..', 'SilentInfarctionSegmentationFLAIR')))

from SilentInfarctionSegmentationFLAIR.utils import get_array_from_image
from SilentInfarctionSegmentationFLAIR.utils import resample_to_reference
from SilentInfarctionSegmentationFLAIR.refinement import connected_components
from SilentInfarctionSegmentationFLAIR.refinement import find_diameters
from SilentInfarctionSegmentationFLAIR.refinement import diameter_filter
from SilentInfarctionSegmentationFLAIR.refinement import label_filter
from SilentInfarctionSegmentationFLAIR.refinement import evaluate_region_wise
from SilentInfarctionSegmentationFLAIR.refinement import pve_filter
from SilentInfarctionSegmentationFLAIR.refinement import nearly_isotropic_kernel
from SilentInfarctionSegmentationFLAIR.refinement import surrounding_filter
from SilentInfarctionSegmentationFLAIR.refinement import gaussian_transform
from SilentInfarctionSegmentationFLAIR.refinement import extend_lesions







####################
###  STRATEGIES  ###
####################

@st.composite
def four_voxel_strategy(draw):
    """
    Generates a 3D binary image with 4 active voxels:
    - v1 = (i, i, i)
    - v2 = (i, i, i+1)      # 6-connected to v1 (face)
    - v3 = (i+1, i+1, i+1)  # 18-connected to v2 (edge)
    - v4 = (i+2, i+2, i+2)  # 26-connected to v3 (corner)
    """

    size = (10, 10, 10)
    i = draw(st.integers(0, size[0] - 3))
    
    # 0s
    arr = np.zeros(size, dtype=np.uint8)

    # 1s
    v1 = (i, i, i)
    v2 = (i, i, i+1)
    v3 = (i+1, i+1, i+1)
    v4 = (i+2, i+2, i+2)
    
    arr[v1] = 1
    arr[v2] = 1
    arr[v3] = 1
    arr[v4] = 1
    
    # draw image
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((1.0, 1.0, 1.0))

    return image



@st.composite
def labeled_image_strategy(draw):
    """
    Generates a small 3D labeled image with two connected components:
    - Component 1: a 2×2×2 cube (label 1)
    - Component 2: a 1×1×3 line (label 2)
    Positioned randomly within the array without overlap.
    """
    size = (20, 20, 20)
    arr = np.zeros(size, dtype=np.uint8)

    # component 1
    i1 = draw(st.integers(0, 8))
    j1 = draw(st.integers(0, 8))
    k1 = draw(st.integers(0, 8))
    arr[i1:i1+2, j1:j1+2, k1:k1+2] = 1

    # component 2
    axis = draw(st.sampled_from([0, 1, 2]))         # random orientation
    if axis == 0:
        i2 = draw(st.integers(10, size[0] - 3))
        j2 = draw(st.integers(10, size[1] - 1))
        k2 = draw(st.integers(10, size[2] - 1))
        arr[i2:i2+3, j2, k2] = 2
    elif axis == 1:
        i2 = draw(st.integers(10, size[0] - 1))
        j2 = draw(st.integers(10, size[1] - 3))
        k2 = draw(st.integers(10, size[2] - 1))
        arr[i2, j2:j2+3, k2] = 2
    else:  
        i2 = draw(st.integers(10, size[0] - 1))
        j2 = draw(st.integers(10, size[1] - 1))
        k2 = draw(st.integers(10, size[2] - 3))
        arr[i2, j2, k2:k2+3] = 2

    # draw image
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((1.0, 1.0, 1.0))

    return image


@st.composite
def binary_mask_pair_strategy(draw, size=(32, 32, 32)):
    """
    Generate a pair of binary masks (SimpleITK images) of given size.
    """
    shape = size
    arr1 = draw(stnp.arrays(dtype=np.uint8, shape=shape, elements=st.integers(0, 1)))
    arr2 = draw(stnp.arrays(dtype=np.uint8, shape=shape, elements=st.integers(0, 1)))

    img1 = sitk.GetImageFromArray(arr1)
    img2 = sitk.GetImageFromArray(arr2)

    img1.CopyInformation(img2)

    return img1, img2


@st.composite
def non_binary_image_strategy(draw):
    """
    Generate a non-binary SimpleITK image with gray levels from 10 to 15.
    """
    shape = draw(st.tuples(st.integers(5, 20), st.integers(5, 20), st.integers(5, 20)))
    arr = draw(stnp.arrays(dtype=np.int32, shape=shape, elements=st.integers(10, 15)))
    img = sitk.GetImageFromArray(arr)

    return img

@st.composite
def blurred_sphere_strategy(draw):
    """
    Generates a 3D SimpleITK image with:
    - a spherical bright region (lesion core)
    - smoothly fading intensity toward the background (Gaussian-like falloff)
    
    Useful for testing functions that rely on local intensity thresholds
    or lesion extension (e.g., extend_lesions).
    """

    size = draw(st.tuples(st.integers(30, 50), st.integers(30, 50), st.integers(30, 50)))
    radius = draw(st.integers(5, min(size) // 4))
    center = (
        draw(st.integers(radius + 2, size[0] - radius - 2)),
        draw(st.integers(radius + 2, size[1] - radius - 2)),
        draw(st.integers(radius + 2, size[2] - radius - 2)))

    z, y, x = np.ogrid[:size[0], :size[1], :size[2]]
    dist = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)

    sigma = draw(st.floats(min_value=radius / 2, max_value=radius * 1.5))
    intensity_peak = draw(st.floats(min_value=80.0, max_value=150.0))
    arr = intensity_peak * np.exp(-(dist**2) / (2 * sigma**2))

    arr = arr.astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))

    return img



##################
###  TESTING   ###
##################

@given(four_voxel_strategy(), st.sampled_from([6, 18, 26]))
@settings(max_examples=5, deadline=None)
def test_connected_components_valid_return(image, connectivity):
    """
    Given:
        - 4 voxel binary image (see four_voxel_strategy docstring for details)
        - 6, 18 or 26 connectivity
    Then:
        - find connected components
    Assert:
        - returned number of components is an integer >=1
        - unique values in the labels are consistent with the number of components
    """
    labels, n_components = connected_components(image, connectivity=connectivity)
    labels_arr = sitk.GetArrayFromImage(labels)
    unique_labels = np.unique(labels_arr)   # background is not included

    assert isinstance(n_components, int)
    assert n_components >= 1
    assert len(unique_labels) == n_components + 1


@given(four_voxel_strategy())
@settings(max_examples=5, deadline=None)
def test_connected_components_connectivity_type(image):
    """
    Given:
        - 4 voxel binary image (see four_voxel_strategy docstring for details)
    Then:
        - find connected components with connectivity 6, 18 and 26
    Assert:
        - when connectivity is 6 there are 3 connected components 
        - when connectivity is 18 there are 2 connected components
        - when connectivity is 26 there is 1 connected component
    """

    _, n6 = connected_components(image, connectivity=6)
    _, n18 = connected_components(image, connectivity=18)
    _, n26 = connected_components(image, connectivity=26)

    assert n6 == 3
    assert n18 == 2
    assert n26 == 1


@given(labeled_image_strategy())
@settings(max_examples=5, deadline=None)
def test_find_diameters_valid_return(ccs):
    """
    Given:
        - a labeled image (see labeled_image_strategy for details)
    Then:
        - find diameters
    Assert:
        - a 2-element dict of tuples is returned
    """
    diameters = find_diameters(ccs)

    assert isinstance(diameters, dict)
    for val in diameters.values():
        assert isinstance(val, tuple)
    assert len(diameters) == 2


@given(labeled_image_strategy())
@settings(max_examples=5, deadline=None)
def test_find_diameters_expected_results(ccs):
    """
    Given:
        - a labeled image (see labeled_image_strategy for details)
    Then:
        - find diameters
    Assert:
        - returned diameters are (2,2) and (1,3)
    """
    diameters = find_diameters(ccs)

    assert diameters == {1 : (2,2), 2 : (1,3)}



@given(labeled_image_strategy(), st.floats(0, 1.99), st.floats(2.01, 4))
@settings(max_examples=5, deadline=None)
def test_diameter_filter_valid_return(ccs, lower_thr, upper_thr):
    """
    Given:
        - a labeled image (see labeled_image_strategy for details)
        - random lower and upper thresholds (lower_thr < 2 and upper_thr > 2)
    Then:
        - filter diameters with random thresholds
    Assert:
        - a SimpleITK image is returned
        - labels in the returned image after filtering are a subset
            or equal of the labels before filtering
        - number of components after filtering is an integer
        - number of components after filtering is <= than before filtering
        - returned points is a series
        - the cube is never removed (1 point for label 1)
    """

    points, filtered_n, filtered_ccs = diameter_filter(ccs, lower_thr, upper_thr)
    labels = set(np.unique(get_array_from_image(ccs)))
    n = len(labels) - 1         # no background
    filtered_labels = set(np.unique(get_array_from_image(filtered_ccs)))


    assert isinstance(filtered_ccs, sitk.Image)
    assert filtered_labels.issubset(labels)
    assert filtered_n <= n
    assert isinstance(points, pd.Series)
    assert points[1] == 1


@given(labeled_image_strategy())
@settings(max_examples=5, deadline=None)
def test_diameter_filter_expected_results(ccs):
    """
    Given:
        - a labeled image (see labeled_image_strategy for details)
    Then:
        - filter diameters with 1.5 as lower threshold (case 1)
        - filter diameters with 2.5 as upper threshold (case 2)
        - filter diameters with 1.5 as lower and 2.5 as upper threshold (case 3)
        - filter diameters with 1 as lower threshold (case 4)
        - filter diameters with 3 as upper threshold (case 5)
        - filter diameters with 1 as lower and 3 as higher threshold (case 6)
        - filter diameters with 1 as upper threshold (case 7)
        - filter diameters with 3 as lower threshold (case 8)
        - filter diameters with 3 as lower and 1 as upper threshold (case 9)
    Assert:
        - the line is removed in cases 1,2,3
        - nothing is removed in cases 4,5,6
        - everything is removed in cases 7,8,9
    """
    points1,_,_ = diameter_filter(ccs, lower_thr=1.5)
    points2,_,_ = diameter_filter(ccs, upper_thr=2.5)
    points3,_,_ = diameter_filter(ccs, lower_thr=1.5, upper_thr=2.5)
    points4,_,_ = diameter_filter(ccs, lower_thr=1)
    points5,_,_ = diameter_filter(ccs, upper_thr=3)
    points6,_,_ = diameter_filter(ccs, lower_thr=1, upper_thr=3)
    points7,_,_ = diameter_filter(ccs, lower_thr=3)
    points8,_,_ = diameter_filter(ccs, upper_thr=1)
    points9,_,_ = diameter_filter(ccs, lower_thr=3, upper_thr=1)

    assert points1.equals(pd.Series({1:1, 2:0}))
    assert points2.equals(pd.Series({1:1, 2:0}))
    assert points3.equals(pd.Series({1:1, 2:0}))
    assert points4.equals(pd.Series({1:1, 2:1}))
    assert points5.equals(pd.Series({1:1, 2:1}))
    assert points6.equals(pd.Series({1:1, 2:1}))
    assert points7.equals(pd.Series({1:0, 2:0}))
    assert points8.equals(pd.Series({1:0, 2:0}))
    assert points9.equals(pd.Series({1:0, 2:0}))


@given(labeled_image_strategy(),
    st.lists(st.integers(min_value=1, max_value=2), unique=True))
@settings(max_examples=5, deadline=None)
def test_label_filter_valid_return(segm, labels_to_remove):
    """
    Given:
        - a labeled image (see labeled_image_strategy for details)
        - a random subset of labels to remove
    Then:
        - filter labels
    Assert:
        - segm_filtered is a SimpleITK image
        - removed is a dict with str keys and int values
    """
    segm_filtered, removed = label_filter(segm, labels_to_remove=labels_to_remove)

    assert isinstance(segm_filtered, sitk.Image)
    assert isinstance(removed, dict)
    assert all(isinstance(k, str) for k in removed.keys())
    assert all(isinstance(v, int) and v >= 0 for v in removed.values())


@given(labeled_image_strategy(),
    st.lists(st.integers(min_value=1, max_value=2), unique=True))
@settings(max_examples=5, deadline=None)
def test_label_filter_properties(segm, labels_to_remove):
    """
    Given:
        - a labeled image (see labeled_image_strategy for details)
        - a random subset of labels to remove
    Then:
        - filter labels
    Assert:
        - specified labels are set to 0 in the filtered image
        - only specified labels are removed, others unchanged
        - removed counts are consistent with the original image
        - applying filter twice does not change the result
    """
    segm_filtered, removed = label_filter(segm, labels_to_remove=labels_to_remove)
    arr = get_array_from_image(segm)
    arr_filtered = get_array_from_image(segm_filtered)

    segm_filtered2, removed2 = label_filter(segm_filtered, labels_to_remove=labels_to_remove)

    labels_to_keep = set(np.unique(arr)) - set(labels_to_remove) - {0}
    
    for label in labels_to_remove:
        assert np.all(arr_filtered[arr == int(label)] == 0)
    for label in labels_to_keep:
        assert np.array_equal(arr_filtered[arr == int(label)], arr[arr == int(label)])
    for label, count in removed.items():
        assert count == np.sum(arr == int(label))
    assert np.array_equal(get_array_from_image(segm_filtered), get_array_from_image(segm_filtered2))
    assert removed2 == {}


@given(
    labeled_image_strategy(),   
    stnp.arrays(dtype=np.float32, shape=(20,20,20), elements=st.floats(0, 1)),
    stnp.arrays(dtype=np.float32, shape=(20,20,20), elements=st.floats(0, 1)),
    stnp.arrays(dtype=np.float32, shape=(20,20,20), elements=st.floats(0, 1))
)
@settings(max_examples=5, deadline=None)
def test_pve_filter_valid_return(ccs, wm_arr, gm_arr, csf_arr):
    """
    Given:
        - ccs: labeled image
        - wm_arr, gm_arr, csf_arr: partial volume estimate images
    Then:
        - compute points
    Assert that:
        - points is a pandas Series with index = labels in ccs
            and values are in {-2,0,1,2}
        - summary is a 4 elements tuple
        - all of the elements in summary are integers <= n_components
        - pve_sums is a pandas DataFrame with index = labels in ccs
    """
    wm_img = sitk.GetImageFromArray(wm_arr)
    gm_img = sitk.GetImageFromArray(gm_arr)
    csf_img = sitk.GetImageFromArray(csf_arr)
    
    wm_img.CopyInformation(ccs)
    gm_img.CopyInformation(ccs)
    csf_img.CopyInformation(ccs)
    
    labels_arr = sitk.GetArrayFromImage(ccs)
    n_components = len(np.unique(labels_arr)) - 1
    
    points, summary, pve_sums = pve_filter(ccs, n_components, [wm_img, gm_img, csf_img])
    
    assert set(points.index) == set(range(1, n_components+1))
    assert set(points.values).issubset({-2,0,1,2})
    assert isinstance(summary, tuple)
    assert len(summary) == 4
    for n in summary:
        assert isinstance(n, int)
        assert n <= n_components
    assert isinstance(pve_sums, pd.DataFrame)
    assert set(pve_sums.index) == set(range(1, n_components+1))


@given(
    labeled_image_strategy(),   
    stnp.arrays(dtype=np.float32, shape=(20,20,20), elements=st.floats(0, 1)),
    stnp.arrays(dtype=np.float32, shape=(20,20,20), elements=st.floats(0, 1)),
    stnp.arrays(dtype=np.float32, shape=(20,20,20), elements=st.floats(0, 1))
)
@settings(max_examples=5, deadline=None)
def test_pve_filter_expected_results(ccs, wm_arr, gm_arr, csf_arr):
    """
    Given:
        - ccs: labeled image
        - wm_arr, gm_arr, csf_arr: partial volume estimate images
    Then:
        - compute points
    Assert that:
        - points are assigned as expected
    """
    wm_img = sitk.GetImageFromArray(wm_arr)
    gm_img = sitk.GetImageFromArray(gm_arr)
    csf_img = sitk.GetImageFromArray(csf_arr)
    
    wm_img.CopyInformation(ccs)
    gm_img.CopyInformation(ccs)
    csf_img.CopyInformation(ccs)
    
    labels_arr = sitk.GetArrayFromImage(ccs)
    n_components = len(np.unique(labels_arr)) - 1
    
    points, _, _ = pve_filter(ccs, n_components, [wm_img, gm_img, csf_img])
    
    for label in points.index:
        mask = labels_arr == label
        sums = [wm_arr[mask].sum(), gm_arr[mask].sum(), csf_arr[mask].sum()]
        if sum(sums) == 0:
            assert points[label] == -2
        else:
            max_idx = np.argmax(sums)
            expected = [2,1,0][max_idx]  # wm, gm, csf
            assert points[label] == expected


@given(
    spacing=st.lists(st.floats(min_value=0.1, max_value=10), min_size=1, max_size=5),
    desired_radius=st.floats(min_value=0.1, max_value=20)
)
@settings(max_examples=20, deadline=None)
def test_nearly_isotropic_kernel_valid_return(spacing, desired_radius):
    """
    Given:
        - spacing: list of positive floats
        - desired_radius: positive float
    Then:
        - compute nearly isotropic kernel
    Assert that:
        - output is a list of integers
        - length of output == length of spacing
        - each element >= 0
    """
    kernel = nearly_isotropic_kernel(spacing, desired_radius)

    assert isinstance(kernel, list)
    assert all(isinstance(k, int) for k in kernel)
    assert len(kernel) == len(spacing)
    assert all(k >= 0 for k in kernel)


@given(
    spacing=st.lists(st.floats(min_value=0.1, max_value=10), min_size=1, max_size=3),
    desired_radius=st.floats(min_value=0.1, max_value=20)
)
@settings(max_examples=20, deadline=None)
def test_nearly_isotropic_kernel_expected_behavior(spacing, desired_radius):
    """
    Given:
        - spacing: list of positive floats
        - desired_radius: positive float
    Then:
        - compute nearly isotropic kernel
    Assert that:
        - each element is close to desired_radius / spacing[i], rounded
    """
    kernel = nearly_isotropic_kernel(spacing, desired_radius)

    for k, s in zip(kernel, spacing):
        expected = max(1, round(desired_radius / s))
        assert k == expected


@given(
    labeled_image_strategy(),   
    stnp.arrays(dtype=np.float32, shape=(20,20,20), elements=st.floats(0, 1)),
    stnp.arrays(dtype=np.float32, shape=(20,20,20), elements=st.floats(0, 1)),
    stnp.arrays(dtype=np.float32, shape=(20,20,20), elements=st.floats(0, 1))
)
@settings(max_examples=5, deadline=None)
def test_surrounding_filter_valid_return(ccs, wm_arr, gm_arr, csf_arr):
    """
    Given:
        - ccs: labeled image
        - wm_arr, gm_arr, csf_arr: partial volume estimate images
    Then:
        - compute points
    Assert that:
        - points is a pandas Series with index = labels in ccs
            and values are in {-2,0,1,2}
        - summary is a 4 elements tuple
        - all of the elements in summary are integers <= n_components
        - pve_sums is a pandas DataFrame with index = labels in ccs
    """
    wm_img = sitk.GetImageFromArray(wm_arr)
    gm_img = sitk.GetImageFromArray(gm_arr)
    csf_img = sitk.GetImageFromArray(csf_arr)
    
    wm_img.CopyInformation(ccs)
    gm_img.CopyInformation(ccs)
    csf_img.CopyInformation(ccs)
    
    labels_arr = sitk.GetArrayFromImage(ccs)
    n_components = len(np.unique(labels_arr)) - 1
    
    points, summary, pve_sums = surrounding_filter(ccs, n_components, [wm_img, gm_img, csf_img])
    
    assert set(points.index) == set(range(1, n_components+1))
    assert set(points.values).issubset({-2,0,1,2})
    assert isinstance(summary, tuple)
    assert len(summary) == 4
    for n in summary:
        assert isinstance(n, int)
        assert n <= n_components
    assert isinstance(pve_sums, pd.DataFrame)
    assert set(pve_sums.index) == set(range(1, n_components+1))


@given(
    labeled_image_strategy(),   
    stnp.arrays(dtype=np.float32, shape=(20,20,20), elements=st.floats(0, 1)),
    stnp.arrays(dtype=np.float32, shape=(20,20,20), elements=st.floats(0, 1)),
    stnp.arrays(dtype=np.float32, shape=(20,20,20), elements=st.floats(0, 1))
)
@settings(max_examples=5, deadline=None)
def test_surrounding_filter_expected_results(ccs, wm_arr, gm_arr, csf_arr):
    """
    Given:
        - ccs: labeled image
        - wm_arr, gm_arr, csf_arr: partial volume estimate images
    Then:
        - compute points
    Assert that:
        - points are assigned as expected
    """
    wm_img = sitk.GetImageFromArray(wm_arr)
    gm_img = sitk.GetImageFromArray(gm_arr)
    csf_img = sitk.GetImageFromArray(csf_arr)
    
    wm_img.CopyInformation(ccs)
    gm_img.CopyInformation(ccs)
    csf_img.CopyInformation(ccs)
    
    labels_arr = sitk.GetArrayFromImage(ccs)
    n_components = len(np.unique(labels_arr)) - 1
    
    points, _, _ = surrounding_filter(ccs, n_components, [wm_img, gm_img, csf_img])
    
    for label in points.index:
        lesion_mask = labels_arr == label

        lesion_img = sitk.GetImageFromArray(lesion_mask.astype(np.uint8))
        lesion_img.CopyInformation(ccs)
        dilated = sitk.BinaryDilate(lesion_img, [1, 1, 1], sitk.sitkBox)
        surround_mask = sitk.GetArrayFromImage(dilated) & ~lesion_mask

        wm_mean = wm_arr[surround_mask].mean() if surround_mask.any() else 0
        gm_mean = gm_arr[surround_mask].mean() if surround_mask.any() else 0
        csf_mean = csf_arr[surround_mask].mean() if surround_mask.any() else 0
        sums = [wm_mean, gm_mean, csf_mean]

        if sum(sums) == 0:
            assert points[label] == -2
        else:
            max_idx = np.argmax(sums)
            expected = [2, 1, 0][max_idx]  # wm, gm, csf
            assert points[label] == expected


@given(
    non_binary_image_strategy(),
    st.floats(min_value=0, max_value=20),   # mean
    st.floats(min_value=0.1, max_value=10)  # std > 0
)
@settings(max_examples=5, deadline=None)
def test_gaussian_transform_valid_return(img, mean, std):
    """
    Given:
        - a non-binary SimpleITK image (gray levels 10-15)
        - mean and std for the Gaussian weighting
    Then:
        - apply gaussian_transform
    Assert:
        - output is a SimpleITK image
        - output has same size, spacing, origin, direction as input
        - all output pixel values are non-negative
    """

    enhanced = gaussian_transform(img, mean, std)
    
    arr_in = sitk.GetArrayFromImage(img)
    arr_out = sitk.GetArrayFromImage(enhanced)
    
    assert isinstance(enhanced, sitk.Image)
    assert enhanced.GetSize() == img.GetSize()
    assert enhanced.GetSpacing() == img.GetSpacing()
    assert enhanced.GetOrigin() == img.GetOrigin()
    assert enhanced.GetDirection() == img.GetDirection()
    assert np.all(arr_out >= 0)
    assert arr_out.shape == arr_in.shape


@given(non_binary_image_strategy(),
    st.floats(min_value=0, max_value=20),
    st.floats(min_value=0.1, max_value=10))
@settings(max_examples=5, deadline=None)
def test_gaussian_transform_peak_behavior(img, mean, std):
    """
    Given:
        - a non-binary SimpleITK image
        - mean and std for Gaussian weighting
    Then:
        - apply gaussian_transform
    Assert:
        - pixels close to 'mean' are enhanced more than pixels far from 'mean'
    """

    arr_in = sitk.GetArrayFromImage(img)
    enhanced = gaussian_transform(img, mean, std, return_float=True)
    arr_out = sitk.GetArrayFromImage(enhanced)
    
    # Pick one pixel close to mean, one far
    diff = np.abs(arr_in - mean)
    close_idx = np.unravel_index(np.argmin(diff), diff.shape)
    far_idx = np.unravel_index(np.argmax(diff), diff.shape)
    
    assert arr_out[close_idx] >= arr_out[far_idx]


@settings(deadline=None, max_examples=5)
@given(blurred_sphere_strategy())
def test_extend_lesions_valid_return(image):
    """
    Given:
        - a blurred spherical lesion image
    Then:
        - call extend_lesions with a binary lesion mask
    Assert that:
        - the output is a valid binary SimpleITK image
        - the size matches the input
        - only 0 and 1 values are present
    """
    lesion_mask = image > np.percentile(sitk.GetArrayFromImage(image), 80)
    ccs, n_components = connected_components(lesion_mask)
    extended = extend_lesions(ccs, n_components, image)
    arr = sitk.GetArrayFromImage(extended)

    assert isinstance(extended, sitk.Image)
    assert extended.GetSize() == image.GetSize()
    assert extended.GetPixelID() == sitk.sitkUInt8
    assert np.isin(arr, [0, 1]).all()


@settings(deadline=None, max_examples=5)
@given(blurred_sphere_strategy())
def test_extend_lesions_includes_original(image):
    """
    Given:
        - a blurred spherical lesion image
    Then:
        - extend_lesions is applied to the binary lesion mask
    Assert that:
        - all voxels in the original lesion remain labeled in the extended mask
    """
    lesion_mask = image > np.percentile(sitk.GetArrayFromImage(image), 80)
    ccs, n_components = connected_components(lesion_mask)
    extended = extend_lesions(ccs, n_components, image)

    arr_lesion = sitk.GetArrayFromImage(lesion_mask)
    arr_extended = sitk.GetArrayFromImage(extended)

    assert np.all(arr_extended[arr_lesion == 1] == 1)


@settings(deadline=None, max_examples=5)
@given(blurred_sphere_strategy(), st.floats(min_value=0.5, max_value=2.0))
def test_extend_lesions_respects_threshold(image, n_std):
    """
    Given:
        - a blurred spherical lesion image
        - two different n_std values
    Then:
        - increasing n_std should increase the extension
    Assert that:
        - the number of lesion voxels increases with larger n_std
    """
    lesion_mask = image > np.percentile(sitk.GetArrayFromImage(image), 80)
    ccs, n_components = connected_components(lesion_mask)
    small_ext = extend_lesions(ccs, n_components, image, n_std=0.5)
    large_ext = extend_lesions(ccs, n_components, image, n_std=n_std)

    arr_small = sitk.GetArrayFromImage(small_ext)
    arr_large = sitk.GetArrayFromImage(large_ext)

    assert arr_large.sum() >= arr_small.sum()


@given(binary_mask_pair_strategy())
@settings(max_examples=5, deadline=None)
def test_evaluate_region_wise_valid_return(mask_pair):
    """
    Given:
        - two random binary masks
    Then:
        - evaluate them region-wise
    Assert that:
        - output is a dict that contains rw-TPF and rw-FPF as keys
        - all returned values are floats
    """
    mask, gt = mask_pair
    metrics = evaluate_region_wise(mask, gt)

    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {'rw-TPF', 'rw-FPF'}
    assert isinstance(metrics['rw-TPF'], float)
    assert isinstance(metrics['rw-FPF'], float)


@given(binary_mask_pair_strategy())
@settings(max_examples=5, deadline=None)
def test_evaluate_region_wise_output_range(mask_pair):
    """
    Given:
        - two random binary masks
    Then:
        - evaluate them region-wise
    Assert that:
        - all returned values are between 0 and 1
    """
    mask, gt = mask_pair
    metrics = evaluate_region_wise(mask, gt)

    assert 0.0 <= metrics["rw-TPF"] <= 1.0
    assert 0.0 <= metrics["rw-FPF"] <= 1.0




@given(binary_mask_pair_strategy())
@settings(max_examples=5, deadline=None)
def test_evaluate_region_wise_identical_masks(mask_pair):
    """
    Given:
        - mask and gt are identical and binary
    Then:
        - evaluate them region-wise
    Assert that:
        - rw-TPF = 1.0
        - rw-FPF = 0.0
    """
    img, _ = mask_pair 
    
    arr = sitk.GetArrayFromImage(img)
    assume(np.any(arr == 1) and np.any(arr == 0))

    metrics = evaluate_region_wise(img, img)

    assert metrics == {'rw-TPF': 1.0, 'rw-FPF': 0.0}


def test_evaluate_region_wise_all_zero_vs_all_one():
    """
    Given:
        - mask is all zeros
        - gt is all ones
    Then:
        - evaluate them region-wise
    Assert that:
        - rw-TPF = 0.0
        - rw-FPF = 0.0
    """
    shape = (16, 16, 16)
    mask_arr = np.zeros(shape, dtype=np.uint8)
    gt_arr = np.ones(shape, dtype=np.uint8)

    mask = sitk.GetImageFromArray(mask_arr)
    gt = sitk.GetImageFromArray(gt_arr)

    metrics = evaluate_region_wise(mask, gt)

    assert metrics == {'rw-TPF': 0.0, 'rw-FPF': 0.0}



@given(binary_mask_pair_strategy(), non_binary_image_strategy())
@settings(max_examples=5, deadline=None)
def test_evaluate_region_wise_not_binary(mask_pair, not_binary):
    """
    Given:
        - binary gt
        - non-binary mask
    Then:
        - evaluate non-binary mask with gt
        - evaluate the same binarized mask with gt
    Assert that:
        - the same metrics are returned
    """
    _, gt = mask_pair
    binary = not_binary > 0
    
    binary = resample_to_reference(binary, gt)
    not_binary = resample_to_reference(not_binary, gt)

    assert evaluate_region_wise(binary, gt) == evaluate_region_wise(not_binary, gt)



