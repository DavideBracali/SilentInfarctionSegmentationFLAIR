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

matplotlib.use('Agg')
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                            '..', 'SilentInfarctionSegmentationFLAIR')))

from SilentInfarctionSegmentationFLAIR.utils import get_array_from_image
from SilentInfarctionSegmentationFLAIR.refinement import connected_components
from SilentInfarctionSegmentationFLAIR.refinement import find_diameters
from SilentInfarctionSegmentationFLAIR.refinement import diameter_filter
from SilentInfarctionSegmentationFLAIR.refinement import label_filter
from SilentInfarctionSegmentationFLAIR.refinement import evaluate_region_wise





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
        - returned removed is a dictionary
        - the cube is never removed
    """

    filtered_ccs, filtered_n, removed = diameter_filter(ccs, lower_thr, upper_thr)
    labels = set(np.unique(get_array_from_image(ccs)))
    n = len(labels) - 1         # no background
    filtered_labels = set(np.unique(get_array_from_image(filtered_ccs)))


    assert isinstance(filtered_ccs, sitk.Image)
    assert filtered_labels.issubset(labels)
    assert filtered_n <= n
    assert isinstance(removed, dict)
    assert 1 not in removed.keys()


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
    _,_,removed1 = diameter_filter(ccs, lower_thr=1.5)
    _,_,removed2 = diameter_filter(ccs, upper_thr=2.5)
    _,_,removed3 = diameter_filter(ccs, lower_thr=1.5, upper_thr=2.5)
    _,_,removed4 = diameter_filter(ccs, lower_thr=1)
    _,_,removed5 = diameter_filter(ccs, upper_thr=3)
    _,_,removed6 = diameter_filter(ccs, lower_thr=1, upper_thr=3)
    _,_,removed7 = diameter_filter(ccs, lower_thr=3)
    _,_,removed8 = diameter_filter(ccs, upper_thr=1)
    _,_,removed9 = diameter_filter(ccs, lower_thr=3, upper_thr=1)

    assert removed1 == removed2 == removed3 == {2 : (1,3)}
    assert removed4 == removed5 == removed6 == {}
    assert removed7 == removed8 == removed9 == {1 : (2,2), 2 : (1,3)}



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
        - removed is a dict with int keys and values
    """
    segm_filtered, removed = label_filter(segm, labels_to_remove=labels_to_remove)

    assert isinstance(segm_filtered, sitk.Image)
    assert isinstance(removed, dict)
    assert all(isinstance(k, int) for k in removed.keys())
    assert all(isinstance(v, int) and v >= 0 for v in removed.values())


@given(labeled_image_strategy(),
    st.lists(st.integers(min_value=1, max_value=2), unique=True))
@settings(max_examples=5, deadline=None)
def test_labels_filter_properties(segm, labels_to_remove):
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
        assert np.all(arr_filtered[arr == label] == 0)
    for label in labels_to_keep:
        assert np.array_equal(arr_filtered[arr == label], arr[arr == label])
    for label, count in removed.items():
        assert count == np.sum(arr == label)
    assert np.array_equal(get_array_from_image(segm_filtered), get_array_from_image(segm_filtered2))
    assert removed2 == {}



@given(binary_mask_pair_strategy())
@settings(max_examples=5, deadline=None)
def test_evaluate_region_wise_valid_return(mask_pair):
    """
    Given:
        - two random binary masks
    Then:
        - evaluate them region-wise
    Assert that:
        - output is a dict that contains TPF, FPF and DSC as keys
        - all returned values are floats
    """
    mask, gt = mask_pair
    metrics = evaluate_region_wise(mask, gt)

    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {'TPF', 'FPF', 'DSC'}
    for _, v in metrics.items():
        assert isinstance(v, float)



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

    for _, v in metrics.items():
        assert 0.0 <= v <= 1.0




@given(binary_mask_pair_strategy())
@settings(max_examples=5, deadline=None)
def test_evaluate_region_wise_identical_masks(mask_pair):
    """
    Given:
        - mask and gt are identical and binary
    Then:
        - evaluate them region-wise
    Assert that:
        - TPF = 1.0
        - FPF = 0.0
        - DSC = 1.0
    """
    img, _ = mask_pair 
    
    arr = sitk.GetArrayFromImage(img)
    assume(np.any(arr == 1) and np.any(arr == 0))

    metrics = evaluate_region_wise(img, img)

    assert metrics == {'TPF': 1.0, 'FPF': 0.0, 'DSC': 1.0}


def test_evaluate_region_wise_all_zero_vs_all_one():
    """
    Given:
        - mask is all zeros
        - gt is all ones
    Then:
        - evaluate them region-wise
    Assert that:
        - TPF = 0.0
        - FPF = 0.0
        - DSC = 0.0
    """
    shape = (16, 16, 16)
    mask_arr = np.zeros(shape, dtype=np.uint8)
    gt_arr = np.ones(shape, dtype=np.uint8)

    mask = sitk.GetImageFromArray(mask_arr)
    gt = sitk.GetImageFromArray(gt_arr)

    metrics = evaluate_region_wise(mask, gt)

    assert metrics == {'TPF': 0.0, 'FPF': 0.0, 'DSC': 0.0}



@given(binary_mask_pair_strategy(), non_binary_image_strategy())
@settings(max_examples=5, deadline=None)
def test_evaluate_region_wise_raises_value_error_if_not_binary(mask_pair, not_binary):
    """
    Given:
        - binary mask and gt
        - non-binary mask
    Then:
        - call evaluate(mask, gt)
    Assert that:
        - a ValueError is raised when mask is not binary
        - no exceptions are raised when both mask and gt are binary
    """
    mask, gt = mask_pair
    
    with pytest.raises(ValueError):
        evaluate_region_wise(not_binary, gt)
        evaluate_region_wise(mask, not_binary)

    try:
        evaluate_region_wise(mask, gt)
    except Exception:
        assert False


