#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:01:21 2025

@author: david
"""

import pytest
import hypothesis.strategies as st
import hypothesis.extra.numpy as stnp
from hypothesis import given, settings, assume

import sys
import os
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                            '..', 'SilentInfarctionSegmentationFLAIR')))

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')

from SilentInfarctionSegmentationFLAIR.utils import resample_to_reference
from SilentInfarctionSegmentationFLAIR.segmentation import get_mask_from_segmentation
from SilentInfarctionSegmentationFLAIR.segmentation import get_mask_from_pve
from SilentInfarctionSegmentationFLAIR.segmentation import apply_threshold
from SilentInfarctionSegmentationFLAIR.segmentation import evaluate_voxel_wise




####################
###  STRATEGIES  ###
####################


@st.composite
def random_orientation_strategy(draw):
    """
    Generate a random orientation string like 'RAS', 'LPI', ...
    """
    axis_directions = {
        'X': ['L', 'R'],
        'Y': ['P', 'A'],
        'Z': ['I', 'S']
    }

    axis_order = draw(st.permutations(['X', 'Y', 'Z']))
    orientation = ''.join(draw(st.sampled_from(axis_directions[ax])) for ax in axis_order)
    
    return orientation


@st.composite
def gauss_noise_strategy_3D(draw):    
    """
    Creates a 3D image with gaussian noise for testing.
    """
    origin = draw(st.tuples(*[st.floats(0., 100.)] * 3))
    spacing = draw(st.tuples(*[st.floats(.1, 1.)] * 3))
    direction = tuple([0., 0., 1., 1., 0., 0., 0., 1., 0.])
    size = (draw(st.integers(10, 100)), draw(st.integers(10, 100)), draw(st.integers(10, 100)))
    
    #draw image    
    filter_ = sitk.GaussianImageSource()
    filter_.SetSize(size)
    filter_.SetOrigin(origin)
    filter_.SetSpacing(spacing)
    filter_.SetDirection(direction)
    image = filter_.Execute()

    # random orientation
    orientation = draw(random_orientation_strategy())
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(orientation)
    image = orienter.Execute(image)

    return image


@st.composite
def random_segmentation_strategy(draw):    
    """
    Creates random 3D segmentation (image filled with integers) for testing.
    """
    origin = draw(st.tuples(*[st.floats(0., 100.)] * 3))
    spacing = draw(st.tuples(*[st.floats(.1, 1.)] * 3))
    direction = tuple([0., 0., 1., 1., 0., 0., 0., 1., 0.])
    size = (draw(st.integers(10, 100)), draw(st.integers(10, 100)),
            draw(st.integers(10, 100)))
    image_array=draw(stnp.arrays(dtype=np.int32, shape=size,
                     elements=st.integers(1,100)))
    
    #draw image    
    segm = sitk.GetImageFromArray(image_array)
    segm.SetOrigin(origin)
    segm.SetSpacing(spacing)
    segm.SetDirection(direction)

    # random orientation
    orientation = draw(random_orientation_strategy())
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(orientation)
    segm = orienter.Execute(segm)

    return segm


@st.composite
def random_pve_strategy(draw):    
    """
    Creates random 3D pve (image filled with floats in [0,1]) for testing.
    """
    origin = draw(st.tuples(*[st.floats(0., 100.)] * 3))
    spacing = draw(st.tuples(*[st.floats(.1, 1.)] * 3))
    direction = tuple([0., 0., 1., 1., 0., 0., 0., 1., 0.])
    size = (draw(st.integers(10, 100)), draw(st.integers(10, 100)),
            draw(st.integers(10, 100)))
    image_array=draw(stnp.arrays(dtype=np.float32, shape=size,
                     elements=st.floats(0,1)))
    
    #draw image    
    segm = sitk.GetImageFromArray(image_array)
    segm.SetOrigin(origin)
    segm.SetSpacing(spacing)
    segm.SetDirection(direction)

    # random orientation
    orientation = draw(random_orientation_strategy())
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(orientation)
    segm = orienter.Execute(segm)

    return segm


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


@given(random_segmentation_strategy(), st.lists(st.integers(1,100),
                                                min_size=1, max_size=5))
@settings(max_examples=5, deadline=None)
def test_get_mask_from_segmentation_valid_return(segm, labels):
    """
    Given:
        - random segmentation
        - random labels
    Then:
        - get mask
    Assert that:
        - returned mask has the same size, spacing, origin, direction
        """
    mask = get_mask_from_segmentation(segm, labels)
    
    assert mask.GetSize() == segm.GetSize()
    assert mask.GetSpacing() == segm.GetSpacing()
    assert mask.GetOrigin() == segm.GetOrigin()
    assert mask.GetDirection() == segm.GetDirection()
    

@given(random_segmentation_strategy(), st.lists(st.integers(1,100),
                                                min_size=1, max_size=5))
@settings(max_examples=5, deadline=None)
def test_get_mask_from_segmentation_is_binary(segm, labels):
    """
    Given:
        - random segmentation
        - random labels
    Then:
        - get mask
    Assert that:
        - returned mask is binary
        """
    mask = get_mask_from_segmentation(segm, labels)    
    mask_array = sitk.GetArrayFromImage(mask)
    
    assert set(mask_array.flatten()) <= {0,1}


@given(random_pve_strategy(), st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=5, deadline=None)
def test_get_mask_from_pve_valid_return(pve, thr):
    """
    Given:
        - random pve image
        - random threshold in [0,1]
    Then:
        - get mask
    Assert that:
        - returned mask has the same size, spacing, origin, direction
    """
    mask = get_mask_from_pve(pve, thr)

    assert mask.GetSize() == pve.GetSize()
    assert mask.GetSpacing() == pve.GetSpacing()
    assert mask.GetOrigin() == pve.GetOrigin()
    assert mask.GetDirection() == pve.GetDirection()


@given(random_pve_strategy(), st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=5, deadline=None)
def test_get_mask_from_pve_is_binary(pve, thr):
    """
    Given:
        - random pve image
        - random threshold in [0,1]
    Then:
        - get mask
    Assert that:
        - returned mask is binary
    """
    mask = get_mask_from_pve(pve, thr)
    mask_array = sitk.GetArrayFromImage(mask)
    
    assert set(mask_array.flatten()) <= {0,1}


@given(random_pve_strategy())
@settings(max_examples=5, deadline=None)
def test_get_mask_from_pve_thr_zero(pve):
    """
    Given:
        - random pve image
        - threshold = 0
    Then:
        - get mask
    Assert that:
        - all voxels >= 0 are set to 1
    """
    mask = get_mask_from_pve(pve, thr=0.0)
    mask_array = sitk.GetArrayFromImage(mask)
    
    assert set(mask_array.flatten()) == {1}
    


@given(gauss_noise_strategy_3D(), st.floats(0.0, 500.0))
@settings(max_examples=5, deadline=None)
def test_apply_threshold_valid_return(image, thr):
    """
    Given:
        - gaussian noise SimpleITK image
        - a random threshold
    Then:
        - apply threshold
    Assert that:
        - returned mask is binary
        - all voxels >= threshold are 1
        - all voxels < threshold are 0
        - output image has same properties as input
    """
    
    arr = sitk.GetArrayFromImage(image)

    mask = apply_threshold(image, thr, show=False)    
    mask_arr = sitk.GetArrayFromImage(mask)
    

    assert set(mask_arr.flatten()) <= {0,1}
    assert np.all(mask_arr[arr >= thr] == 1)
    assert np.all(mask_arr[arr < thr] == 0)
    assert mask.GetSize() == image.GetSize()
    assert mask.GetSpacing() == image.GetSpacing()
    assert mask.GetOrigin() == image.GetOrigin()


@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_threshold_lower_than_min(image):
    """
    Given:
        - gaussian noise SimpleITK image
    Then:
        - apply a threshold lower than the minimum gray level
    Assert that:
        - returned mask contains only ones

    """   
    arr = sitk.GetArrayFromImage(image)
    min_gl = np.min(arr)

    mask = apply_threshold(image, thr=float(min_gl-1e-6), show=False)
    
    mask_arr = sitk.GetArrayFromImage(mask)

    assert np.all(mask_arr == 1)



@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_threshold_higher_than_max(image):
    """
    Given:
        - gaussian noise SimpleITK image
    Then:
        - apply a threshold higher than the maximum gray level
    Assert that:
        - returned mask contains only zeros

    """   
    arr = sitk.GetArrayFromImage(image)
    max_gl = np.max(arr)

    mask = apply_threshold(image, thr=float(max_gl+1e-3), show=False)
    
    mask_arr = sitk.GetArrayFromImage(mask)


    assert np.all(mask_arr == 0)



@given(binary_mask_pair_strategy())
@settings(max_examples=5, deadline=None)
def test_evaluate_voxel_wise_valid_return(mask_pair):
    """
    Given:
        - two random binary masks
    Then:
        - evaluate them voxel-wise
    Assert that:
        - output is a dict that contains vw-TPF, vw-FPF and vw-DSC as keys
        - all returned values are floats
    """
    mask, gt = mask_pair
    metrics = evaluate_voxel_wise(mask, gt)

    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {'vw-TPF', 'vw-FPF',
                                   'vw-DSC', 'vw-MCC'}
    for _, v in metrics.items():
        assert isinstance(v, float)



@given(binary_mask_pair_strategy())
@settings(max_examples=5, deadline=None)
def test_evaluate_voxel_wise_output_range(mask_pair):
    """
    Given:
        - two random binary masks
    Then:
        - evaluate them voxel-wise
    Assert that:
        - MCC ranges between -1 and 1
        - TPF, FPF and DSC are between 0 and 1
    """
    mask, gt = mask_pair
    metrics = evaluate_voxel_wise(mask, gt)

    for k, v in metrics.items():
        if k == "vw-MCC":
            assert -1.0 <= v <= 1.0
        else:
            assert 0.0 <= v <= 1.0




@given(binary_mask_pair_strategy())
@settings(max_examples=5, deadline=None)
def test_evaluate_voxel_wise_identical_masks(mask_pair):
    """
    Given:
        - mask and gt are identical and binary
    Then:
        - evaluate them voxel-wise
    Assert that:
        - vw-TPF = 1.0
        - vw-FPF = 0.0
        - vw-DSC = 1.0
        - vw-MCC = 1.0
    """
    img, _ = mask_pair 
    
    arr = sitk.GetArrayFromImage(img)
    assume(np.any(arr == 1) and np.any(arr == 0))

    metrics = evaluate_voxel_wise(img, img)

    assert metrics == {'vw-TPF': 1.0, 'vw-FPF': 0.0,
                       'vw-DSC': 1.0, 'vw-MCC': 1.0}


def test_evaluate_voxel_wise_all_zero_vs_all_one():
    """
    Given:
        - mask is all zeros
        - gt is all ones
    Then:
        - evaluate them voxel-wise
    Assert that:
        - vw-TPF = 0.0
        - vw-FPF = 0.0
        - vw-DSC = 0.0
        - vw-MCC = 0.0
    """
    shape = (16, 16, 16)
    mask_arr = np.zeros(shape, dtype=np.uint8)
    gt_arr = np.ones(shape, dtype=np.uint8)

    mask = sitk.GetImageFromArray(mask_arr)
    gt = sitk.GetImageFromArray(gt_arr)

    metrics = evaluate_voxel_wise(mask, gt)

    assert metrics == {'vw-TPF': 0.0, 'vw-FPF': 0.0,
                       'vw-DSC': 0.0, 'vw-MCC': 0.0}



@given(binary_mask_pair_strategy(), non_binary_image_strategy())
@settings(max_examples=5, deadline=None)
def test_evaluate_voxel_wise_raises_value_error_if_not_binary(mask_pair, not_binary):
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

    assert evaluate_voxel_wise(binary, gt) == evaluate_voxel_wise(not_binary, gt)

