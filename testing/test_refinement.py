#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-13 15:33:12

@author: david
"""

import pytest
import hypothesis.strategies as st
from hypothesis import given, settings

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
def test_diameter_filter_expected_results(image):
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
    _,_,removed1 = diameter_filter(image, lower_thr=1.5)
    _,_,removed2 = diameter_filter(image, upper_thr=2.5)
    _,_,removed3 = diameter_filter(image, lower_thr=1.5, upper_thr=2.5)
    _,_,removed4 = diameter_filter(image, lower_thr=1)
    _,_,removed5 = diameter_filter(image, upper_thr=3)
    _,_,removed6 = diameter_filter(image, lower_thr=1, upper_thr=3)
    _,_,removed7 = diameter_filter(image, lower_thr=3)
    _,_,removed8 = diameter_filter(image, upper_thr=1)
    _,_,removed9 = diameter_filter(image, lower_thr=3, upper_thr=1)

    assert removed1 == removed2 == removed3 == {2 : (1,3)}
    assert removed4 == removed5 == removed6 == {}
    assert removed7 == removed8 == removed9 == {1 : (2,2), 2 : (1,3)}