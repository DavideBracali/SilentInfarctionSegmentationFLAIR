#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:01:21 2025

@author: david
"""

import pytest
import hypothesis.strategies as st
import hypothesis.extra.numpy as stnp
from hypothesis import given, settings

import sys
import os
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                            '..', 'SilentInfarctionSegmentationFLAIR')))

import numpy as np
import SimpleITK as sitk

from SilentInfarctionSegmentationFLAIR.segmentation import get_mask_from_segmentation



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



##################
###  TESTING   ###
##################


#@given(random_segmentation_strategy())
#@settings(max_examples=5, deadline=None)
#def test_get_mask_from_segmentation_same_info(segm):
    """
    Given:
        - random segmentation
    Then:
        - get mask
    Assert that:
        - returned mask has the same size, spacing, origin, direction
        """
#    mask = get_mask_from_segmentation(segm)
    
#    assert mask.GetSize() == segm.GetSize()
#    assert mask.GetSpacing() == segm.GetSpacing()
#    assert mask.GetOrigin() == segm.GetOrigin()
#    assert mask.GetDirection() == segm.GetDirection()
    
#@given(random_segmentation_strategy())
#@settings(max_examples=5, deadline=None)
#def test_get_mask_from_segmentation_is_binary(segm):
    """
    Given:
        - random segmentation
    Then:
        - get mask
    Assert that:
        - returned mask is binary
        """
#    mask = get_mask_from_segmentation(segm)    
#    mask_array = sitk.GetArrayFromImage(mask)
    
#    assert set(mask_array.flatten()) == {0,1}


