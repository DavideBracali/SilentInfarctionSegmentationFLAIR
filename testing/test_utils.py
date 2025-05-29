#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:17:21 2025

@author: david
"""

import pytest
import hypothesis.strategies as st
from hypothesis import given, settings

import sys
import os
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                            '..', 'SilentInfarctionSegmentationFLAIR')))
import numpy as np
import SimpleITK as sitk

from SilentInfarctionSegmentationFLAIR.utils import DimensionError
from SilentInfarctionSegmentationFLAIR.utils import check_3d
from SilentInfarctionSegmentationFLAIR.utils import get_info
from SilentInfarctionSegmentationFLAIR.utils import get_array_from_image
from SilentInfarctionSegmentationFLAIR.utils import plot_image
from SilentInfarctionSegmentationFLAIR.utils import orient_image
from SilentInfarctionSegmentationFLAIR.utils import resample_to_reference
from SilentInfarctionSegmentationFLAIR.utils import plot_histogram



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
    Creates a 3D image with gaussian noise for testing
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
def gauss_noise_strategy_2D(draw):    
    """
    Creates a 2D image with gaussian noise for testing
    """
    origin = draw(st.tuples(*[st.floats(0., 100.)] * 2))
    spacing = draw(st.tuples(*[st.floats(.1, 1.)] * 2))
    direction = tuple([0., 1., 1., 0.])
    size = (draw(st.integers(10, 100)), draw(st.integers(10, 100)))
    
    #draw image    
    filter_ = sitk.GaussianImageSource()
    filter_.SetSize(size)
    filter_.SetOrigin(origin)
    filter_.SetSpacing(spacing)
    filter_.SetDirection(direction)
    image = filter_.Execute()

    return image


##################
###  TESTING   ###
##################


@given(gauss_noise_strategy_2D())
@settings(max_examples=5, deadline=None)
def test_check_3d_raise_dimensionerror(image2d):
    """
    Given:
        - gaussian noise SimpleITK 2D image
    Then:
        - apply check_3d 
    Assert that:
        - image2d raises DimensionError
        """
    with pytest.raises(DimensionError):
        check_3d(image2d)
        
        
@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_check_3d_doesnt_raise_exception(image3d):
    """
    Given:
        - gaussian noise SimpleITK 3D image
    Then:
        - apply check_3d 
    Assert that:
        - image3d doesn't raise any exception
        """
    try:
        check_3d(image3d)
    except Exception:
        assert False        
        

@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_get_info_valid_return(image):
    """
    Given:
        - gaussian noise SimpleITK 3D image
    Then:
        - apply get_info 
    Assert that:
        - a 4-items non-empty dict is returned
        - returned size, spacing, origin are 3-elements tuple each
        - returned direction matrix is a 9-element tuple 
    """
    info = get_info(image)
    
    assert isinstance(info, dict),       "return type is not dict"
    assert len(info) == 4,               "returned dict doesn't have 4 items"
    
    for k, i in info.items():
        assert isinstance(i, tuple),     f"{k} is not a tuple"
    
    assert len(info["size"]) == 3,       "size is a tuple with length different than 3"
    assert len(info["spacing"]) == 3,    "spacing is a tuple with length different than 3"
    assert len(info["origin"]) == 3,     "origin is a tuple with length different than 3"
    assert len(info["direction"]) == 9,  "direction is a tuple with length different than 9"
    
    
@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_get_array_from_image_not_uniform(image):
    """
    Given:
        - gaussian noise SimpleITK 3D image
    Then:
        - apply get_array_from_image 
    Assert that:
        - array is not uniform (gaussian noise)
    """
    
    array = get_array_from_image(image)
    
    assert not np.all(array == array[0,0,0])
    
    
@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_get_array_from_image_same_size(image):
    """
    Given:
        - gaussian noise SimpleITK 3D image
    Then:
        - apply get_array_from_image 
    Assert that:
        - array size is consistent with image size
    """
    
    array = get_array_from_image(image)
    
    assert array.shape == image.GetSize()


@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_plot_image_valid_return(image):
    """
    Given:
        - gaussian noise SimpleITK 3D image
    Then:
        - plot the image with plot_image
    Assert that:
        - a 3-items dict is returned
        - returned size, spacing, aspects are 3-elements tuple each
    """
    plot_info = plot_image(image)
    
    assert isinstance(plot_info, dict),              "return type is not dict"
    assert len(plot_info) == 3,                      "returned dict doesn't have 3 items"
    
    for k, i in plot_info.items():
        assert isinstance(i, tuple),                 f"{k} is not a tuple"
    
    assert len(plot_info["size"]) == 3,              "size is a tuple with length different than 3"
    assert len(plot_info["spacing"]) == 3,           "spacing is a tuple with length different than 3"
    assert len(plot_info["aspects"]) == 3,           "origin is a tuple with length different than 3"
    
    
    
@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_plot_image_spacings_aspects(image):
    """
    Given:
        - gaussian noise SimpleITK 3D image
    Then:
        - plot the image with plot_image
    Assert that:
        - if all spacings are the same then all aspects are the same
        - if all spacings are not the same then all aspects are not the same
    """
    plot_info = plot_image(image)
        
    if len(set(plot_info["spacing"])) == 1:
        assert len(set(plot_info["aspects"])) == 1,  "spacings are equal but aspects are different"
    else:
        assert len(set(plot_info["aspects"])) != 1,  "spacings are different but aspects are equal"
          
        
@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_plot_image_raise_indexerror(image):
    """
    Given:
        - gaussian noise SimpleITK 3D image
    Then:
        - apply plot_image with out of bounds coordinates
    Assert that:
        - IndexError is raised
    """
    oob = [(image.GetSize()[0]+1, 0, 0),
           (0, image.GetSize()[1]+1, 0),
           (0, 0, image.GetSize()[2]+1),
           (-image.GetSize()[0]-1, 0, 0),
           (0, -image.GetSize()[1]-1, 0),
           (0, 0, -image.GetSize()[2]-1)]
    
    for i in range(3):
        with pytest.raises(IndexError):
            plot_image(image, xyz=oob[i])
           
            
@given(gauss_noise_strategy_3D(), random_orientation_strategy())
@settings(max_examples=5, deadline=None)
def test_orient_image_valid_return(image, new_orientation):
    """
    Given:
        - gaussian noise SimpleITK 3D image
        - a random orientation string
    Then:
        - orient the image with orient_image
    Assert that:
        - a SimpleITK.Image is returned
        - if the new orientation is equal to the old then nothing changes
        - if the new orientation is not equal to the old one then a different
        image is returned
    """
    old_orientation = sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(image.GetDirection())
    oriented_image_old = orient_image(image, old_orientation)
    oriented_image_new = orient_image(image, new_orientation)

    assert isinstance(oriented_image_old, sitk.Image)  
    assert get_info(oriented_image_old) == get_info(image)
    if new_orientation != old_orientation:
        assert get_info(oriented_image_new) != get_info(image)
           
        
@given(gauss_noise_strategy_3D(), gauss_noise_strategy_3D(), st.integers(0,255))
@settings(max_examples=5, deadline=None)
def test_resample_to_reference_valid_return(image, reference, default_value):
    """
    Given:
        - 2 gaussian noise SimpleITK images
    Then:
        - resample image to reference
    Assert that:
        - a SimpleITK.Image is returned
        - size, spacing, origin, direction of the resampled image are the same
        as the reference
    """
    image_rs = resample_to_reference(image, reference)
    
    assert isinstance(image_rs, sitk.Image)
    assert get_info(image_rs) == get_info(reference)
        
        
@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_plot_histogram_valid_return(image):
    """
    Given:
        - a gaussian noise SimpleITK 3D image
    Then:
        - apply plot_histogram
    Assert that:
        - a flattened numpy array is returned
        - if no_bkg = True the returned array doesn't contain 0s
    """
    flattened = plot_histogram(image, no_bkg=True)
    
    assert len(flattened.shape) == 1
    assert np.all(flattened != 0) == True
    
    