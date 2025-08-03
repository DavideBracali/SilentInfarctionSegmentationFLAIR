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
import matplotlib
matplotlib.use('Agg')

from SilentInfarctionSegmentationFLAIR.utils import DimensionError
from SilentInfarctionSegmentationFLAIR.utils import check_3d
from SilentInfarctionSegmentationFLAIR.utils import get_info
from SilentInfarctionSegmentationFLAIR.utils import get_array_from_image
from SilentInfarctionSegmentationFLAIR.utils import plot_image
from SilentInfarctionSegmentationFLAIR.utils import orient_image
from SilentInfarctionSegmentationFLAIR.utils import resample_to_reference
from SilentInfarctionSegmentationFLAIR.utils import plot_histogram
from SilentInfarctionSegmentationFLAIR.utils import plot_multiple_histograms
from SilentInfarctionSegmentationFLAIR.utils import histogram_stats



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
    
    assert isinstance(info, dict)
    assert len(info) == 4
    
    for k, i in info.items():
        assert isinstance(i, tuple)
    
    assert len(info["size"]) == 3
    assert len(info["spacing"]) == 3
    assert len(info["origin"]) == 3
    assert len(info["direction"]) == 9
    
    
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
    
    assert isinstance(plot_info, dict)
    assert len(plot_info) == 3
    
    for k, i in plot_info.items():
        assert isinstance(i, tuple)
    
    assert len(plot_info["size"]) == 3
    assert len(plot_info["spacing"]) == 3
    assert len(plot_info["aspects"]) == 3
    
    
    
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
        assert len(set(plot_info["aspects"])) == 1
    else:
        assert len(set(plot_info["aspects"])) != 1
          
        
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
           
        
@given(gauss_noise_strategy_3D(), gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_resample_to_reference_valid_return(image, reference):
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


@given(gauss_noise_strategy_3D(), st.integers(1,256))
@settings(max_examples=5, deadline=None)
def test_plot_histogram_valid_return(image, bins):
    """
    Given:
        - gaussian noise SimpleITK image
        - a random number of bins
    Then:
        - plot histogram with the specified number of bins
    Assert that:
        - a 2-dimensional tuple is returned
        - the retured counts is an array with length equal to the number of bins
        - the returned bin edges is an array with lenght equal to the number of bins + 1
        - the sum of the returned counts is equal to the number of voxels
    """
    hist = plot_histogram(image, bins)
    counts, edges = hist
    n_voxels = len(get_array_from_image(image).flatten())

    assert isinstance(hist, tuple)
    assert len(hist) == 2
    assert isinstance(counts, np.ndarray)
    assert len(counts) == bins
    assert isinstance(edges, np.ndarray)
    assert len(edges) == bins + 1
    assert np.sum(counts) == n_voxels
    


@given(gauss_noise_strategy_3D(), st.integers(1,256))
@settings(max_examples=5, deadline=None)
def test_plot_histogram_no_bkg(image, bins):
    """
    Given:
        - gaussian noise SimpleITK image
        - a random number of bins
    Then:
        - plot histogram excluding gray level 0 
    Assert that:
        - the returned histogram has 0 counts for gray level 0
    """

    counts_no_bkg, edges_no_bkg = plot_histogram(image, bins=bins, no_bkg=True)
    counts, edges = plot_histogram(image, bins=bins, no_bkg=False)

    assert 0 not in edges_no_bkg
    if 0 in np.unique(get_array_from_image(image).flatten()):
        assert 0 in edges

    


@given(st.lists(gauss_noise_strategy_3D(), min_size=1, max_size=5))
@settings(max_examples=5, deadline=None)
def test_plot_multiple_histograms_valid_return(images):
    """
    Given:
        - a list of gaussian noise SimpleITK images
        - a list of random number of bins
    Then:
        - plot all histograms with the corresponding number of bins
    Assert that:
        - a list of 2-dimensional tuples is returned
        - (for all histograms) the retured counts is an array with length equal to the number of bins
        - (for all histograms) the returned bin edges is an array with lenght equal to the number of bins + 1
        - (for all histograms) the sum of the returned counts is equal to the number of voxels
    """
    bins = [np.random.randint(1, 100) for _ in images]
    hists = plot_multiple_histograms(images, bins)
    n_voxels = [len(get_array_from_image(image).flatten()) for image in images]

    assert isinstance(hists, list)
    for i, hist in enumerate(hists):
        counts, edges = hist
        assert isinstance(hist, tuple)
        assert len(hist) == 2
        assert isinstance(counts, np.ndarray)
        assert len(counts) == bins[i]
        assert isinstance(edges, np.ndarray)
        assert len(edges) == bins[i] + 1
        assert np.sum(counts) == n_voxels[i]


@given(st.lists(gauss_noise_strategy_3D(), min_size=1, max_size=5))
@settings(max_examples=5, deadline=None)
def test_plot_multiple_histograms_no_bkg(images):
    """
    Given:
        - a list of gaussian noise SimpleITK images
        - a list of random number of bins
    Then:
        - plot all histograms with the corresponding number of bins
    Assert that:
        - each of the returned histograms has 0 counts for gray level 0
    """
    
    bins = [np.random.randint(1, 100) for _ in images]
    for image in images:
        counts, edges = plot_histogram(image, bins=bins, no_bkg=True)
        assert 0 not in edges


@given(gauss_noise_strategy_3D(), st.integers(10, 256))
@settings(max_examples=5, deadline=None)
def test_histogram_stats_valid_return(image, bins):
    """
    Given:
        - gaussian noise SimpleITK image
        - a random number of histogram bins
    Then:
        - compute histogram and pass it to histogram_stats
    Assert that:
        - returned value is a 6-tuple of floats
        - mean is between min and max of the histogram bin centers
        - percentiles are ordered: q1 <= q2
        - variance is non-negative
        - skewness and kurtosis are finite numbers
    """
    hist = plot_histogram(image, bins)
    stats = histogram_stats(hist)

    counts, edges = hist
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    assert isinstance(stats, tuple)               
    assert len(stats) == 6                 
    for val in stats:
        assert isinstance(val, np.floating)             

    mean, p1, p2, var, skew, kurt = stats
    assert edges[0] <= mean <= edges[-1]          
    assert p1 <= p2                               
    assert var >= 0                               
    assert np.isfinite(skew)                      
    assert np.isfinite(kurt)                      