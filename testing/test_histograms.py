#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-11 17:22:59

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

from SilentInfarctionSegmentationFLAIR.histograms import plot_histogram
from SilentInfarctionSegmentationFLAIR.histograms import plot_multiple_histograms
from SilentInfarctionSegmentationFLAIR.histograms import histogram_stats 
from SilentInfarctionSegmentationFLAIR.histograms import gaussian_smooth_histogram
from SilentInfarctionSegmentationFLAIR.histograms import mode_and_rhwhm



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


##################
###  TESTING   ###
##################


@given(gauss_noise_strategy_3D(), st.integers(10,256))
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
    hist = plot_histogram(image, bins, show=False)
    counts, edges = hist
    n_voxels = len(sitk.GetArrayFromImage(image).flatten())

    assert isinstance(hist, tuple)
    assert len(hist) == 2
    assert isinstance(counts, np.ndarray)
    assert len(counts) == bins
    assert isinstance(edges, np.ndarray)
    assert len(edges) == bins + 1
    assert np.sum(counts) == n_voxels    


@given(gauss_noise_strategy_3D(), st.integers(10,256))
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
    counts, edges = plot_histogram(image, bins=bins, no_bkg=False, show=False)

    assert 0 not in edges_no_bkg
    if 0 in np.unique(sitk.GetArrayFromImage(image)):
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
    bins = [np.random.randint(10, 256) for _ in images]
    hists = plot_multiple_histograms(images, bins, show=False)
    n_voxels = [len(sitk.GetArrayFromImage(image).flatten()) for image in images]

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
    
    bins = [np.random.randint(10, 100) for _ in images]
    hists = plot_multiple_histograms(images, bins=bins, no_bkg=True, show=False)
    for i, hist in enumerate(hists):
        _, edges = hist
        if 0 in np.unique(sitk.GetArrayFromImage(images[i])):
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


@given(gauss_noise_strategy_3D(), st.integers(10, 256))
@settings(max_examples=5, deadline=None)
def test_gaussian_smooth_histogram_valid_return(image, bins):
    """
    Given:
        - gaussian noise SimpleITK image
        - a random number of histogram bins

    Then:
        - compute histogram
        - smooth histogram
    Assert:
        - returned output is a 2-element tuple of arrays
        - counts and smooth counts have the same number of elements
        - all smooth counts are positive or null
        - binning is not affected by histogram smoothing
    """
    hist = plot_histogram(image, bins, show=False)
    counts, bins = hist
    smooth_counts, smooth_bins = gaussian_smooth_histogram(hist, sigma=3, show=False)

    assert isinstance(smooth_counts, np.ndarray)
    assert isinstance(smooth_bins, np.ndarray)
    assert smooth_counts.shape == (len(counts),)
    assert np.all(smooth_counts >= 0)
    assert np.allclose(smooth_bins, bins)


@given(gauss_noise_strategy_3D(), st.integers(10, 256))
@settings(max_examples=5, deadline=None)
def test_gaussian_smooth_histogram_min_max(image, bins):
    """
    Given:
        - gaussian noise SimpleITK image
        - a random number of histogram bins

    Then:
        - compute histogram
        - smooth histogram
    Assert:
        - minimum of smoothed histogram is >= than the original minimum
        - maximum of smoothed histogram is <= than the original maximum
    """
    hist = plot_histogram(image, bins, show=False)
    counts, _ = hist
    smooth_counts, _ = gaussian_smooth_histogram(hist, sigma=3, show=False)

    assert smooth_counts.min() >= min(counts)
    assert smooth_counts.max() <= max(counts)


@given(gauss_noise_strategy_3D(), st.integers(10, 256))
@settings(max_examples=5, deadline=None)
def test_mode_and_rhwhm_valid_return(image, bins):
    """
    Given:
        - gaussian noise SimpleITK image
        - a random number of histogram bins

    Then:
        - compute histogram
        - find mode and rhwhm
    Assert:
        - returned mode and rhwhm are floats
        - returned mode is in the bins range
        - returned mode + rhwhm is in the bins range
        - rhwhm is positive

    """

    hist = plot_histogram(image, bins, show=False)
    _, edges = hist
    mode, rhwhm = mode_and_rhwhm(hist, show=False)

    assert isinstance(mode, float) or isinstance(mode, np.floating)
    assert isinstance(rhwhm, float) or isinstance(rhwhm, np.floating)
    assert edges[0] <= mode <= edges[-1]
    assert edges[0] <= mode + rhwhm <= edges[-1]
    assert rhwhm > 0