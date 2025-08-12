#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-11 17:18:09

@author: david
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter1d

from SilentInfarctionSegmentationFLAIR.utils import get_array_from_image
from SilentInfarctionSegmentationFLAIR.utils import downsample_array



def plot_histogram(image, bins=None, title="Gray level histogram",
                   save_path=None, no_bkg=False, show=True, normalize=False,
                   alpha=1, downsampling=None, ax=None):
    """
    Plots histogram of a gray-scale SimpleITK 3D image.
    
    Parameters
    ----------
        image (SimpleITK.Image): Image to compute the histogram.
        bins (int or str): Number of bins or binning strategy.
            If None, uses the maximum gray level (recommended for integer gray levels).
        title (str): Title of the figure.
        save_path (str): Saves the histogram to the desired path.
        no_bkg (bool): If True, removes gray level 0 (background) from the histogram.
        alpha (float): Transparency level for overlapping histograms.
        normalize (bool): Whether to normalize the histograms to show densities.
        show (bool): Whether to show the plot (ignored if ax is provided).
        downsampling (float or None): Percentage of voxels to use for histogram calculation.
        ax (matplotlib.axes.Axes): Axis on which to plot.
            If not specified, uses current axis.
    
    Returns
    -------
        hist (tuple): (counts: np.ndarray, bin_edges: np.ndarray)
    """

    image_array = get_array_from_image(image)
    arr = image_array.flatten()

    if no_bkg:
        arr = arr[arr != 0]

    if downsampling is not None:
        arr = downsample_array(arr, perc=downsampling)

    if bins is None:
        bins = int(max(np.unique(arr)))
    
    hist = np.histogram(arr, bins=bins, density=normalize)
    counts, bins_edges = hist
    bins_width = bins_edges[1] - bins_edges[0]
    bins_center = (bins_edges[:-1] + bins_edges[1:]) / 2

    if arr.size != 0 and show:
        if ax is None:
            ax=plt.gca()
        ax.bar(bins_center, counts, alpha=alpha, width=bins_width, align='center')
        xlabel = "Gray level (excluding black)" if no_bkg else "Gray level"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.set_title(title)


    if save_path is not None:
        plt.savefig(save_path)

    return hist



def plot_multiple_histograms(images, bins=None, labels=None, title="Gray level histograms",
                              legend_title="", save_path=None, no_bkg=False, alpha=0.5,
                              normalize=False, downsampling=None, show=True, ax=None):
    """
    Plots histograms for multiple SimpleITK 3D gray-scale images.

    Parameters
    ----------
        images (list): List of SimpleITK.Image objects.
        bins (int, list or None): 
            If int, number of bins for all images. If list, must match length of images.
            If None, uses maximum gray level per image (recommended for integer gray levels).
        labels (list): List of labels for each histogram (for the legend).
        title (str): Title of the plot.
        legend_title (str): Title of the legend.
        save_path (str): Path to save the figure.
        no_bkg (bool): If True, exclude gray level 0 from histograms.
        alpha (float): Transparency level for overlapping histograms.
        normalize (bool): Whether to normalize the histograms to show densities.
        downsampling (float or None): Percentage of voxels to use for histogram calculation.
        show (bool): Whether to display the plot.
        ax (matplotlib.axes.Axes): Matplotlib Axes object where to draw.
            If not specified, uses current axis.

    Returns
    -------
        histograms (list): List of NumPy histograms in the format (counts: np.ndarray, bin_edges: np.ndarray).
    """

    if labels is None:
        labels = [f"Image {i+1}" for i in range(len(images))]

    if show and ax is None:
        ax=plt.gca()


    hist_tables = []
    for idx, (image, label) in enumerate(zip(images, labels)):    
        
        arr = get_array_from_image(image).flatten()

        if no_bkg:
            arr = arr[arr != 0]
        
        if downsampling is not None:
            arr = downsample_array(arr, perc=downsampling)
        
        if isinstance(bins, list):
            my_bins = bins[idx]
        else:
            my_bins = bins
        if my_bins is None:
            my_bins = int(max(np.unique(arr)))
        
        hist = np.histogram(arr, bins=my_bins, density=normalize)
        counts, bins_edges = hist
        bins_width = bins_edges[1] - bins_edges[0]
        bins_center = (bins_edges[:-1] + bins_edges[1:]) / 2

        if arr.size != 0 and show:
            ax.bar(bins_center, counts, alpha=alpha, label=label,
                   width=bins_width, align='center')

        hist_tables.append(hist)

    if show:
        ax.set_xlabel("Gray level (excluding black)" if no_bkg else "Gray level")
        ax.set_ylabel("Density" if normalize else "Counts")
        ax.set_title(title)
        ax.legend(title=legend_title)

    if save_path:
        plt.savefig(save_path)


    return hist_tables

# !!!! per ora inutile...
def histogram_stats(hist, q1=25, q2=75):
    """
    Estimate the mean and two percentiles from a histogram.

    Parameters
    ----------
        hist (tuple): Tuple (counts, bin_edges) as returned by np.histogram().
        q1 (float): First percentile to compute (default is 25).
        q2 (float): Second percentile to compute (default is 75).

    Returns
    -------
        tuple: (mean, percentile_q1, percentile_q2, variance, skwness, kurtosis)
    """
    counts, bin_edges = hist
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean = np.average(bin_centers, weights=counts)

    # compute cdf
    cdf = np.cumsum(counts).astype(float)
    cdf /= cdf[-1]  # normalize

    # percentiles from the cdf
    p1 = np.interp(q1 / 100, cdf, bin_centers)
    p2 = np.interp(q2 / 100, cdf, bin_centers)

    # statistical moments
    variance = np.average((bin_centers - mean) ** 2, weights=counts)
    skewness = skew(np.repeat(bin_centers, counts.astype(int)))
    kurt = kurtosis(np.repeat(bin_centers, counts.astype(int)))

    return mean, p1, p2, variance, skewness, kurt


def gaussian_smooth_histogram(hist, sigma=3, show=True, ax=None):
    """
    Smoothens an histogram convoluting with a gaussian kernel, and optionally plots it.

    Parameters
    ----------
        hist (tuple): Tuple (counts, bin_edges) as returned by np.histogram().
        sigma (float): Standard deviation of the gaussian kernel.
        show (bool): Whether to display the plot.
        ax (matplotlib.axes.Axes): Matplotlib Axes object where to draw.
            If not specified, uses current axis.

    Returns
    -------
        smooth_hist (tuple):  Tuple (counts, bin_edges) of the smoothed histogram.
    """

    counts, bins = hist
    bins_center = (bins[:-1] + bins[1:]) / 2
    smooth_counts = gaussian_filter1d(counts, sigma=sigma)
    if show:
        if ax is None:
            ax = plt.gca()
        ax.plot(bins_center, smooth_counts, 'r-', linewidth=2, 
            alpha=0.7, label=f'Gaussian smoothing (σ={sigma})')
    smooth_hist = (smooth_counts, bins)
    return smooth_hist


def mode_and_rhwhm(hist, show=True, ax=None):
    """
    Computes mode (most frequent gray level) and Right-side Half Width at Half Maximum,
    and optionally plots them.
    
    Parameters
    ----------
        hist (tuple): Tuple (counts, bin_edges) as returned by np.histogram().
        show (bool): Whether to display the plot.
        ax (matplotlib.axes.Axes): Matplotlib Axes object where to draw.
            If not specified, uses current axis.

    Returns
    -------
        mode (float): Most frequent gray level.
        rfwhm (float): Right-side FWHM.
    """
    counts, bins = hist
    bins_center = (bins[:-1] + bins[1:]) / 2

    mode = bins_center[np.argmax(counts)]

    hm = np.max(counts) / 2     # half maximum
    less_than_hm_gl = bins_center[counts < hm]      # gl where counts < hm
    right_less_than_hm_gl = less_than_hm_gl[less_than_hm_gl > mode]     # only right side
    right_hm_gl = np.min(right_less_than_hm_gl)     # first time hist crosses the hm line
    rhwhm = right_hm_gl - mode

    if show:
        if ax is None:
            ax = plt.gca()
        ax.axvline(mode, linestyle='--', color='red',
                linewidth=2,label=f"Mode ({mode:.1f})")
        ax.annotate('',
            xy=(mode, hm),  
            xytext=(right_hm_gl, hm),
            arrowprops=dict(
                arrowstyle='<->',
                color='red',
                lw=2,
                shrinkA=0, 
                shrinkB=0,
                ),
            )
        x_text = (mode + right_hm_gl) / 2
        y_text = hm - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(
            x_text, y_text,
            f'Right-side HWHM\n({rhwhm:.1f})',
            color='black',
            ha='center',
            va='top',
        )
        
    return mode, rhwhm


