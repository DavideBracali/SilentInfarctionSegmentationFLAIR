#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 20:16:28 2025

@author: david
"""


import numpy as np
import SimpleITK as sitk
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter1d


class DimensionError(Exception):
    pass


def check_3d(image):
    """
    Checks is the SimpleITK image is 3-dimensional, otherwise raises a DimensionError.
    
    Parameters
    ----------
        image (SimpleITK.Image): SimpleITK image object.
    """
    if image.GetDimension() != 3:
        raise DimensionError("Image must be 3-dimensional.")


def get_info(image):
    """
    Extracts size, spacing, origin and direction matrix from a SimpleITK 3D image.
    
    Parameters
    ----------
        image (SimpleITK.Image): The image to get information from.
        
    Returns
    -------
        info (dict): Size, spacing, origin and direction matrix.
    """ 
    info = {
       "size": image.GetSize(),           
       "spacing": image.GetSpacing(),     
       "origin": image.GetOrigin(),       
       "direction": image.GetDirection()} 
    
    return info


def get_array_from_image(image):
    
    """
    Extracts Numpy array from a SimpleITK 3D image.
    Trasposes it in a format so that:
        Axial plane is xy plane;
        Sagittal plane is yz plane;
        Coronal plane is xz plane;
    
    Parameters
    ----------
        image (SimpleITK.Image): SimpleITK image object
        
    Returns
    -------
        image_array (np.array): NumPy array of the image
        
    """

    check_3d(image)
    
    image_array = sitk.GetArrayFromImage(image)
    image_array = np.transpose(image_array, (2,1,0))       # SimpleITK returns zyx
        
    return image_array


def plot_image(image, xyz=None):
    """
    Plots a 3D image using SimpleITK, Matplotlib.pyplot, Seaborn.
    You can specify the intersection between the three planes. By default it is
    set to the center of the image.
    
    Parameters
    ----------
        image (SimpleITK.Image): The image to be plotted.
        xyz (tuple): Intersection between the three planes of the 3D image.
    
    Returns
    -------
        array_info (dict): Array size, spacing and aspects.

    """
    image_array = get_array_from_image(image)

    
    # default values: center of the image
    if xyz == None:
        xyz = tuple(int(np.round(image_array.shape[i] / 2)) for i in range(3))

    # get spacing
    
    sx, sy, sz = image.GetSpacing()
    
    sagittal_aspect = sz / sy
    axial_aspect = sy / sx
    coronal_aspect = sz / sx
    
    # plot
    with sns.plotting_context('notebook'):
    
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(7, 14))
    
        _ = ax0.axis('off')
        _ = ax1.axis('off')
        _ = ax2.axis('off')
        
        _ = ax0.imshow(image_array[:,:,xyz[2]].T, cmap='gray', 
                       origin='lower', aspect=axial_aspect)        #xy
        _ = ax1.imshow(image_array[xyz[0],:,:].T, cmap='gray',
                       origin='lower', aspect=sagittal_aspect)     #yz
        _ = ax2.imshow(image_array[:,xyz[1],:].T, cmap='gray',
                       origin='lower', aspect=coronal_aspect)      #xz

        _ = ax0.set_title('Axial')
        _ = ax1.set_title('Sagittal')
        _ = ax2.set_title('Coronal')

    plt.show()
    plt.close()
        
    # plot info
    plot_info = {
       "size": image_array.shape,           
       "spacing": (sx, sy, sz),     
       "aspects": (axial_aspect, sagittal_aspect, coronal_aspect)}
    
    return plot_info
        

def orient_image(image, orientation):
    
    """
    Orients a SimpleITK image to a specified coordinate orientation system.
    
    Parameters
    ----------
        image (SimpleITK.Image): The input image to be reoriented.
        orientation (str): The desired coordinate orientation, such as 'RAS' or 'LPS'.
    
    Returns
    -------
        oriented_image (SimpleITK.Image): The image reoriented to the specified system.
    """
    check_3d(image)
    orient_filter = sitk.DICOMOrientImageFilter()           
    orient_filter.SetDesiredCoordinateOrientation(orientation) 
    oriented_image = orient_filter.Execute(image)
    
    return oriented_image


def resample_to_reference(image, reference, interpolator=sitk.sitkNearestNeighbor, default_value=0):
    
    """
    Resamples moving_image onto the space of reference_image.

    Parameters
    ----------
        moving_image (SimpleITK.Image): The image to be resampled.
        reference_image (SimpleITK.Image): The target space.
        interpolator (SimpleITK interpolator): e.g., sitk.sitkLinear, sitk.sitkNearestNeighbor.
        default_value (float): Value for areas outside original image.

    Returns
    -------
        resampled_image (SimpleITK.Image): The resampled image in reference space.
    """
    check_3d(image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    image_rs = resampler.Execute(image)
    return image_rs


def downsample_array(arr, perc=0.01):
    """
    Returns an evenly spaced subsample of a 1D array.
    Useful to reduce memory and computational complexity on very large arrays.

    Parameters
    ----------
        arr (np.ndarray): Input 1D array.
        perc (float): Percentage of elements to keep (as a decimal, e.g. 0.01 = 1%).

    Returns
    -------
        arr_ds (np.ndarray): Evenly spaced downsampled array.
    """
    if arr.size == 0 or perc >= 1.0:
        return arr
    if perc <= 0:
        raise ValueError("Invalid downsample percentage.")
    step = max(1, int(1/perc))
    return arr[::step]


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
            if ax is None:
                ax=plt.gca()
            ax.bar(bins_center, counts, alpha=alpha, label=label,
                   width=bins_width, align='center')

        hist_tables.append(hist)

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
    Smoothens an histogram convoluting with a gaussian kernel.

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
        ax.legend()
    smooth_hist = (smooth_counts, bins)
    return smooth_hist