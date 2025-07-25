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
    image_array = np.transpose(image_array, (2,1,0))   # SimpleITK returns zyx
        
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


def resample_to_reference(image, reference, interpolator=sitk.sitkLinear, default_value=0):
    
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


def plot_histogram(image, bins=None, title="Gray level histogram",
                   save_path=None, no_bkg=False):
    """
    Plots histogram of a gray-scale SimpleITK 3D image.
    
    Parameters
    ----------
        image (SimpleITK.Image): Image to compute the histogram.
        bins (int): Number of bins.
        title (str): Title of the figure.
        save_path (str): Saves the histogram to the desired path.
        no_bkg (boolean): If True, removes gray level 0 (background) from the histogram.
    
    Returns
    -------
        histogram (np.array): Flattened array of gray levels.
    """
    image_array = get_array_from_image(image)
    flattened = image_array.flatten()
    
    if no_bkg == True:
        flattened = flattened[flattened != 0]
    
    if bins == None:
        bins = range(int(min(flattened)), int(max(flattened)) + 2)
    
    plt.hist(flattened, bins=bins)
    if no_bkg == True:
        plt.xlabel("Gray level (excluding black)")
    else:
        plt.xlabel("Gray level")
    plt.ylabel("Counts")
    plt.title(title)
    
    if save_path != None:
        plt.savefig(save_path)

    plt.show()    
    plt.close()

    return flattened


def plot_multiple_histograms(images, labels=None, bins=None, title="Gray level histograms",
                              save_path=None, no_bkg=False, alpha=0.5, normalize=False):
    """
    Plots histograms for multiple SimpleITK 3D gray-scale images.

    Parameters
    ----------
        images (list): List of SimpleITK.Image objects.
        labels (list): List of labels for each histogram (for the legend).
        bins (int or list): Number of bins or shared bin edges.
        title (str): Title of the plot.
        save_path (str): Path to save the figure.
        no_bkg (bool): If True, exclude gray level 0 from histograms.
        alpha (float): Transparency level for overlapping histograms.

    Returns
    -------
        histograms (list): List of flattened gray-level arrays for each image.
    """

    if labels == None:
        labels = [f"Image {i+1}" for i in range(len(images))]

    histograms = []
    all_values = []

    for image in images:
        flattened = get_array_from_image(image).flatten()
        if no_bkg:
            flattened = flattened[flattened != 0]
        histograms.append(flattened)
        all_values.extend(flattened)

    if bins == None:
        bins = range(int(min(all_values)),
                     int(max(all_values)) + 2)

    for hist, label in zip(histograms, labels):
        plt.hist(hist, bins=bins, alpha=alpha, label=label,
                  density=normalize)

    plt.xlabel("Gray level (excluding 0)" if no_bkg else "Gray level")
    plt.ylabel("Density" if normalize else "Counts")
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()

    return histograms