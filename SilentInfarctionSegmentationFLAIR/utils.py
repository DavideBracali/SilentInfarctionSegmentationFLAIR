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
import time
import os
import pandas as pd


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


def get_image_from_array(image_array, reference_image=None):
    """
    Converts a NumPy array back to a SimpleITK image.
    Assumes the input array has axes:
        Axial plane = xy plane
        Sagittal plane = yz plane
        Coronal plane = xz plane
    
    Parameters
    ----------
        image_array (np.array): NumPy array with shape (x, y, z)
        reference_image (SimpleITK.Image, optional): 
            If provided, copy spacing, origin, and direction from this image.
    
    Returns
    -------
        image (SimpleITK.Image): SimpleITK image
    """
    
    sitk_array = np.transpose(image_array, (2, 1, 0))       # SimpleITK expects zyx
    
    image = sitk.GetImageFromArray(sitk_array)
    
    if reference_image is not None:
        image.CopyInformation(reference_image)
        image = sitk.Cast(image, reference_image.GetPixelID())
    
    return image


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


def get_paths_df(folder, extensions=""):
    """
    Returns paths DataFrame containing all files paths in a folder.

    Parameters
    ----------
        - folder (str): Path of the folder to collect file paths.
    
    Returns
    -------
        - paths_df (pandas.DataFrame): DataFrame containing folder paths as indexes
            and file types as columns.
    """
    if not isinstance(extensions, list):
        extensions = [extensions]

    paths_list = []
    for root, _, files in os.walk(folder):
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    paths_list.append((root,file))

    paths_df = pd.DataFrame()
    for root, file in paths_list:
        paths_df.loc[root,file] = os.path.join(root,file)

    return paths_df


def label_names(label_name_file_path):
    """
    Reads a .txt file and assigns a string label to each numeric label.

    File format:
        - Each line should start with a number followed by the label name.
        - Lines not starting with a number are ignored.
        - Only the first word after the number is used as the label name.
          For example:
            '1 Left-Thalamus' -> label 1 is 'Left-Thalamus'
            '2 Left Thalamus' -> label 2 is 'Left' (only the first word is taken)

    Parameters
    ----------
        label_name_file_path (str): Path to the label .txt file.

    Returns
    -------
        label_dict (dict): Mapping from integer label numbers to string names.
    """
    label_dict = {}
    with open(label_name_file_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            parts = line.split()
            num = int(parts[0])
            name = parts[1]
            label_dict[num] = name
    
    return label_dict


def progress_bar(iteration, total, start_time=None, prefix="", length=40):
    """
    Displays or updates a textual progress bar in the console.

    Parameters
    ----------
        iteration (int): Current iteration (0-based).
        total (int): Total number of iterations.
        start_time (float, optional): Start time of the operation, e.g., time.time().
            If provided, an estimated remaining time (ETA) is displayed.
        prefix (str, optional): Text to display before the progress bar.
        length (int, optional): Character length of the progress bar.

    Notes
    -----
        - The progress bar is updated in-place using carriage return.
        - At the last iteration, a newline is printed to end the bar.
        - Handles edge cases where iteration >= total or total <= 0.
    """
    
    if total <= 0:
        total = 1
    iteration = min(iteration, total - 1)

    percent = f"{100 * ((iteration + 1) / float(total)):.1f}"
    filled_length = int(length * (iteration + 1) // total)
    bar = "█" * filled_length + "-" * (length - filled_length)

    eta_str = ""
    if start_time is not None:
        elapsed = time.time() - start_time
        avg_time = elapsed / (iteration + 1)
        remaining = int(round(avg_time * (total - (iteration + 1))))
        h = remaining // 3600
        m = (remaining % 3600) // 60
        s = remaining % 60
        eta_str = f" ETA: {h:d}:{m:02d}:{s:02d}"

    line = f"\r{prefix} |{bar}| {percent}% ({iteration+1}/{total}){eta_str}"
    print(line.ljust(120), end="\r")

    if iteration + 1 >= total:
        print()


def to_n_bit(image, n_bits=8):
    """
    Normalize an image to 0..(2^n_bits - 1) and cast to unsigned integer.

    Parameters
    ----------
    image : SimpleITK.Image
        Input image (can be float or integer)
    n_bits : int
        Number of bits for output image (default 8)

    Returns
    -------
    SimpleITK.Image
        Image normalized and cast to UInt{8,16,32} depending on n_bits
    """
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    min_val = stats.GetMinimum()
    max_val = stats.GetMaximum()
    
    # data type
    if n_bits <= 8:
        voxel_type = sitk.sitkUInt8
        dtype = np.uint8
    elif n_bits <= 16:
        voxel_type = sitk.sitkUInt16
        dtype = np.uint16
    else:
        voxel_type = sitk.sitkUInt32
        dtype = np.uint32

    # uniform image
    if max_val == min_val:
        scaled = sitk.Image(image.GetSize(), voxel_type)
        scaled.CopyInformation(image)
        return scaled

    arr = sitk.GetArrayFromImage(image)
    arr = (arr - min_val) / (max_val - min_val)  # 0..1
    arr = arr * (2**n_bits - 1)                 # 0..2^n_bits-1
    arr = np.round(arr).astype(dtype)
    
    is_vector = image.GetNumberOfComponentsPerPixel() > 1
    image_uint = sitk.GetImageFromArray(arr, isVector=is_vector)
    image_uint.CopyInformation(image)
    
    return image_uint