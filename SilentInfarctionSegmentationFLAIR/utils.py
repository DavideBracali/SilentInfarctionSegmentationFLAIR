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


def get_image_from_array(image_array, reference_image=None, cast_to_reference=True):
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
        cast_to_reference (bool, optional):
            Whether to set voxel data type to reference image.
            If True, inherits voxel type from reference image if provided.
            If False, sets voxel type automatically from array. 
    
    Returns
    -------
        image (SimpleITK.Image): SimpleITK image from array.
    """
    
    sitk_array = np.transpose(image_array, (2, 1, 0))       # SimpleITK expects zyx
    
    image = sitk.GetImageFromArray(sitk_array)
    
    if reference_image is not None:
        image.CopyInformation(reference_image)
        
        if cast_to_reference:
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


def normalize(image, n_bits=8):
    """
    Normalize an image to 0..(2^n_bits - 1) using a min-max rescaling.
    If n_bits == 0 and cast to float64, else cast to unsigned integer.

    Parameters
    ----------
    image : SimpleITK.Image
        Input image (can be float or integer)
    n_bits : int
        Number of bits for output image (default 8)

    Returns
    -------
    SimpleITK.Image
        Image normalized and cast to UInt{8,16,32} or float32 depending on n_bits
    """
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    min_val = stats.GetMinimum()
    max_val = stats.GetMaximum()
    
    # data type
    if n_bits <= 1:
        voxel_type = sitk.sitkFloat64
        dtype = np.float64
    elif n_bits <= 8:
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
    
    if n_bits > 1:
        arr = arr * (2**n_bits - 1)                 # 0..2^n_bits-1
        arr = np.round(arr).astype(dtype)
    
    is_vector = image.GetNumberOfComponentsPerPixel() > 1
    image_uint = sitk.GetImageFromArray(arr, isVector=is_vector)
    image_uint.CopyInformation(image)
    
    return image_uint

def train_val_test_split(data_folder, validation_fraction=0, test_fraction=0, pos_neg_stratify=False,
                     gt_file="GT.nii", show=False, save_series=True, title=None):

    """
    Split patients into train, validation and test sets based on imaging files found
    in the given folder, optionally stratifying according to the ratio of positive/negative
    labels in the ground-truth image.

    Parameters
    ----------
    data_folder : str
        Path to the root folder containing patient subdirectories, each holding
        `.nii` files including the ground truth file.
    validation_fraction : float, optional
        Fraction of patients to include in the validation set. Default is 0.
    test_fraction : float, optional
        Fraction of patients to include in the test set. Default is 0.
    pos_neg_stratify : bool, optional
        If True, perform stratified sampling based on quartiles of the positive-label
        ratio. Default is False.
    gt_file : str, optional
        Filename of the ground truth image inside each patient directory. Default is "GT.nii".
    show : bool, optional
        Whether to show the ratio distribution plot. Default is False.

    Returns
    -------
    tr_patients : list of str
        List of patient directory names assigned to the training set.
    val_patients : list of str
        List of patient directory names assigned to the validation set.
    ts_patients : list of str
        List of patient directory names assigned to the test set.

    Notes
    -----
    - Patients missing one of the required `.nii` files are discarded.
    - Positive/negative ratio is defined as:

      ``ratio = count(label == 1) / (count(label == 0) + count(label == 1))``

    - If `pos_neg_stratify=True`, the stratification is computed by dividing
      patient ratios into quartiles and splitting proportionally within each group.
    - A `ratios.png` file is saved one directory above `data_folder`, showing the
      distributions of positive-label ratios for train/validation/test sets.
    """

    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"data_folder not found: {data_folder}")

    if not (0 <= validation_fraction < 1):
        raise ValueError("validation_fraction must be in [0, 1)")

    if not (0 <= test_fraction < 1):
        raise ValueError("test_fraction must be in [0, 1)")

    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1")

    # organize all images in a df
    paths_list = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".nii"):
                paths_list.append((root, file))

    paths_df = pd.DataFrame()
    for root, file in paths_list:
        patient = os.path.basename(root)
        paths_df.loc[patient, file] = os.path.join(root, file)

    if len(paths_list) == 0:
        raise RuntimeError("No .nii files found inside data_folder")

    # drop NaN
    dropped_patients = paths_df.index[paths_df.isna().any(axis=1)]
    if len(dropped_patients) > 0:
       print("WARNING!!! The following patients will be removed because " \
                      f"one of the required images is missing: {list(dropped_patients)}")
    paths_df = paths_df.dropna(how="any")

    all_patients = list(paths_df.index)

    ratios = pd.Series(index=all_patients)
    for patient in all_patients:
        gt = sitk.ReadImage(paths_df.loc[patient, gt_file])
        label_stats = sitk.LabelStatisticsImageFilter()
        label_stats.Execute(gt, gt)  # same image as input and label
        count_0 = label_stats.GetCount(0)
        count_1 = label_stats.GetCount(1) if label_stats.HasLabel(1) else 0
        if count_0 + count_1 == 0:
            raise ValueError(f"Ground truth for patient {patient} has no valid labels.")
        else:
            ratios.loc[patient] = (count_1 / (count_0 + count_1))
        
    q1 = ratios.quantile(0.25)
    q2 = ratios.quantile(0.50)
    q3 = ratios.quantile(0.75)

    quartiles = pd.Series(index=all_patients, dtype=int)
    quartiles[ratios <= q1] = 1
    quartiles[(ratios > q1) & (ratios <= q2)] = 2
    quartiles[(ratios > q2) & (ratios <= q3)] = 3
    quartiles[ratios > q3] = 4

    tr_patients = []
    val_patients = []
    ts_patients = []

    if pos_neg_stratify:
        total_n = len(all_patients)
        total_val = int(round(total_n * validation_fraction))
        total_ts = int(round(total_n * test_fraction))

        assigned_val = 0
        assigned_ts = 0
        remaining_pool = []

        # proportional allocation per quartile
        for q in sorted(quartiles.unique()):
            group = list(quartiles[quartiles == q].index)
            n = len(group)
            if n == 0:
                continue

            prop = n / total_n
            n_val_q = int(round(total_val * prop))
            n_ts_q = int(round(total_ts * prop))

            # cap by remaining budgets
            n_val_q = min(n_val_q, total_val - assigned_val)
            n_ts_q = min(n_ts_q, total_ts - assigned_ts)

            # cap by group size
            if n_val_q + n_ts_q > n:
                excess = (n_val_q + n_ts_q) - n
                reduce_ts = min(excess, n_ts_q)
                n_ts_q -= reduce_ts
                excess -= reduce_ts
                if excess > 0:
                    reduce_val = min(excess, n_val_q)
                    n_val_q -= reduce_val

            np.random.shuffle(group)

            val_patients.extend(group[:n_val_q])
            assigned_val += n_val_q

            ts_patients.extend(group[n_val_q:n_val_q + n_ts_q])
            assigned_ts += n_ts_q

            remaining_pool.extend(group[n_val_q + n_ts_q:])

        # fix rounding leftovers
        np.random.shuffle(remaining_pool)

        need_val = total_val - assigned_val
        if need_val > 0:
            val_patients.extend(remaining_pool[:need_val])
            remaining_pool = remaining_pool[need_val:]

        need_ts = total_ts - assigned_ts
        if need_ts > 0:
            ts_patients.extend(remaining_pool[:need_ts])
            remaining_pool = remaining_pool[need_ts:]

        tr_patients.extend(remaining_pool)

        # enforce exact totals
        if len(val_patients) < total_val:
            diff = total_val - len(val_patients)
            move = tr_patients[:diff]
            val_patients.extend(move)
            tr_patients = tr_patients[diff:]

        if len(ts_patients) < total_ts:
            diff = total_ts - len(ts_patients)
            move = tr_patients[:diff]
            ts_patients.extend(move)
            tr_patients = tr_patients[diff:]

    else:
        n = len(all_patients)
        total_val = int(round(n * validation_fraction))
        total_ts = int(round(n * test_fraction))
        total_tr = n - total_val - total_ts

        np.random.shuffle(all_patients)
        tr_patients = all_patients[:total_tr]
        val_patients = all_patients[total_tr:total_tr + total_val]
        ts_patients = all_patients[total_tr + total_val:total_tr + total_val + total_ts]

    # build plotting df
    df_plot = pd.DataFrame(
        [{"Set": "Train", "Ratio": ratios[p]} for p in tr_patients] +
        [{"Set": "Validation", "Ratio": ratios[p]} for p in val_patients] +
        [{"Set": "Test", "Ratio": ratios[p]} for p in ts_patients]
    )

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_plot,
        x="Set",
        y="Ratio",
        showcaps=False,
        showfliers=False,
        boxprops={'facecolor':'none'},
        whiskerprops={'linewidth':1},
        medianprops={'linewidth':1}
    )
    sns.stripplot(data=df_plot, x="Set", y="Ratio", jitter=0.1, size=8)
    plt.grid(axis='y', alpha=0.3)
    if show:    plt.show()
    if title is not None: plt.title(title)
    plt.ylabel("positive/total voxels ratio")
    plt.savefig(os.path.join(data_folder, "ratios.png"))
    plt.close()

    if save_series:
        pd.Series(tr_patients).to_pickle(os.path.join(data_folder, "train_patients.pkl"))
        pd.Series(val_patients).to_pickle(os.path.join(data_folder, "validation_patients.pkl"))
        pd.Series(ts_patients).to_pickle(os.path.join(data_folder, "test_patients.pkl"))

    return tr_patients, val_patients, ts_patients


def gaussian_transform(image, mean, std, return_float = False, normalized = False):
    """
    Apply a Gaussian weighting to the voxel intensities of a SimpleITK image.

    Each voxel intensity `I` is transformed according to:
        G(I) = exp(-0.5 * ((I - mean) / std)²) * I
    If `return_float=True` and `normalized=True`, the result is further scaled by:
        1 / (std * sqrt(2π))

    Parameters
    ----------
    image : SimpleITK.Image
        Input image with non-negative voxel intensities.
    mean : float
        Mean of the Gaussian distribution (center of enhancement).
    std : float
        Standard deviation of the Gaussian (controls the enhancement width).
    return_float : bool, optional (default=False)
        If True, return a float-valued image between 0 and 1.
        If False, the result is cast back to the original voxel type.
    normalized : bool, optional (default=False)
        If True and `return_float=True`, apply the normalization factor
        1 / (std * sqrt(2π)).

    Returns
    -------
    SimpleITK.Image
        The Gaussian-weighted image. Spacing, origin and direction are
        preserved from the input image.

    Raises
    ------
    ValueError
        If the input image contains negative voxel intensities.
    """
    arr = get_array_from_image(image)

    if np.any(arr < 0):
        raise ValueError("The input image contains negative values.")

    if return_float:
        if normalized:
            norm = 1 / (std * np.sqrt(2*np.pi))
        else:
            norm = 1
        # return transformed image of floats between 0 and 1 
        gaussian_arr = norm * np.exp(-0.5 * ((arr - mean) / std)**2)
        return get_image_from_array(gaussian_arr, image, cast_to_reference=False)

    else:
        gaussian_arr = np.exp(-0.5 * ((arr - mean) / std)**2) * arr
        # return transformed image with the same voxel type 
        return get_image_from_array(gaussian_arr, image, cast_to_reference=True)

def cliffs_delta(x, y):
    diff = x[:, None] - y[None, :]
    return (np.sum(diff > 0) - np.sum(diff < 0)) / (x.size * y.size)