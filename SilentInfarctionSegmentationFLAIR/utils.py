#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Utility functions for 3D SimpleITK image handling and visualization.

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
from importlib.resources import files
from pathlib import Path


class DimensionError(Exception):
    """Custom exception for non‑3D images."""
    pass

def get_settings_path(filename: str) -> Path:
    """
    Return an absolute Path to a file inside the package's 'settings' folder.

    Parameters
    ----------
    filename : str
        Name of the file in the 'settings' folder, e.g. "config.yaml"

    Returns
    -------
    Path
        Absolute path to the resource on disk.
    """
    return Path(files("SilentInfarctionSegmentationFLAIR") / "settings" / filename)

def check_3d(image):
    """
    Ensure that a SimpleITK image is 3‑dimensional.

    Parameters
    ----------
    image : SimpleITK.Image
        Input image.

    Raises
    ------
    DimensionError
        If the image is not 3‑dimensional.
    """
    if image.GetDimension() != 3:
        raise DimensionError("Image must be 3‑dimensional.")


def get_info(image):
    """
    Extract metadata from a SimpleITK 3D image.

    Parameters
    ----------
    image : SimpleITK.Image
        Input image.

    Returns
    -------
    dict
        Dictionary containing size, spacing, origin and direction.
    """
    return {
        "size": image.GetSize(),
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
    }


def get_array_from_image(image):
    """
    Convert a SimpleITK 3D image into a NumPy array.

    The returned array is transposed so that:
    - axial plane   → xy
    - sagittal plane → yz
    - coronal plane → xz

    Parameters
    ----------
    image : SimpleITK.Image
        Input 3D image.

    Returns
    -------
    np.ndarray
        Array with shape (x, y, z).
    """
    check_3d(image)
    arr = sitk.GetArrayFromImage(image)      # z, y, x
    return np.transpose(arr, (2, 1, 0))      # x, y, z


def get_image_from_array(image_array, reference_image=None,
                         cast_to_reference=True):
    """
    Convert a NumPy array into a SimpleITK image.

    The input array must follow the convention:
    - axial plane   → xy
    - sagittal plane → yz
    - coronal plane → xz

    Parameters
    ----------
    image_array : np.ndarray
        Array with shape (x, y, z).
    reference_image : SimpleITK.Image, optional
        If provided, copy spacing, origin and direction.
    cast_to_reference : bool, optional
        If True, cast voxel type to match ``reference_image``.

    Returns
    -------
    SimpleITK.Image
        Output image.
    """
    sitk_arr = np.transpose(image_array, (2, 1, 0))  # z, y, x
    image = sitk.GetImageFromArray(sitk_arr)

    if reference_image is not None:
        image.CopyInformation(reference_image)
        if cast_to_reference:
            image = sitk.Cast(image, reference_image.GetPixelID())

    return image

def plot_image(image, xyz=None, mask=None, title=None,
               show=True, save_path=None):
    """
    Plot three orthogonal slices (axial, sagittal, coronal) of a 3D image.

    Parameters
    ----------
    image : SimpleITK.Image
        Input 3D image.
    xyz : tuple of int, optional
        Coordinates of the slice intersection (x, y, z). If None, the
        center of the image is used.
    mask : SimpleITK.Image, optional
        Binary mask to overlay on the image. Masked voxels are shown in
        red.
    title : str, optional
        Title of the figure.
    show : bool, optional
        Whether to display the figure.
    save_path : str, optional
        Path where the figure is saved.

    Returns
    -------
    dict
        Dictionary containing array size, spacing and aspect ratios.
    """
    image_array = get_array_from_image(image)

    if xyz is None:
        xyz = tuple(int(round(image_array.shape[i] / 2))
                    for i in range(3))

    sx, sy, sz = image.GetSpacing()

    sagittal_aspect = sz / sy
    axial_aspect = sy / sx
    coronal_aspect = sz / sx

    with sns.plotting_context("notebook"):
        fig, (ax0, ax1, ax2) = plt.subplots(
            nrows=1, ncols=3, figsize=(7, 7)
        )

        ax0.axis("off")
        ax1.axis("off")
        ax2.axis("off")

        if mask is not None:
            mask_arr = get_array_from_image(mask).astype(bool)

            rgb_arr = np.stack(
                [
                    image_array,
                    image_array * (~mask_arr),
                    image_array * (~mask_arr),
                ],
                axis=-1,
            )

            xy = np.transpose(rgb_arr[:, :, xyz[2], :], (1, 0, 2))
            yz = np.transpose(rgb_arr[xyz[0], :, :, :], (1, 0, 2))
            xz = np.transpose(rgb_arr[:, xyz[1], :, :], (1, 0, 2))

            ax0.imshow(xy, origin="lower", aspect=axial_aspect)
            ax1.imshow(yz, origin="lower", aspect=sagittal_aspect)
            ax2.imshow(xz, origin="lower", aspect=coronal_aspect)

        else:
            ax0.imshow(
                image_array[:, :, xyz[2]].T,
                cmap="gray",
                origin="lower",
                aspect=axial_aspect,
            )
            ax1.imshow(
                image_array[xyz[0], :, :].T,
                cmap="gray",
                origin="lower",
                aspect=sagittal_aspect,
            )
            ax2.imshow(
                image_array[:, xyz[1], :].T,
                cmap="gray",
                origin="lower",
                aspect=coronal_aspect,
            )

        ax0.set_title("Axial")
        ax1.set_title("Sagittal")
        ax2.set_title("Coronal")

    if title is not None:
        plt.suptitle(title, y=0.8)

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.close()

    return {
        "size": image_array.shape,
        "spacing": (sx, sy, sz),
        "aspects": (axial_aspect, sagittal_aspect, coronal_aspect),
    }

def orient_image(image, orientation):
    """
    Reorient a SimpleITK image to a given coordinate system.

    Parameters
    ----------
    image : SimpleITK.Image
        Input 3D image.
    orientation : str
        Desired orientation code (e.g., 'RAS', 'LPS').

    Returns
    -------
    SimpleITK.Image
        Reoriented image.
    """
    check_3d(image)
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(orientation)
    return orient_filter.Execute(image)

def resample_to_reference(image, reference,
                          interpolator=sitk.sitkNearestNeighbor,
                          default_value=0):
    """
    Resample an image into the space of a reference image.

    Parameters
    ----------
    image : SimpleITK.Image
        Image to resample.
    reference : SimpleITK.Image
        Target reference image.
    interpolator : SimpleITK interpolator, optional
        Interpolation method (e.g., sitk.sitkLinear).
    default_value : float, optional
        Value assigned outside the original image domain.

    Returns
    -------
    SimpleITK.Image
        Resampled image.
    """
    check_3d(image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(image)

def downsample_array(arr, perc=0.01):
    """
    Evenly downsample a 1D NumPy array.

    Useful for reducing memory and computation on very large arrays by
    selecting regularly spaced samples.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array.
    perc : float, optional
        Fraction of elements to keep (e.g., 0.01 = 1%). Must be in
        (0, 1]. Default is 0.01.

    Returns
    -------
    np.ndarray
        Downsampled array.

    Raises
    ------
    ValueError
        If ``perc`` is <= 0.
    """
    if arr.size == 0 or perc >= 1.0:
        return arr
    if perc <= 0:
        raise ValueError("Invalid downsample percentage.")

    step = max(1, int(1 / perc))
    return arr[::step]


def get_paths_df(folder, extensions=""):
    """
    Build a DataFrame containing paths of files inside a folder.

    Parameters
    ----------
    folder : str
        Path to the folder to scan.
    extensions : str or list of str, optional
        File extensions to include. If a single string is provided, it is
        converted to a list.

    Returns
    -------
    pandas.DataFrame
        DataFrame where:
        - index = folder paths
        - columns = filenames
        - values = full file paths
    """
    if not isinstance(extensions, list):
        extensions = [extensions]

    paths_list = []
    for root, _, files in os.walk(folder):
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    paths_list.append((root, file))

    paths_df = pd.DataFrame()
    for root, file in paths_list:
        paths_df.loc[root, file] = os.path.join(root, file)

    return paths_df

def label_names(label_name_file_path):
    """
    Read a text file and map numeric labels to string names.

    The file must contain lines beginning with an integer followed by a
    label name. Only the first word after the number is used.

    Examples
    --------
    '1 Left-Thalamus' → {1: 'Left-Thalamus'}
    '2 Left Thalamus' → {2: 'Left'}

    Parameters
    ----------
    label_name_file_path : str
        Path to the label text file.

    Returns
    -------
    dict
        Mapping from integer labels to string names.
    """
    label_dict = {}
    with open(get_settings_path(label_name_file_path)) as f:
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
    Display or update a textual progress bar in the console.

    Parameters
    ----------
    iteration : int
        Current iteration (0‑based).
    total : int
        Total number of iterations.
    start_time : float, optional
        Start time (e.g., ``time.time()``). If provided, an ETA is shown.
    prefix : str, optional
        Text displayed before the bar.
    length : int, optional
        Length of the bar in characters.

    Notes
    -----
    - The bar updates in place using carriage return.
    - A newline is printed at the last iteration.
    - Handles cases where ``iteration >= total`` or ``total <= 0``.
    """
    if total <= 0:
        total = 1
    iteration = min(iteration, total - 1)

    percent = f"{100 * ((iteration + 1) / float(total)):.1f}"
    filled = int(length * (iteration + 1) // total)
    bar = "█" * filled + "-" * (length - filled)

    eta_str = ""
    if start_time is not None:
        elapsed = time.time() - start_time
        avg = elapsed / (iteration + 1)
        remaining = int(round(avg * (total - (iteration + 1))))
        h = remaining // 3600
        m = (remaining % 3600) // 60
        s = remaining % 60
        eta_str = f" ETA: {h:d}:{m:02d}:{s:02d}"

    line = (
        f"\r{prefix} |{bar}| {percent}% "
        f"({iteration+1}/{total}){eta_str}"
    )
    print(line.ljust(120), end="\r")

    if iteration + 1 >= total:
        print()

def normalize(image, n_bits=8):
    """
    Normalize an image using min‑max rescaling.

    If ``n_bits == 0``, the output is float64 in [0, 1].
    Otherwise, the output is cast to an unsigned integer type.

    Parameters
    ----------
    image : SimpleITK.Image
        Input image.
    n_bits : int, optional
        Number of bits for the output image. Default is 8.

    Returns
    -------
    SimpleITK.Image
        Normalized image with preserved metadata.
    """
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    min_val = stats.GetMinimum()
    max_val = stats.GetMaximum()

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

    if max_val == min_val:
        scaled = sitk.Image(image.GetSize(), voxel_type)
        scaled.CopyInformation(image)
        return scaled

    arr = sitk.GetArrayFromImage(image)
    arr = (arr - min_val) / (max_val - min_val)

    if n_bits > 1:
        arr = arr * (2**n_bits - 1)
        arr = np.round(arr).astype(dtype)

    is_vector = image.GetNumberOfComponentsPerPixel() > 1
    out = sitk.GetImageFromArray(arr, isVector=is_vector)
    out.CopyInformation(image)

    return out

def train_val_test_split(
    data_folder,
    validation_fraction=0,
    test_fraction=0,
    gt_file="GT.nii",
    show=False,
    save_series=True,
    title=None,
):
    """
    Split patients into train/validation/test sets based on ground‑truth
    lesion ratios.

    Patients are grouped by quartiles of the positive‑label ratio and
    sampled proportionally within each quartile.

    Parameters
    ----------
    data_folder : str
        Root folder containing patient subdirectories with `.nii` files.
    validation_fraction : float, optional
        Fraction of patients assigned to validation. Default is 0.
    test_fraction : float, optional
        Fraction of patients assigned to test. Default is 0.
    gt_file : str, optional
        Name of the ground‑truth file inside each patient folder.
    show : bool, optional
        Whether to display the ratio distribution plot.
    save_series : bool, optional
        Whether to save train/val/test patient lists as pickle files.
    title : str, optional
        Title for the ratio distribution plot.

    Returns
    -------
    tr_patients : list of str
        Patients assigned to the training set.
    val_patients : list of str
        Patients assigned to the validation set.
    ts_patients : list of str
        Patients assigned to the test set.

    Notes
    -----
    - Patients missing required `.nii` files are discarded.
    - Positive ratio is defined as::

          ratio = count(label == 1) / (count(label == 0) + count(label == 1))

    - Quartile‑based stratification ensures balanced sampling.
    - A plot ``ratios.png`` is saved in ``data_folder``.
    """
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"data_folder not found: {data_folder}")

    if not (0 <= validation_fraction < 1):
        raise ValueError("validation_fraction must be in [0, 1).")

    if not (0 <= test_fraction < 1):
        raise ValueError("test_fraction must be in [0, 1).")

    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1.")

    # ------------------------------------------------------------
    # Collect all .nii files
    # ------------------------------------------------------------
    paths_list = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".nii"):
                paths_list.append((root, file))

    if len(paths_list) == 0:
        raise RuntimeError("No .nii files found inside data_folder.")

    paths_df = pd.DataFrame()
    for root, file in paths_list:
        patient = os.path.basename(root)
        paths_df.loc[patient, file] = os.path.join(root, file)

    # Remove patients missing required files
    dropped = paths_df.index[paths_df.isna().any(axis=1)]
    if len(dropped) > 0:
        print(
            "WARNING: Removing patients missing required images: "
            f"{list(dropped)}"
        )
    paths_df = paths_df.dropna(how="any")

    all_patients = list(paths_df.index)

    # ------------------------------------------------------------
    # Compute positive/negative ratios
    # ------------------------------------------------------------
    ratios = {}
    excluded = []

    for patient in all_patients:
        gt = sitk.ReadImage(paths_df.loc[patient, gt_file])
        stats = sitk.LabelStatisticsImageFilter()
        stats.Execute(gt, gt)

        count_0 = stats.GetCount(0)
        count_1 = stats.GetCount(1) if stats.HasLabel(1) else 0

        if count_0 + count_1 == 0:
            raise ValueError(
                f"Ground truth for patient {patient} has no valid labels."
            )

        if count_1 == 0:
            excluded.append(patient)
            continue

        ratios[patient] = count_1 / (count_0 + count_1)

    ratios = pd.Series(ratios)
    all_patients = list(ratios.index)

    if len(excluded) > 0:
        print(
            f"[INFO] Excluding {len(excluded)} patients with zero lesions: "
            f"{excluded}"
        )

    # ------------------------------------------------------------
    # Quartile assignment
    # ------------------------------------------------------------
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

    total_n = len(all_patients)
    total_val = int(round(total_n * validation_fraction))
    total_ts = int(round(total_n * test_fraction))

    assigned_val = 0
    assigned_ts = 0
    remaining_pool = []

    # ------------------------------------------------------------
    # Proportional allocation per quartile
    # ------------------------------------------------------------
    for q in sorted(quartiles.unique()):
        group = list(quartiles[quartiles == q].index)
        n = len(group)
        if n == 0:
            continue

        prop = n / total_n
        n_val_q = int(round(total_val * prop))
        n_ts_q = int(round(total_ts * prop))

        n_val_q = min(n_val_q, total_val - assigned_val)
        n_ts_q = min(n_ts_q, total_ts - assigned_ts)

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

    # ------------------------------------------------------------
    # Fix rounding leftovers
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Enforce exact totals
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Plot distribution
    # ------------------------------------------------------------
    df_plot = pd.DataFrame(
        [{"Set": "Train", "Ratio": ratios[p]} for p in tr_patients]
        + [{"Set": "Validation", "Ratio": ratios[p]} for p in val_patients]
        + [{"Set": "Test", "Ratio": ratios[p]} for p in ts_patients]
    )

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df_plot,
        x="Set",
        y="Ratio",
        showcaps=False,
        showfliers=False,
        boxprops={"facecolor": "none"},
        whiskerprops={"linewidth": 1},
        medianprops={"linewidth": 1},
    )
    sns.stripplot(data=df_plot, x="Set", y="Ratio", jitter=0.1, size=8)
    plt.grid(axis="y", alpha=0.3)

    if title is not None:
        plt.title(title)

    plt.ylabel("positive / total voxels ratio")

    if show:
        plt.show()

    plt.savefig(os.path.join(data_folder, "ratios.png"))
    plt.close()

    # ------------------------------------------------------------
    # Save patient lists
    # ------------------------------------------------------------
    if save_series:
        pd.Series(tr_patients).to_pickle(
            os.path.join(data_folder, "train_patients.pkl")
        )
        pd.Series(val_patients).to_pickle(
            os.path.join(data_folder, "validation_patients.pkl")
        )
        pd.Series(ts_patients).to_pickle(
            os.path.join(data_folder, "test_patients.pkl")
        )

    return tr_patients, val_patients, ts_patients

def gaussian_transform(image, mean, std, return_float=False,
                       normalized=False):
    """
    Apply a Gaussian weighting to voxel intensities.

    Each voxel intensity ``I`` is transformed as::

        G(I) = exp(-0.5 * ((I - mean) / std)**2) * I

    If ``return_float=True`` and ``normalized=True``, the result is
    additionally scaled by::

        1 / (std * sqrt(2π))

    Parameters
    ----------
    image : SimpleITK.Image
        Input image with non‑negative voxel intensities.
    mean : float
        Mean of the Gaussian distribution.
    std : float
        Standard deviation of the Gaussian.
    return_float : bool, optional
        If True, return a float image in [0, 1].
    normalized : bool, optional
        If True and ``return_float=True``, apply Gaussian normalization.

    Returns
    -------
    SimpleITK.Image
        Gaussian‑weighted image with preserved metadata.

    Raises
    ------
    ValueError
        If the input image contains negative voxel values.
    """
    arr = get_array_from_image(image)

    if np.any(arr < 0):
        raise ValueError("The input image contains negative values.")

    if return_float:
        norm = 1 / (std * np.sqrt(2 * np.pi)) if normalized else 1
        gaussian_arr = norm * np.exp(-0.5 * ((arr - mean) / std) ** 2)
        return get_image_from_array(
            gaussian_arr, image, cast_to_reference=False
        )

    gaussian_arr = np.exp(-0.5 * ((arr - mean) / std) ** 2) * arr
    return get_image_from_array(
        gaussian_arr, image, cast_to_reference=True
    )

def cliffs_delta(x, y):
    """
    Compute Cliff's delta effect size between two 1D samples.

    Cliff's delta measures how often values in ``x`` exceed values in
    ``y`` minus how often they are smaller, normalized by the total
    number of pairwise comparisons.

    The statistic ranges from -1 to +1:
    - +1  → all values in ``x`` > all values in ``y``
    -  0  → distributions largely overlap
    - -1 → all values in ``x`` < all values in ``y``

    Parameters
    ----------
    x : np.ndarray
        First 1D sample.
    y : np.ndarray
        Second 1D sample.

    Returns
    -------
    float
        Cliff's delta effect size.
    """
    diff = x[:, None] - y[None, :]
    return (np.sum(diff > 0) - np.sum(diff < 0)) / (x.size * y.size)
