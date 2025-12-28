#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-18 12:10:40

@author: david
"""

import SimpleITK as sitk
import os
import argparse
import yaml
import time
import matplotlib

from SilentInfarctionSegmentationFLAIR.utils import (
    get_array_from_image,
    get_image_from_array,
    normalize,
    orient_image,
    resample_to_reference,
    gaussian_transform,
    plot_image
)
from SilentInfarctionSegmentationFLAIR.segmentation import get_mask_from_segmentation
from SilentInfarctionSegmentationFLAIR.histograms import plot_multiple_histograms

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(os.path.dirname(MODULE_DIR), "config.yaml")

def parse_args():
    description = (
        "Combines FLAIR and T1 images using a gaussian-transformed T1 weighted "
        "sum. Optionally plots tissue histograms for WM, GM and lesions."
    )

    parser = argparse.ArgumentParser(description=description)

    _ = parser.add_argument(
        '--flair', dest='flair', action='store', type=str, required=True,
        help='Path to FLAIR image'
    )

    _ = parser.add_argument(
        '--t1', dest='t1', action='store', type=str, required=True,
        help='Path to T1 image'
    )

    _ = parser.add_argument(
        '--segm', dest='segm', action='store', type=str, required=True,
        help='Path to segmentation image'
    )

    _ = parser.add_argument(
        '--gt', dest='gt', action='store', type=str, default=None,
        help='Path to ground-truth lesion mask'
    )

    _ = parser.add_argument(
        '--alpha', dest='alpha', action='store', type=float, required=True,
        help='Alpha scaling factor for T1 gaussian std'
    )

    _ = parser.add_argument(
        '--beta', dest='beta', action='store', type=float, required=True,
        help='Beta weight for gaussian-transformed T1'
    )

    _ = parser.add_argument(
        '--save_dir', dest='save_dir', action='store', type=str, default=None,
        help='Directory where histograms and output image will be saved'
    )

    _ = parser.add_argument(
        '--show', dest='show', action='store_true',
        help='Show histograms interactively'
    )

    _ = parser.add_argument(
        '--no_verbose', dest='verbose', action='store_false',
        help='Disable verbose output'
    )

    args = parser.parse_args()
    return args


def main(flair, t1, alpha, beta, gm_mask, wm_mask, gt=None,
         show=False, save_dir=None, verbose=True):
    """
    Combine FLAIR and T1 images using a weighted sum of FLAIR and 
    gaussian-transformed T1. Optionally, compute and plot tissue histograms.

    Parameters
    ----------
    flair : SimpleITK.Image
        Preloaded FLAIR image.
    t1 : SimpleITK.Image
        Preloaded T1 image.
    alpha : float
        Scaling factor for the standard deviation of the gaussian transform
        applied to T1.
    beta : float
        Weight applied to the gaussian-transformed T1 image.
    gm_mask : SimpleITK.Image
        Mask for gray matter (GM).
    wm_mask : SimpleITK.Image
        Mask for white matter (WM).
    gt : SimpleITK.Image, optional
        Ground-truth lesion mask, if available (default: None).
    show : bool, optional
        Whether to show histograms interactively (default: False).
    save_dir : str, optional
        Directory to save output images and histograms (default: None).
    verbose : bool, optional
        Print progress messages (default: True).

    Returns
    -------
    SimpleITK.Image
        Weighted sum of FLAIR and gaussian-transformed T1, normalized to 8-bit.
    """
    if show == False:
        matplotlib.use("Agg")

    # min-max normalization (float64)
    flair = normalize(flair, 0)
    t1 = normalize(t1, 0)

    # wm statistics
    wm_t1 = sitk.Mask(t1, wm_mask)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(wm_t1)
    wm_mean = stats.GetMean()
    wm_std = stats.GetSigma()

    if verbose:
        print(
            "Computing weighted sum of FLAIR and gaussian-transformed T1 "
            f"(alpha = {alpha}, beta = {beta})..."
        )

    # gaussian transform and weighted sum 
    t1_gauss = gaussian_transform(t1, mean=wm_mean, std=alpha*wm_std)
    t1_gauss_arr = beta * get_array_from_image(t1_gauss)
    t1_gauss_scaled = get_image_from_array(t1_gauss_arr, t1_gauss)
    image = flair + t1_gauss_scaled

    # min-max normalization (8-bit int)
    image = normalize(image, 8)

    # save and plot
    if (save_dir is not None or show) and gt is not None:
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            sitk.WriteImage(image, os.path.join(save_dir, "image.nii"))

        flair = normalize(flair, 8)
        t1 = normalize(t1, 8)

        plot_multiple_histograms(
            [sitk.Mask(flair, wm_mask), sitk.Mask(flair, gm_mask), sitk.Mask(flair, gt)],
            normalize=True, no_bkg=True, bins=['fd', 'fd', 'fd'],
            labels=["WM (white matter)", "GM (gray matter)", "lesions"],
            title="Tissue histograms in FLAIR image",
            show=show,
            save_path=os.path.join(save_dir, "histogram_FLAIR.png") if save_dir else None
        )
        plot_multiple_histograms(
            [sitk.Mask(t1, wm_mask), sitk.Mask(t1, gm_mask), sitk.Mask(t1, gt)],
            normalize=True, no_bkg=True, bins=['fd', 'fd', 'fd'],
            labels=["WM (white matter)", "GM (gray matter)", "lesions"],
            title="Tissue histograms in T1 image",
            show=show,
            save_path=os.path.join(save_dir, "histogram_T1.png") if save_dir else None
        )
        plot_multiple_histograms(
            [sitk.Mask(image, wm_mask), sitk.Mask(image, gm_mask), sitk.Mask(image, gt)],
            normalize=True, no_bkg=True, bins=['fd', 'fd', 'fd'],
            labels=["WM (white matter)", "GM (gray matter)", "lesions"],
            title=f"Tissue histograms in FLAIR and T1 integrated image\n(α={alpha}, β={beta})",
            show=show,
            save_path=os.path.join(save_dir, "histogram_image.png") if save_dir else None
        )
        _ = plot_image(
            image,
            title=f"Weighted sum of FLAIR and gaussian-transformed T1\n(α = {alpha}, β = {beta})",
            show=show,
            save_path=os.path.join(save_dir, "image.png") if save_dir else None
        )

    return image


if __name__ == "__main__":
    
    start_time = time.time()

    args = parse_args()

    # load constants
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    gm_labels = config["labels"]["gm"]
    wm_labels = config["labels"]["wm"]

    # load images segmentation and ground truth
    flair = sitk.ReadImage(args.flair)
    flair = orient_image(flair, "RAS")
    t1 = sitk.ReadImage(args.t1)
    t1 = resample_to_reference(t1, flair, sitk.sitkLinear)
    segm = sitk.ReadImage(args.segm, sitk.sitkUInt8)
    segm = resample_to_reference(segm, flair, sitk.sitkNearestNeighbor)
    gm_mask = get_mask_from_segmentation(segm, gm_labels)
    wm_mask = get_mask_from_segmentation(segm, wm_labels)
    gt = sitk.ReadImage(args.gt)
    gt = resample_to_reference(gt, flair)

    image = main(
        flair=flair,
        t1=t1,
        alpha=args.alpha,
        beta=args.beta,
        gm_mask=gm_mask,
        wm_mask=wm_mask,
        gt=gt,
        show=args.show,
        save_dir=args.save_dir,
        verbose=args.verbose
    )

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.1f} s")