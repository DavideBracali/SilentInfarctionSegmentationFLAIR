#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-03 17:32:07

@author: david
"""

import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import argparse
import yaml
import matplotlib
import time
import pathlib

from SilentInfarctionSegmentationFLAIR.histograms import (
    plot_histogram,
    gaussian_smooth_histogram,
    mode_and_rhwhm
)
from SilentInfarctionSegmentationFLAIR.segmentation import (
    apply_threshold,
    get_mask_from_segmentation
)
from SilentInfarctionSegmentationFLAIR.utils import (
    orient_image,
    resample_to_reference,
    plot_image,
    get_package_path
)

CONFIG_PATH = get_package_path("config.yaml")

def parse_args():
    """
    Parse command-line arguments for GM-based thresholding using RHWHM.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: `image`, `segm`, `gamma`, `show`,
        `verbose`, `save_dir`.
    """
    description = (
        "Computes a GM-based threshold using histogram mode and RHWHM. "
        "Applies the threshold to the input image."
    )

    parser = argparse.ArgumentParser(description=description)

    _ = parser.add_argument(
        '--image',
        dest='image',
        action='store',
        type=str,
        required=True,
        help='Path to input image'
    )

    _ = parser.add_argument(
        '--segm',
        dest='segm',
        action='store',
        type=str,
        required=True,
        help='Path to segmentation image'
    )

    _ = parser.add_argument(
        '--gamma',
        dest='gamma',
        action='store',
        type=float,
        required=True,
        help='Gamma multiplier for RHWHM'
    )

    _ = parser.add_argument(
        '--show',
        dest='show',
        action='store_true',
        help='Enable plot visualization'
    )

    _ = parser.add_argument(
        '--verbose',
        dest='verbose',
        action='store_true',
        help='Enable verbose output'
    )

    _ = parser.add_argument(
        '--save_dir',
        dest='save_dir',
        action='store',
        type=str,
        required=False,
        default=None,
        help='Directory to save figure and segmentation mask'
    )

    args = parser.parse_args()
    return args


def main(image, gm_mask, gamma, show=False, verbose=True, save_dir=None):
    """
    Compute a gray-matter-based threshold using histogram mode and RHWHM,
    apply it to the input image, and optionally plot/save the results.

    Parameters
    ----------
    image : SimpleITK.Image
        Input image to be thresholded.
    gm_mask : SimpleITK.Image
        Gray matter mask used to compute the histogram.
    gamma : float
        Multiplier for the right-side half-width at half-maximum (RHWHM)
        to set the threshold.
    show : bool, optional
        Whether to display plots interactively (default: False).
    verbose : bool, optional
        Whether to print progress messages (default: True).
    save_dir : str, optional
        Directory to save histogram figure and thresholded mask (default: None).

    Returns
    -------
    thr_mask : SimpleITK.Image
        Thresholded binary mask of the input image.
    """
    if show == False:
        matplotlib.use("Agg")

    fig, ax = plt.subplots()

    # GM histogram
    gm = sitk.Mask(image, gm_mask)
    gm_hist = plot_histogram(gm, no_bkg=True, bins='fd', ax=ax, show=False)
    gm_smooth_hist = gaussian_smooth_histogram(gm_hist, ax=ax)
    mode, rhwhm = mode_and_rhwhm(gm_smooth_hist, ax=ax)

    # apply threshold
    thr = mode + gamma * rhwhm
    if verbose:
        print(f"Applying threshold at gray level {thr:.1f} "
              f"(gamma = {gamma:.1f})...")
    thr_mask = apply_threshold(image, float(thr), ax=ax)

    # plot
    ax.legend()
    ax.set_title(f"GM histogram and threshold with γ={gamma}")
    plt.tight_layout()
    
    # save and show
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "thr.png"))
        sitk.WriteImage(thr_mask, os.path.join(save_dir, "thr_mask.nii"))

    if show:
        plt.show()
    plt.close()

    if show or save_dir is not None:
        _ = plot_image(
            image,
            mask=thr_mask,
            title=f"Threshold segmented mask (γ = {gamma})",
            show=show,
            save_path=os.path.join(save_dir, "thr_mask.png")
            if save_dir else None
        )

    return thr_mask


if __name__ == "__main__":

    start_time = time.time()

    args = parse_args()

    # load constants
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    gm_labels = config["labels"]["gm"]

    # load image and segmentation
    image = sitk.ReadImage(args.image)
    image = orient_image(image, "RAS")
    segm = sitk.ReadImage(args.segm, sitk.sitkUInt8)
    segm = resample_to_reference(segm, image)
    gm_mask = get_mask_from_segmentation(segm, gm_labels)

    thr_mask = main(
        image=image,
        gm_mask=gm_mask,
        gamma=args.gamma,
        show=args.show,
        verbose=args.verbose,
        save_dir=args.save_dir
    )

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.1f} s")
