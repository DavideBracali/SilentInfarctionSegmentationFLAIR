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
import numpy as np
import yaml
import matplotlib


from SilentInfarctionSegmentationFLAIR.histograms import (plot_histogram,
                                                          gaussian_smooth_histogram,
                                                          mode_and_rhwhm)
from SilentInfarctionSegmentationFLAIR.segmentation import (apply_threshold,
                                                            get_mask_from_segmentation)
from SilentInfarctionSegmentationFLAIR.utils import (orient_image,
                                                     resample_to_reference,
                                                     plot_image,
                                                     get_array_from_image)

# load constants from yaml file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

gm_labels = config["labels"]["gm"]
wm_labels = config["labels"]["wm"]

flair_file = config["files"]["flair"]
t1_file = config["files"]["t1"]
segm_file = config["files"]["segmentation"]
gm_pve_file = config["files"]["gm_pve"]
wm_pve_file = config["files"]["wm_pve"]
csf_pve_file = config["files"]["csf_pve"]
gt_file = config["files"]["gt"]
label_name_file = config["files"]["label_name"]

def parse_args():
    description = (
        "Computes a GM-based threshold using histogram mode and RHWHM. "
        "Applies the threshold to the input image."
    )

    parser = argparse.ArgumentParser(description=description)

    _ = parser.add_argument('--image',
                            dest='image',
                            action='store',
                            type=str,
                            required=True,
                            help='Path to input image')

    _ = parser.add_argument('--segm',
                            dest='segm',
                            action='store',
                            type=str,
                            required=True,
                            help='Path to segmentation image')

    _ = parser.add_argument('--gamma',
                            dest='gamma',
                            action='store',
                            type=float,
                            required=True,
                            help='Gamma multiplier for RHWHM')

    _ = parser.add_argument('--no_show',
                            dest='show',
                            action='store_false',
                            help='Disable plot visualization')

    _ = parser.add_argument('--no_verbose',
                            dest='verbose',
                            action='store_false',
                            help='Disable verbose output')

    _ = parser.add_argument('--save_dir',
                            dest='save_dir',
                            action='store',
                            type=str,
                            required=False,
                            default=None,
                            help='Directory where the figure and the segmentation mask will be saved')

    args = parser.parse_args()
    return args


def main(image, gm_mask, gamma, show=False, verbose=True, save_dir=None):

    if show==False:
        matplotlib.use("Agg")
    # initialize figure
    fig, ax = plt.subplots()

    # compute gm histogram
    gm = sitk.Mask(image, gm_mask)
    gm_hist = plot_histogram(gm, no_bkg=True, bins='fd', ax=ax, show=False)
    
    # smooth histogram with gaussian filter
    gm_smooth_hist = gaussian_smooth_histogram(gm_hist, ax=ax)

    # find mode and right-side half width at half maximum
    mode, rhwhm = mode_and_rhwhm(gm_smooth_hist, ax=ax)

    # apply threshold
    thr = mode + gamma * rhwhm
    if verbose:
        print(f"Applying threshold at gray level {thr:.1f} (gamma = {gamma:.1f})...")
    thr_mask = apply_threshold(image, float(thr), ax=ax)

    # plot additional details
    ax.legend()
    ax.set_title(f"GM histogram and threshold with γ={gamma}")
    plt.tight_layout()
    
    # save histogram and segmentation
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "thr.png"))
        sitk.WriteImage(thr_mask, os.path.join(save_dir, "thr_mask.nii"))

    if show:
        plt.show()
    plt.close()

    if show or save_dir is not None:
        _ = plot_image(image, mask=thr_mask,
                title=f"Threshold segmented mask (γ = {gamma})",
                show=show,
                save_path=os.path.join(save_dir,"thr_mask.png")
                    if save_dir else None)

    return thr_mask


if __name__ == "__main__":

    args = parse_args()

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