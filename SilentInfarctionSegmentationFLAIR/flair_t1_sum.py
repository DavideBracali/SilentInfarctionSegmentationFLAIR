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

from SilentInfarctionSegmentationFLAIR.utils import (get_array_from_image,
                                                     get_image_from_array,
                                                     normalize,
                                                     orient_image,
                                                     resample_to_reference,
                                                     gaussian_transform,
                                                     plot_image)
from SilentInfarctionSegmentationFLAIR.segmentation import get_mask_from_segmentation
from SilentInfarctionSegmentationFLAIR.histograms import plot_multiple_histograms

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
        "Combines FLAIR and T1 images using a gaussian-transformed T1 weighted sum. "
        "Optionally plots tissue histograms for WM, GM and lesions."
    )

    parser = argparse.ArgumentParser(description=description)

    _ = parser.add_argument('--flair',
                            dest='flair',
                            action='store',
                            type=str,
                            required=True,
                            help='Path to FLAIR image')

    _ = parser.add_argument('--t1',
                            dest='t1',
                            action='store',
                            type=str,
                            required=True,
                            help='Path to T1 image')

    _ = parser.add_argument('--segm',
                            dest='segm',
                            action='store',
                            type=str,
                            required=True,
                            help='Path to segmentation image')

    _ = parser.add_argument('--gt',
                            dest='gt',
                            action='store',
                            type=str,
                            required=False,
                            default=None,
                            help='Path to ground-truth lesion mask')

    _ = parser.add_argument('--alpha',
                            dest='alpha',
                            action='store',
                            type=float,
                            required=True,
                            help='Alpha scaling factor for T1 gaussian std')

    _ = parser.add_argument('--beta',
                            dest='beta',
                            action='store',
                            type=float,
                            required=True,
                            help='Beta weight for gaussian-transformed T1')

    _ = parser.add_argument('--save_dir',
                            dest='save_dir',
                            action='store',
                            type=str,
                            required=False,
                            default=None,
                            help='Directory where histograms and output image will be saved')

    _ = parser.add_argument('--show',
                            dest='show',
                            action='store_true',
                            help='Show histograms interactively')

    _ = parser.add_argument('--no_verbose',
                            dest='verbose',
                            action='store_false',
                            help='Disable verbose output')

    args = parser.parse_args()
    return args


def main(flair, t1, alpha, beta, gm_mask, wm_mask, gt=None,
         show=False, save_dir=None, verbose=True):

    # normalize FLAIR and T1 between 0 and 1 and cast to float32
    flair = normalize(flair, 1)
    t1 = normalize(t1, 1)

    # extract WM statistics from T1
    wm_t1 = sitk.Mask(t1, wm_mask)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(wm_t1)
    wm_mean = stats.GetMean()
    wm_std = stats.GetSigma()

    if verbose:
        print("Computing weighted sum of FLAIR and gaussian-transformed T1 "\
              f"(alpha = {alpha}, beta = {beta})...")

    # weighted sum of FLAIR and gaussian transformed T1
    t1_gauss = gaussian_transform(t1, mean=wm_mean, std=alpha*wm_std)
    t1_gauss_arr = beta * get_array_from_image(t1_gauss)
    t1_gauss_scaled = get_image_from_array(
        t1_gauss_arr, t1_gauss)
    image = flair + t1_gauss_scaled

    # cast to 8-bit unsigned int
    image = normalize(image, 8)

    # save histograms
    if (save_dir is not None or show) and gt is not None:
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            sitk.WriteImage(image, os.path.join(save_dir, "image.nii"))

        flair = normalize(flair, 8)
        t1 = normalize(t1, 8)

        plot_multiple_histograms([sitk.Mask(flair, wm_mask),
                                sitk.Mask(flair, gm_mask),
                                sitk.Mask(flair, gt)],
                                normalize=True,
                                no_bkg=True,
                                bins=['fd', 'fd', 'fd'],
                                labels=["WM (white matter)", "GM (gray matter)", "lesions"],
                                title=f"Tissue histograms in FLAIR image",
                                show=show,
                                save_path=os.path.join(save_dir, "histogram_FLAIR.png")
                                    if save_dir else None)
        plot_multiple_histograms([sitk.Mask(t1, wm_mask),
                                sitk.Mask(t1, gm_mask),
                                sitk.Mask(t1, gt)],
                                normalize=True,
                                no_bkg=True,
                                bins=['fd', 'fd', 'fd'],
                                labels=["WM (white matter)", "GM (gray matter)", "lesions"],
                                title=f"Tissue histograms in T1 image",
                                show=show,
                                save_path=os.path.join(save_dir, "histogram_T1.png")
                                    if save_dir else None)
        plot_multiple_histograms([sitk.Mask(image, wm_mask),
                                sitk.Mask(image, gm_mask),
                                sitk.Mask(image, gt)],
                                normalize=True,
                                no_bkg=True,
                                bins=['fd', 'fd', 'fd'],
                                labels=["WM (white matter)", "GM (gray matter)", "lesions"],
                                title="Tissue histograms in FLAIR and T1 integrated image\n"\
                                    f"(α={alpha}, β={beta})",
                                show=show,
                                save_path=os.path.join(save_dir, "histogram_image.png")
                                    if save_dir else None)
        _ = plot_image(image,
                        title="Weighted sum of FLAIR and gaussian-transformed T1\n"\
                              f"(α = {alpha}, β = {beta})",
                        show=show,
                        save_path=os.path.join(save_dir,"image.png")
                            if save_dir else None)

    return image


if __name__ == "__main__":

    args = parse_args()

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