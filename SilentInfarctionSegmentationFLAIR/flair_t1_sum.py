#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-18 12:10:40

@author: david
"""

import SimpleITK as sitk
import warnings
import numpy as np
import os
import matplotlib.pyplot as plt

from SilentInfarctionSegmentationFLAIR.utils import (get_array_from_image,
                                                     get_image_from_array,
                                                     normalize)
from SilentInfarctionSegmentationFLAIR.utils import gaussian_transform
from SilentInfarctionSegmentationFLAIR.histograms import plot_multiple_histograms


def main(flair, t1, alpha, beta, gm_mask, wm_mask, gt, show=False, save_dir=None, verbose=True):

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
        print("Computing weighted sum of FLAIR and gaussian-transformed T1...")

    # weighted sum of FLAIR and gaussian transformed T1
    t1_gauss = gaussian_transform(t1, mean=wm_mean, std=alpha*wm_std)
    t1_gauss_arr = beta * get_array_from_image(t1_gauss)
    t1_gauss_scaled = get_image_from_array(
        t1_gauss_arr, t1_gauss)
    image = flair + t1_gauss_scaled

    # cast to 8-bit unsigned int
    image = normalize(image, 8)

    # save histograms
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plot_multiple_histograms([sitk.Mask(normalize(flair, 8), wm_mask),
                                sitk.Mask(normalize(flair, 8), gm_mask),
                                sitk.Mask(normalize(flair, 8), gt)],
                                normalize=True,
                                no_bkg=True,
                                bins=['fd', 'fd', 'fd'],
                                labels=["WM (white matter)", "GM (gray matter)", "lesions"],
                                title=f"Tissue histograms in FLAIR image",
                                show=False,
                                save_path=os.path.join(save_dir, "histogram_FLAIR.png"))
        plot_multiple_histograms([sitk.Mask(normalize(t1, 8), wm_mask),
                                sitk.Mask(normalize(t1, 8), gm_mask),
                                sitk.Mask(normalize(t1, 8), gt)],
                                normalize=True,
                                no_bkg=True,
                                bins=['fd', 'fd', 'fd'],
                                labels=["WM (white matter)", "GM (gray matter)", "lesions"],
                                title=f"Tissue histograms in T1 image",
                                show=False,
                                save_path=os.path.join(save_dir, "histogram_T1.png"))
        plot_multiple_histograms([sitk.Mask(image, wm_mask),
                                sitk.Mask(image, gm_mask),
                                sitk.Mask(image, gt)],
                                normalize=True,
                                no_bkg=True,
                                bins=['fd', 'fd', 'fd'],
                                labels=["WM (white matter)", "GM (gray matter)", "lesions"],
                                title=f"Tissue histograms in FLAIR and T1 integrated image\n(α={alpha}, β={beta})",
                                show=False,
                                save_path=os.path.join(save_dir, "histogram_image.png"))

    
    return image