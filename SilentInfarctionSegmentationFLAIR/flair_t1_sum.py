#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-18 12:10:40

@author: david
"""

import SimpleITK as sitk
import warnings
import numpy as np

from SilentInfarctionSegmentationFLAIR.utils import (get_array_from_image,
                                                     get_image_from_array,
                                                     normalize)
from SilentInfarctionSegmentationFLAIR.utils import gaussian_transform
from SilentInfarctionSegmentationFLAIR.histograms import plot_multiple_histograms

def main(flair, t1, alpha, beta, wm_mask, verbose=True):

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
        print(f"WM mean in the T1 image: {wm_mean:.2f}")
        print(f"WM std in the T1 image: {wm_std:.2f}")

    # weighted sum of FLAIR and gaussian transformed T1
    t1_gauss = gaussian_transform(t1, mean=wm_mean, std=alpha*wm_std)
    t1_gauss_arr = beta * get_array_from_image(t1_gauss)
    t1_gauss_scaled = get_image_from_array(
        t1_gauss_arr, t1_gauss)
    image = flair + t1_gauss_scaled

    # cast to 8-bit unsigned int
    image = normalize(image, 8)
    
    return image