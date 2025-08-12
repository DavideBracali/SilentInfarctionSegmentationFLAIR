#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-03 17:32:07

@author: david
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d

from SilentInfarctionSegmentationFLAIR.histograms import plot_histogram
from SilentInfarctionSegmentationFLAIR.histograms import gaussian_smooth_histogram
from SilentInfarctionSegmentationFLAIR.histograms import mode_and_rhwhm
from SilentInfarctionSegmentationFLAIR.segmentation import apply_threshold

def main(image, gm, gamma):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # compute gm histogram
    gm_hist = plot_histogram(gm, no_bkg=True, ax=ax)
    
    # smooth histogram with gaussian filter
    gm_smooth_hist = gaussian_smooth_histogram(gm_hist)

    # find mode and
    mode, rhwhm = mode_and_rhwhm(gm_smooth_hist)

    thr = mode + gamma * rhwhm
    thr_mask = apply_threshold(image, float(thr))

    ax.legend()
    ax.set_title(f"GM histogram and threshold with γ={gamma}")
    plt.tight_layout()
    plt.show()

    return thr_mask