#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-03 17:32:07

@author: david
"""

import matplotlib.pyplot as plt
import SimpleITK as sitk

from SilentInfarctionSegmentationFLAIR.histograms import plot_histogram
from SilentInfarctionSegmentationFLAIR.histograms import plot_multiple_histograms
from SilentInfarctionSegmentationFLAIR.histograms import gaussian_smooth_histogram
from SilentInfarctionSegmentationFLAIR.histograms import mode_and_rhwhm
from SilentInfarctionSegmentationFLAIR.segmentation import apply_threshold
from SilentInfarctionSegmentationFLAIR.segmentation import evaluate_voxel_wise
from SilentInfarctionSegmentationFLAIR.refinement import evaluate_region_wise

def main(image, gm, gamma, show=True, verbose=True, save_path=None):

    # evaluation parameters
    metrics_rw = []
    metrics_vw = []

    # initialize figure
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = None

    # compute gm histogram
    gm_hist = plot_histogram(gm, no_bkg=True, bins='fd', show=show, ax=ax)
    
    # smooth histogram with gaussian filter
    gm_smooth_hist = gaussian_smooth_histogram(gm_hist, show=show)

    # find mode and right-side half width at half maximum
    mode, rhwhm = mode_and_rhwhm(gm_smooth_hist, show=show)

    # apply threshold
    thr = mode + gamma * rhwhm
    if verbose:
        print(f"Applying threshold at gray level {thr:.1f} (gamma = {gamma:.1f})")
    thr_mask = apply_threshold(image, float(thr), show=show)

    # plot additional details
    ax.legend()
    ax.set_title(f"GM histogram and threshold with γ={gamma}")
    plt.tight_layout()
    
    if save_path is not None:   plt.savefig(save_path)

    if show:    plt.show()

    return thr_mask