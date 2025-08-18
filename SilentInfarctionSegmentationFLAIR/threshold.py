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
from SilentInfarctionSegmentationFLAIR.segmentation import evaluate

def main(image, gm, gamma, gt=None, show=True):

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = None

    # compute gm histogram
    if gt is not None:
        lesions = sitk.Mask(image, gt)
        hists = plot_multiple_histograms([gm, lesions], no_bkg=True, show=show, ax=ax,
                                         labels=["Gray matter","Lesions"], normalize=True,
                                         bins=[None,'auto'])
        gm_hist = hists[0]
    else:
        gm_hist = plot_histogram(gm, no_bkg=True, show=show, ax=ax)
    
    # smooth histogram with gaussian filter
    gm_smooth_hist = gaussian_smooth_histogram(gm_hist, show=show)

    # find mode and
    mode, rhwhm = mode_and_rhwhm(gm_smooth_hist, show=show)

    thr = mode + gamma * rhwhm
    thr_mask = apply_threshold(image, float(thr), show=show)


    if show:
        ax.legend()
        ax.set_title(f"GM histogram and threshold with γ={gamma}")
        plt.tight_layout()
        plt.show()

    # evaluate
    if gt is not None:
        _, dice, _ = evaluate(thr_mask, gt)
        print(f"DICE coefficient after thresholding: {dice:.4f}")


    return thr_mask