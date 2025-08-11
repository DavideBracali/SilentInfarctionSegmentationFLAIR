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

from SilentInfarctionSegmentationFLAIR.utils import get_array_from_image
from SilentInfarctionSegmentationFLAIR.utils import plot_histogram
from SilentInfarctionSegmentationFLAIR.utils import gaussian_smooth_histogram

def main(image, gm):

    # compute gm histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    gm_hist = plot_histogram(gm, no_bkg=True, ax=ax,
                             title="GM gray level histogram")
    # smooth histogram
    smooth_hist = gaussian_smooth_histogram(gm_hist)
    
    # plot max
    """
    mode = bins_center[np.argmax(smooth_counts)]
    ax.axvline(mode, linestyle='--', color='red',
               linewidth=2,label="Most frequent gray level")
    
    # estimate right side fwhm
    hm = np.max(smooth_counts) / 2
    less_than_hm_gl = bins_center[smooth_counts < hm]
    right_less_than_hm_gl = less_than_hm_gl[less_than_hm_gl > mode]
    right_hm_gl = np.min(right_less_than_hm_gl)
    rfwhm = right_hm_gl - mode
    # !!!!!! COSA FACCIO SE CI SONO PIU' DI DUE INTERSEZIONI?????????????


    # plot right side fwhm
    ax.annotate(
        'Right side FWHM',
        xy=(mode, hm),  
        xytext=(right_hm_gl, hm),
        arrowprops=dict(
            arrowstyle='<->',
            color='red',
            lw=2,
            shrinkA=0, 
            shrinkB=0,
            ),
        )

    ax.legend()
    """
    