#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-03 17:32:07

@author: david
"""

import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


from SilentInfarctionSegmentationFLAIR.utils import plot_histogram


def main(image, gm):

    # compute gm histogram
    gm_hist = plot_histogram(gm, bins=50, no_bkg=True)
    counts, bins = gm_hist

    # smooth with spline fit
    bins_center = (bins[:-1] + bins[1:]) / 2
    gm_spline = UnivariateSpline(bins_center, counts, s=100000)
    gm_spline_hist = gm_spline(bins_center)
    plt.plot(bins_center, gm_spline_hist)
    plt.show()