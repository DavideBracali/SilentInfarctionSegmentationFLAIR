#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-18 12:10:40

@author: david
"""

import SimpleITK as sitk
import warnings

from SilentInfarctionSegmentationFLAIR.refinement import connected_components
from SilentInfarctionSegmentationFLAIR.refinement import diameter_filter
from SilentInfarctionSegmentationFLAIR.refinement import pve_filter
from SilentInfarctionSegmentationFLAIR.refinement import evaluate_region_wise
from SilentInfarctionSegmentationFLAIR.segmentation import evaluate_voxel_wise

def main(thr_mask, min_diameter=None, pves=[],
         verbose=True):

    points = []

    ccs, n = connected_components(thr_mask)

    if min_diameter is not None:        
        if verbose:
            print(f"Applying diameter filter with a minimum diameter of {min_diameter} mm...")
        
        diameter_points, n_filtered, _ = diameter_filter(ccs, lower_thr=min_diameter)
        points.append(diameter_points)

        if verbose:
            print(f"{n_filtered} / {n} lesions have minimum diameter >= {min_diameter} mm")
    
    if len(pves) != 3:
        warnings.warn("'pves' must be a list containing respectively [pve_wm, pve_gm, pve_csf]. PVE filter will not be applied.")
    else:
        if verbose:
            print(f"Applying PVE filter...")        
        
        pve_points, n_filtered = pve_filter(ccs, n, pves)
        points.append(pve_points)

        if verbose:
            print(f"{n_filtered[0]} / {n} lesions are predominantly composed of WM\n"+
                  f"{n_filtered[1]} / {n} lesions are predominantly composed of GM\n"+
                  f"{n_filtered[2]} / {n} lesions are predominantly composed of CSF\n"+
                  f"{n_filtered[3]} / {n} lesions have null PVE effect for neither WM, GM and CSF")
       