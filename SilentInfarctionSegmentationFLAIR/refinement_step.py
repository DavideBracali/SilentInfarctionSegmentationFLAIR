#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-18 12:10:40

@author: david
"""

import SimpleITK as sitk
import warnings
import numpy as np

from SilentInfarctionSegmentationFLAIR.refinement import connected_components
from SilentInfarctionSegmentationFLAIR.refinement import diameter_filter
from SilentInfarctionSegmentationFLAIR.refinement import pve_filter
from SilentInfarctionSegmentationFLAIR.refinement import surrounding_filter
from SilentInfarctionSegmentationFLAIR.refinement import get_array_from_image
from SilentInfarctionSegmentationFLAIR.refinement import get_image_from_array



def main(thr_mask, min_diameter=None, pves=[], dilation_radius=None,
        min_points=3, verbose=True):

    points = []
    
    # compute connected components (lesions)
    ccs, n = connected_components(thr_mask)
    if verbose:
        print(f"Number of connected components (lesions) in the image: {n}")

    # minimum diameter filter
    if min_diameter is not None:        
        if verbose:
            print(f"Applying diameter filter with a minimum diameter of {min_diameter} mm...")
        
        diameter_points, n_filtered, _ = diameter_filter(ccs, lower_thr=min_diameter)
        points.append(diameter_points)

        if verbose:
            print(f"{n_filtered} / {n} lesions have minimum diameter >= {min_diameter} mm")
    
    # PVEs filters (inside the lesion and surrounding it)
    if len(pves) != 3:
        warnings.warn("'pves' must be a list containing respectively [pve_wm, pve_gm, pve_csf]."+
                      "PVE filter will not be applied.")
    else:
        if verbose:
            print(f"Applying PVE filter inside the lesions...")        
        
        pve_points, n_filtered, _ = pve_filter(ccs, n, pves)
        points.append(pve_points)

        if verbose:
            print(f"{n_filtered[0]} / {n} lesions are predominantly composed of WM\n"+
                  f"{n_filtered[1]} / {n} lesions are predominantly composed of GM\n"+
                  f"{n_filtered[2]} / {n} lesions are predominantly composed of CSF\n"+
                  f"{n_filtered[3]} / {n} lesions have null PVE effect for neither WM, GM and CSF")


        if dilation_radius is not None:    
            if verbose:
                print(f"Applying PVE filter inside the lesions...")

            surround_points, n_filtered, _ = surrounding_filter(ccs, n, pves, dilation_radius=dilation_radius)
            points.append(surround_points)

            if verbose:
                print(f"{n_filtered[0]} / {n} lesions are predominantly surrounded by WM\n"+
                    f"{n_filtered[1]} / {n} lesions are predominantly surrounded by GM\n"+
                    f"{n_filtered[2]} / {n} lesions are predominantly surrounded by CSF\n"+
                    f"{n_filtered[3]} / {n} lesions neighborhoods have null PVE effect for neither WM, GM and CSF")
        

    # only keep lesions with a minimum number of points
    lesion_idx = points[points >= min_points].index
    ccs_arr = get_array_from_image(ccs)
    lesion_mask = np.isin(ccs_arr, lesion_idx)
    ref_mask = get_image_from_array(lesion_mask, thr_mask)

    return ref_mask