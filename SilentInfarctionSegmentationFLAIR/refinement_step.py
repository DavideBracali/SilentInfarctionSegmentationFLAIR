#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-18 12:10:40

@author: david
"""

import SimpleITK as sitk

from SilentInfarctionSegmentationFLAIR.refinement import connected_components
from SilentInfarctionSegmentationFLAIR.refinement import diameter_filter
from SilentInfarctionSegmentationFLAIR.refinement import label_filter
from SilentInfarctionSegmentationFLAIR.refinement import evaluate_region_wise
from SilentInfarctionSegmentationFLAIR.segmentation import evaluate_voxel_wise

def main(thr_mask, min_diameter=None,
         verbose=True):

    ccs_points = []

    if min_diameter is not None:        

        ccs, n = connected_components(thr_mask)
        diameter_points, n_filtered, _ = diameter_filter(ccs, lower_thr=min_diameter)
        ccs_points.append(diameter_points)

        if verbose:
            print(f"Filtered out components with diameter <= {min_diameter} mm")
            print(f"Before filtering: {n} connected components")
            print(f"After filtering: {n_filtered} connected components")

