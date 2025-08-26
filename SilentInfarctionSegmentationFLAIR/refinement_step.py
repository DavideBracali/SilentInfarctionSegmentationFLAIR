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

def main(thr_mask, closure_kernel=None, min_diameter=None,
          segm=None, labels_to_remove=[], labels_dict=None,
          keywords_to_remove=[], verbose=True):

    # binary masks after each refinement step
    results = []

    # closure
    if closure_kernel is not None:        

        _, n = connected_components(thr_mask)
        thr_mask = sitk.BinaryMorphologicalClosing(thr_mask, kernelRadius=closure_kernel, kernelType=sitk.sitkBox)
        _, n_closed = connected_components(thr_mask)

        if verbose:
            print(f"Morphological closing with box kernel {closure_kernel}")
            print(f"Before closing: {n} connected components")
            print(f"After closing: {n_closed} connected components")

        results.append(thr_mask)

    # filter out small components
    if min_diameter is not None:        

        ccs, n = connected_components(thr_mask)
        ccs_filtered, n_filtered, _ = diameter_filter(ccs, lower_thr=min_diameter)
        thr_mask = thr_mask * (ccs_filtered != 0)

        if verbose:
            print(f"Filtered out components with diameter <= {min_diameter} mm")
            print(f"Before filtering: {n} connected components")
            print(f"After filtering: {n_filtered} connected components")

        results.append(thr_mask)


    # remove non-lesionable voxels
    if keywords_to_remove:

        segm_thr = segm * thr_mask      # add segmentation labels to mask
        segm_thr_filtered, removed = label_filter(segm_thr, labels_to_remove=labels_to_remove,
                                         keywords_to_remove=keywords_to_remove, labels_dict=labels_dict)
        thr_mask = segm_thr_filtered != 0   # back to binary

        results.append(thr_mask)

    return results