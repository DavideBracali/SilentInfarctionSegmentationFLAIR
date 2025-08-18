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
from SilentInfarctionSegmentationFLAIR.segmentation import evaluate

def main(thr_mask, dilation_kernel=None, min_diameter=None,
          segm=None, labels_to_remove=[], labels_dict=None,
          keywords_to_remove=[], gt=None):


    # dilation
    if dilation_kernel is not None:    

        thr_mask = sitk.BinaryDilate(thr_mask, kernelType=sitk.sitkBox,
                                     kernelRadius=dilation_kernel)
        
        print(f"Dilated mask with kernel {dilation_kernel}")

        if gt is not None:
            _, dice, _ = evaluate(thr_mask, gt)
            print(f"DICE coefficient after dilation: {dice:.4f}")

    # filter out small components
    if min_diameter is not None:        

        ccs, n = connected_components(thr_mask)
        ccs_filtered, n_filtered, _ = diameter_filter(ccs, lower_thr=min_diameter)
        thr_mask = thr_mask * (ccs_filtered != 0)

        print(f"Filtered out components with diameter <= {min_diameter} mm")
        print(f"Before filtering: {n} connected components")
        print(f"After filtering: {n_filtered} connected components")

        if gt is not None:
            _, dice, _ = evaluate(thr_mask, gt)
            print(f"DICE coefficient after filtering out small components: {dice:.4f}")

    # remove non-lesionable voxels
    if keywords_to_remove:

        segm_thr = segm * thr_mask      # add segmentation labels to mask
        segm_thr_filtered, removed = label_filter(segm_thr, labels_to_remove=labels_to_remove,
                                         keywords_to_remove=keywords_to_remove, labels_dict=labels_dict)
        thr_mask = segm_thr_filtered != 0   # back to binary

        print(f"Filtered out voxels with labels {list(removed.keys())}")


        if gt is not None:
            _, dice, _ = evaluate(thr_mask, gt)
            print(f"DICE coefficient after filtering out non-lesionable voxels: {dice:.4f}")

    return thr_mask