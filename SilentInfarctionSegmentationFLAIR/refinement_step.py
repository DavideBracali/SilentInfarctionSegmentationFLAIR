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

def main(thr_mask, verbose=True, min_diameter=None,
          segm=None, labels_to_remove=[], labels_dict=None,
          keywords_to_remove=[], gt=None):

    # evaluations parameters after each refinement step
    metrics_rw = []
    metrics_vw = []

    # filter out small components
    if min_diameter is not None:        

        ccs, n = connected_components(thr_mask)
        ccs_filtered, n_filtered, _ = diameter_filter(ccs, lower_thr=min_diameter)
        thr_mask = thr_mask * (ccs_filtered != 0)

        if verbose:
            print(f"Filtered out components with diameter <= {min_diameter} mm")
            print(f"Before filtering: {n} connected components")
            print(f"After filtering: {n_filtered} connected components")

        if gt is not None:
            metrics_rw.append(evaluate_region_wise(thr_mask, gt))
            metrics_vw.append(evaluate_voxel_wise(thr_mask, gt))

            if verbose:
                print(f"Region-wise DICE coefficient after filtering out small components: {metrics_rw[-1]['DSC']:.4f}")
                print(f"Voxel-wise DICE coefficient after filtering out small components: {metrics_vw[-1]['DSC']:.4f}")


    # remove non-lesionable voxels
    if keywords_to_remove:

        segm_thr = segm * thr_mask      # add segmentation labels to mask
        segm_thr_filtered, removed = label_filter(segm_thr, labels_to_remove=labels_to_remove,
                                         keywords_to_remove=keywords_to_remove, labels_dict=labels_dict)
        thr_mask = segm_thr_filtered != 0   # back to binary

        print(f"Filtered out voxels with labels {list(removed.keys())}")

        if gt is not None:
            metrics_rw.append(evaluate_region_wise(thr_mask, gt))
            metrics_vw.append(evaluate_voxel_wise(thr_mask, gt))
            
            if verbose:
                print(f"Region-wise DICE coefficient after filtering out non-lesionable voxels: {metrics_rw[-1]['DSC']:.4f}")
                print(f"Voxel-wise DICE coefficient after filtering out non-lesionable voxels: {metrics_vw[-1]['DSC']:.4f}")

    if gt is None:
        return thr_mask
    else:
        return thr_mask, metrics_rw, metrics_vw