#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-18 12:10:40

@author: david
"""

import SimpleITK as sitk
import warnings
import numpy as np
import os
import argparse
import yaml

from SilentInfarctionSegmentationFLAIR.refinement import (connected_components,
                                                          extend_lesions,
                                                          diameter_filter,
                                                          pve_filter,
                                                          surrounding_filter,
                                                          label_filter)
from SilentInfarctionSegmentationFLAIR.utils import (get_array_from_image,
                                                     get_image_from_array,
                                                     orient_image,
                                                     resample_to_reference,
                                                     label_names,
                                                     plot_image)

# load constants from yaml file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

gm_labels = config["labels"]["gm"]
wm_labels = config["labels"]["wm"]

flair_file = config["files"]["flair"]
t1_file = config["files"]["t1"]
segm_file = config["files"]["segmentation"]
gm_pve_file = config["files"]["gm_pve"]
wm_pve_file = config["files"]["wm_pve"]
csf_pve_file = config["files"]["csf_pve"]
gt_file = config["files"]["gt"]
label_name_file = config["files"]["label_name"]


def parse_args():
    description = (
        "Post-processes a threshold-based lesion mask applying connected components, "
        "geometric, PVE-based and anatomical filters."
    )

    parser = argparse.ArgumentParser(description=description)

    _ = parser.add_argument('--thr_mask',
                            dest='thr_mask',
                            action='store',
                            type=str,
                            required=True,
                            help='Path to thresholded lesion mask')

    _ = parser.add_argument('--image',
                            dest='image',
                            action='store',
                            type=str,
                            required=False,
                            default=None,
                            help='Reference image used for lesion extension')

    _ = parser.add_argument('--pve_wm',
                            dest='pve_wm',
                            action='store',
                            type=str,
                            required=False,
                            default=None,
                            help='Path to WM PVE map')

    _ = parser.add_argument('--pve_gm',
                            dest='pve_gm',
                            action='store',
                            type=str,
                            required=False,
                            default=None,
                            help='Path to GM PVE map')

    _ = parser.add_argument('--pve_csf',
                            dest='pve_csf',
                            action='store',
                            type=str,
                            required=False,
                            default=None,
                            help='Path to CSF PVE map')

    _ = parser.add_argument('--segm',
                            dest='segm',
                            action='store',
                            type=str,
                            required=False,
                            default=None,
                            help='Path to anatomical segmentation')

    _ = parser.add_argument('--min_diameter',
                            dest='min_diameter',
                            action='store',
                            type=float,
                            required=False,
                            default=None,
                            help='Minimum lesion diameter in mm')

    _ = parser.add_argument('--surround_dilation_radius',
                            dest='surround_dilation_radius',
                            action='store',
                            type=int,
                            required=False,
                            default=None,
                            help='Dilation radius for surrounding PVE filter')

    _ = parser.add_argument('--n_std',
                            dest='n_std',
                            action='store',
                            type=float,
                            required=False,
                            default=None,
                            help='Number of standard deviations for lesion extension')

    _ = parser.add_argument('--extend_dilation_radius',
                            dest='extend_dilation_radius',
                            action='store',
                            type=int,
                            required=False,
                            default=1,
                            help='Dilation radius for lesion extension')

    _ = parser.add_argument('--min_score',
                            dest='min_points',
                            action='store',
                            type=int,
                            required=False,
                            default=3,
                            help='Minimum score a lesion must satisfy')

    _ = parser.add_argument('--labels_to_remove',
                            dest='labels_to_remove',
                            action='store',
                            type=int,
                            nargs='*',
                            default=[],
                            help='List of segmentation labels to remove')

    _ = parser.add_argument('--keywords_to_remove',
                            dest='keywords_to_remove',
                            action='store',
                            type=str,
                            nargs='*',
                            default=[],
                            help='List of keywords of labels to remove')

    _ = parser.add_argument('--label_name_file',
                            dest='label_name_file',
                            action='store',
                            type=str,
                            required=False,
                            default=None,
                            help='Label-name LUT file')

    _ = parser.add_argument('--save_dir',
                            dest='save_dir',
                            action='store',
                            type=str,
                            required=False,
                            default=None,
                            help='Directory where output segmentation will be saved')

    _ = parser.add_argument('--no_verbose',
                            dest='verbose',
                            action='store_false',
                            help='Disable verbose output')

    args = parser.parse_args()
    return args


def main(thr_mask, image=None, pves=[], segm=None,
         min_diameter=None, surround_dilation_radius=None,
         n_std=None, extend_dilation_radius=1, min_points=3,
         labels_to_remove=[], keywords_to_remove=[], label_name_file=None,
         verbose=True, save_dir=None, show=False):

    points = []
    
    # compute connected components (lesions)
    ccs, n = connected_components(thr_mask)
    if verbose:
        print(f"Number of connected components (lesions) in the image: {n}")

    # extend lesions
    if n_std is not None and image is not None:
        if verbose:
            print("Extending lesions (dilated with a nearly-isotropic kernel "\
                    f"of radius {surround_dilation_radius} mm) with gray levels "\
                    f"{n_std} standard deviations away from mean...")
        
        ext_mask = extend_lesions(ccs, n, image, n_std=n_std,
                                  dilation_radius=extend_dilation_radius)
        if verbose:
            print(f"Before lesion extension: {n} connected components")

        ccs, n = connected_components(ext_mask)
        if verbose:
            print(f"After lesion extension: {n} connected components")
        
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
                  f"{n_filtered[3]} / {n} lesions have null PVE effect for neither\n"+
                  "WM, GM and CSF")


        if surround_dilation_radius is not None:    
            if verbose:
                print("Applying PVE filter around the lesions"\
                      f"(dilated with a nearly-isotropic kernel of radius {extend_dilation_radius} mm)...")

            surround_points, n_filtered, _ = surrounding_filter(ccs, n, pves,
                                                                dilation_radius=surround_dilation_radius)
            points.append(surround_points)

            if verbose:
                print(f"{n_filtered[0]} / {n} lesions are predominantly surrounded by WM\n"+
                    f"{n_filtered[1]} / {n} lesions are predominantly surrounded by GM\n"+
                    f"{n_filtered[2]} / {n} lesions are predominantly surrounded by CSF\n"+
                    f"{n_filtered[3]} / {n} lesions neighborhoods have null PVE effect\n"+
                    "for neither WM, GM and CSF")

    # only keep lesions with a minimum number of points
    if points:
        combined_points = sum(points)  
        lesion_idx = combined_points[combined_points >= min_points].index
    else:
        lesion_idx = np.arange(1, n + 1)
    
    ccs_arr = get_array_from_image(ccs)
    lesion_mask = np.isin(ccs_arr, lesion_idx).astype(np.uint8)
    ref_mask = get_image_from_array(lesion_mask, thr_mask)

    if verbose:
        _, n_kept = connected_components(ref_mask)
        print(f"\n{n_kept} out of {n} lesions obtained a score of at least {min_points}")

    # remove voxels where it is anatomically impossible to have lesions
    if ((keywords_to_remove != [] and label_name_file is not None) or
        (labels_to_remove != [])) and segm is not None:
        if verbose:
            print(f"Applying labels filter...")
        labels_dict = label_names(label_name_file)
        segm_filt, removed = label_filter(segm,
                                          labels_to_remove=labels_to_remove,
                                          keywords_to_remove=keywords_to_remove,
                                          labels_dict=labels_dict)
        ref_mask = ref_mask & (segm_filt > 0)

        if verbose:
            for label, n_removed in removed.items():
                print(f"Removed {n_removed} voxels of label '{label}'")
        
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        sitk.WriteImage(ref_mask, os.path.join(save_dir, "segmentation.nii"))

    if show or save_dir is not None:
        _ = plot_image(image, mask=ref_mask,
                title=f"Refined segmented mask",
                show=show,
                save_path=os.path.join(save_dir,"segmentation.png")
                    if save_dir else None)

    return ref_mask


if __name__ == "__main__":

    args = parse_args()

    thr_mask = sitk.ReadImage(args.thr_mask, sitk.sitkUInt8)
    thr_mask = orient_image(thr_mask, "RAS")

    image = None
    if args.image is not None:
        image = sitk.ReadImage(args.image)
        image = resample_to_reference(image, thr_mask, sitk.sitkLinear)

    pves = []
    if all(p is not None for p in [args.pve_wm, args.pve_gm, args.pve_csf]):
        pves = [
            sitk.ReadImage(args.pve_wm),
            sitk.ReadImage(args.pve_gm),
            sitk.ReadImage(args.pve_csf)
        ]
    pves = [resample_to_reference(p, thr_mask, sitk.sitkLinear) for p in pves]

    segm = None
    if args.segm is not None:
        segm = sitk.ReadImage(args.segm, sitk.sitkUInt8)
        segm = resample_to_reference(segm, thr_mask, sitk.sitkNearestNeighbor)

    ref_mask = main(
        thr_mask=thr_mask,
        image=image,
        pves=pves,
        segm=segm,
        min_diameter=args.min_diameter,
        surround_dilation_radius=args.surround_dilation_radius,
        n_std=args.n_std,
        extend_dilation_radius=args.extend_dilation_radius,
        min_points=args.min_points,
        labels_to_remove=args.labels_to_remove,
        keywords_to_remove=args.keywords_to_remove,
        label_name_file=args.label_name_file,
        verbose=args.verbose,
        save_dir=args.save_dir
    )
