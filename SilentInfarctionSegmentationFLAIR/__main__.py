#!/usr/bin/python
# -*- coding: utf-8 -*-

from __version__ import __version__
import argparse
import os
import pandas as pd
import yaml
import SimpleITK as sitk

from SilentInfarctionSegmentationFLAIR.segmentation import (get_mask_from_segmentation,
                                                            evaluate_voxel_wise)
from SilentInfarctionSegmentationFLAIR.utils import (orient_image,
                                                     resample_to_reference)
from SilentInfarctionSegmentationFLAIR import (flair_t1_sum,
                                               threshold,
                                               refinement_step)

__author__ = ['Davide Bracali']
__email__ = ['davide.bracali@studio.unibo.it']

def parse_args():

    description = ('SilentInfarctionSegmentationFLAIR - '
    '!!!!! aggiungere descrizione'
    )

    parser = argparse.ArgumentParser(
        prog='SilentInfarctionSegmentationFLAIR',
        argument_default=None,
        add_help=True,
        prefix_chars='-',
        allow_abbrev=True,
        exit_on_error=True,
        description=description,
        epilog=f'SilentInfarctionSegmentationFLAIR Python package v{__version__}'
    )

    # version
    parser.add_argument(
        '--version', '-v',
        dest='version',
        required=False,
        action='store_true',
        default=False,
        help='Get the current version installed',
    )

    parser.add_argument(
        '--patient-folder', '-p',
        type=str,
        required=True,
        help='Path to the patient folder containing all input images'
    )

    parser.add_argument(
        '--params-path',
        type=str,
        required=False,
        default="params.yaml",
        help='Path to the .yaml file containing segmentation parameters'
    )

    parser.add_argument(
        '--results-folder', '-o',
        type=str,
        required=False,
        default="results",
        help='Output directory where results will be saved'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Show intermediate results'
    )

    return parser.parse_args()


def main(patient_folder, params_path, results_folder, verbose, show):

    patient = os.path.basename(patient_folder)
    
    # load constants from yaml file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    gm_labels = config['labels']['gm']
    wm_labels = config['labels']['wm']
    keywords_to_remove = config['labels']['keywords_to_remove']

    flair_file = config['files']['flair']
    t1_file = config['files']['t1']
    segm_file = config['files']['segmentation']
    gm_pve_file = config['files']['gm_pve']
    wm_pve_file = config['files']['wm_pve']
    csf_pve_file = config['files']['csf_pve']
    gt_file = config['files']['gt']
    label_name_file = config['files']['label_name']

    # load images from patient folder
    flair = sitk.ReadImage(os.path.join(patient_folder, flair_file))
    flair = orient_image(flair, "RAS")
    t1 = sitk.ReadImage(os.path.join(patient_folder, t1_file))
    t1 = resample_to_reference(t1, flair, sitk.sitkLinear)

    segm = sitk.ReadImage(os.path.join(patient_folder, segm_file))
    segm = resample_to_reference(segm, flair, sitk.sitkNearestNeighbor)
    gm_mask = get_mask_from_segmentation(segm, gm_labels)
    wm_mask = get_mask_from_segmentation(segm, wm_labels)

    wm_pve = sitk.ReadImage(os.path.join(patient_folder, wm_pve_file))
    wm_pve = resample_to_reference(wm_pve, flair, sitk.sitkLinear)
    gm_pve = sitk.ReadImage(os.path.join(patient_folder, gm_pve_file))
    gm_pve = resample_to_reference(gm_pve, flair, sitk.sitkLinear)
    csf_pve = sitk.ReadImage(os.path.join(patient_folder, csf_pve_file))
    csf_pve = resample_to_reference(csf_pve, flair, sitk.sitkLinear)

    # ground truth (optional)    
    if os.path.isfile(os.path.join(patient_folder, gt_file)):
        gt = sitk.ReadImage(os.path.join(patient_folder, gt_file))
        gt = resample_to_reference(gt, flair, sitk.sitkNearestNeighbor)
        gt = sitk.Cast(gt, sitk.sitkUInt8)
    else:
        gt = None

    # load parameters
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # weighted sum of FLAIR and gaussian-transformed T1
    image = flair_t1_sum.main(flair, t1,
                                params['alpha'],
                                params['beta'],
                                wm_mask=wm_mask,
                                gm_mask=gm_mask,
                                gt=gt,  # ground truth is optional, if not provided
                                        # then tissue histograms will not be plotted
                                verbose=verbose,
                                show=show,
                                save_dir=os.path.join(results_folder, patient))
    # adaptive threshold
    thr_mask = threshold.main(image,
                                gm_mask=gm_mask,
                                gamma=params['gamma'],
                                verbose=verbose,
                                show=show,
                                save_dir=os.path.join(results_folder, patient))
    # refine segmentation
    ref_mask = refinement_step.main(thr_mask, image,
                                pves=[wm_pve, gm_pve, csf_pve],
                                segm=segm,
                                min_diameter=params['min_diameter'],
                                surround_dilation_radius=params['surround_dilation_radius'],
                                n_std=params['n_std'],
                                extend_dilation_radius=params['extend_dilation_radius'],
                                min_points=params['min_points'],
                                keywords_to_remove=keywords_to_remove,
                                label_name_file=label_name_file,
                                verbose=verbose,
                                show=show,
                                save_dir=os.path.join(results_folder, patient))

    return image, thr_mask, ref_mask, gt

if __name__ == '__main__':

    args = parse_args()

    if args.version:
        print(__version__)
        raise SystemExit(0)

    # segmentation
    image, thr_mask, ref_mask, gt = main(args.patient_folder,
                                        args.params_path,
                                        args.results_folder,
                                        args.verbose,
                                        args.show)

    # evaluate if ground truth is provided
    if gt is not None:
        
        print("Computing evaluation metrics AFTER THRESHOLDING...")
        thr_results = evaluate_voxel_wise(thr_mask, gt)
        print(f"  - True positives fraction: {thr_results['vw-TPF']:.3g}")
        print(f"  - False positives fraction: {thr_results['vw-FPF']:.3g}")
        print(f"  - DICE coefficient: {thr_results['vw-DSC']:.3g}")
        print(f"  - Mattheus correlation coefficient: {thr_results['vw-MCC']:.3g}")

        print("Computing evaluation metrics AFTER REFINEMENT STEP:")
        ref_results = evaluate_voxel_wise(ref_mask, gt)
        print(f"  - True positives fraction: {ref_results['vw-TPF']:.3g}")
        print(f"  - False positives fraction: {ref_results['vw-FPF']:.3g}")
        print(f"  - DICE coefficient: {ref_results['vw-DSC']:.3g}")
        print(f"  - Mattheus correlation coefficient: {ref_results['vw-MCC']:.3g}")