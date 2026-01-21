#!/usr/bin/python
# -*- coding: utf-8 -*-

from .__version__ import __version__
import argparse
import os
import yaml
import SimpleITK as sitk
import time

from SilentInfarctionSegmentationFLAIR.segmentation import (
    get_mask_from_segmentation,
    evaluate_voxel_wise
)
from SilentInfarctionSegmentationFLAIR.utils import (
    orient_image,
    resample_to_reference,
    get_paths_df,
    normalize,
    get_package_path
)
from SilentInfarctionSegmentationFLAIR import (
    flair_t1_sum,
    threshold,
    refinement_step
)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


__author__ = ['Davide Bracali']
__email__ = ['davide.bracali@studio.unibo.it']

CONFIG_PATH = get_package_path("config.yaml")

def parse_args():
    description = (
        "SilentInfarctionSegmentationFLAIR - "\
        "Pipeline for the segmentation of silent cerebral infarctions"
        "from MRI images based on FLAIR and T1 integration, "\
        "adaptive thresholding and refinement step."
    )

    parser = argparse.ArgumentParser(
        prog='SilentInfarctionSegmentationFLAIR',
        argument_default=None,
        add_help=True,
        prefix_chars='-',
        allow_abbrev=True,
        exit_on_error=True,
        description=description,
        epilog=f'SilentInfarctionSegmentationFLAIR \
            Python package v{__version__}'
    )

    parser.add_argument(
        '--version', '-v',
        dest='version',
        action='store_true',
        default=False,
        help='Get the current version installed'
    )
    parser.add_argument(
        '--data_folder', '-i',
        type=str,
        required=True,
        help='Path to the folder containing all input images'
    )
    parser.add_argument(
        '--params_path',
        type=str,
        default=get_package_path("params.yaml"),
        help='Path to the .yaml file containing segmentation parameters'
    )
    parser.add_argument(
        '--results_folder', '-o',
        type=str,
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
    """
    Run the full FLAIR+T1 segmentation pipeline for a single patient.

    This function performs:
    1. Loading and orienting FLAIR and T1 images.
    2. Loading the segmentation and generating GM/WM masks.
    3. Loading probabilistic tissue maps (PVE) and ground-truth lesion mask (if available).
    4. Computing the weighted sum of FLAIR and Gaussian-transformed T1.
    5. Thresholding and refinement of the lesion mask.
    6. Returns all intermediate and final masks along with the ground truth.

    Parameters
    ----------
    patient_folder : str
        Path to the folder containing the patient's images.
    params_path : str
        Path to the YAML file with segmentation parameters (alpha, beta, gamma, etc.).
    results_folder : str
        Folder where results (images, masks) will be saved.
    verbose : bool
        Whether to print progress messages.
    show : bool
        Whether to display intermediate plots interactively.

    Returns
    -------
    image : SimpleITK.Image
        Weighted sum of FLAIR and Gaussian-transformed T1.
    thr_mask : SimpleITK.Image
        Mask obtained after thresholding.
    ref_mask : SimpleITK.Image
        Mask obtained after the refinement step.
    gt : SimpleITK.Image or None
        Ground-truth lesion mask if available, otherwise None.
    """    
    patient = os.path.basename(patient_folder)
    print(f"\n\nProcessing {patient}...\n")

    # load constants
    with open(CONFIG_PATH, "r") as f:
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


    # load images, segmentations, pves and ground truth if provided
    flair = sitk.ReadImage(os.path.join(patient_folder, flair_file))
    flair = orient_image(flair, "RAS")

    if os.path.isfile(os.path.join(patient_folder, t1_file)):
        t1 = sitk.ReadImage(os.path.join(patient_folder, t1_file))
        t1 = resample_to_reference(t1, flair, sitk.sitkLinear)
    else:
        t1 = None

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

    gt_path = os.path.join(patient_folder, gt_file)
    if os.path.isfile(gt_path):
        gt = sitk.ReadImage(gt_path)
        gt = resample_to_reference(gt, flair, sitk.sitkNearestNeighbor)
        gt = sitk.Cast(gt, sitk.sitkUInt8)
    else:
        gt = None

    # load segmentation parameters
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # weighted FLAIR + gaussian-transformed T1
    if t1 is None:
        print("WARNING!!! No T1 image was found. "
              "FLAIR image will be processed directly")
        image = normalize(flair, 8)
    else:
        image = flair_t1_sum.main(
            flair,
            t1,
            params["alpha"],
            params["beta"],
            wm_mask=wm_mask,
            gm_mask=gm_mask,
            gt=gt,
            verbose=verbose,
            show=show,
            save_dir=os.path.join(results_folder, patient)
        )

    # initial thresholded segmentation
    thr_mask = threshold.main(
        image,
        gm_mask=gm_mask,
        gamma=params['gamma'],
        verbose=verbose,
        show=show,
        save_dir=os.path.join(results_folder, patient)
    )

    #refined segmentation
    ref_mask = refinement_step.main(
        thr_mask,
        image,
        pves=[wm_pve, gm_pve, csf_pve],
        segm=segm,
        min_diameter=params['min_diameter'],
        surround_dilation_radius=params['surround_dilation_radius'],
        n_std=params['n_std'],
        extend_dilation_radius=params['extend_dilation_radius'],
        min_points=params['min_points'],
        keywords_to_remove=keywords_to_remove,
        label_name_file=label_name_file,
        verbose=verbose, show=show,
        save_dir=os.path.join(results_folder, patient)
    )

    return image, thr_mask, ref_mask, gt

def cli():
    
    start_time = time.time()

    args = parse_args()

    if args.version:
        print(__version__)
        raise SystemExit(0)
    
    # iterate over patients
    paths_df = get_paths_df(args.data_folder, extensions=".nii")
    for patient_folder in paths_df.index:

        _, thr_mask, ref_mask, gt = main(
            patient_folder,
            args.params_path,
            args.results_folder,
            args.verbose,
            args.show
        )

        # print evaluation metrics if gt is provided
        if gt is not None:
            print("Computing evaluation metrics AFTER THRESHOLDING...")
            thr_results = evaluate_voxel_wise(thr_mask, gt)
            print(f"  - Sensitivity: "
                f"{thr_results['vw-TPF']:.3g}")
            print(f"  - Precision: "
                f"{thr_results['vw-PPV']:.3g}")
            print(f"  - DICE coefficient: "
                f"{thr_results['vw-DSC']:.3g}")

            print("Computing evaluation metrics AFTER REFINEMENT STEP:")
            ref_results = evaluate_voxel_wise(ref_mask, gt)
            print(f"  - Sensitivity: "
                f"{ref_results['vw-TPF']:.3g}")
            print(f"  - Specificity: "
                f"{ref_results['vw-PPV']:.3g}")
            print(f"  - DICE coefficient: "
                f"{ref_results['vw-DSC']:.3g}")

        print("Saved results in", os.path.join(
            "results", os.path.basename(patient_folder)))

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.1f} s")

    
if __name__ == '__main__': 

    cli()