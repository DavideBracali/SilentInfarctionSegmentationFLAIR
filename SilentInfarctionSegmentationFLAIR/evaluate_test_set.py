#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-12-26 13:21:28

@author: david
"""

import time
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")

from SilentInfarctionSegmentationFLAIR.__main__ import main as process_patient
from SilentInfarctionSegmentationFLAIR.segmentation import evaluate_voxel_wise
from SilentInfarctionSegmentationFLAIR.utils import get_paths_df
from SilentInfarctionSegmentationFLAIR.refinement import connected_components


def parse_args():
    parser = argparse.ArgumentParser(description=
                "Computes evaluation metrics for a test set. "\
                "Plots number of lesions and lesion load distributions.")

    _ = parser.add_argument(
        '--data_folder', '-i',
        type=str,
        required=True,
        help='Path to the folder containing all input images'
    )
    _ = parser.add_argument(
        '--params_path',
        type=str,
        default="params.yaml",
        help='Path to the .yaml file containing segmentation parameters'
    )
    _ = parser.add_argument(
        '--results_folder', '-o',
        type=str,
        default="evaluate_test_set",
        help='Output directory where boxplots will be saved'
    )
    _ = parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    _ = parser.add_argument(
        '--show',
        action='store_true',
        help='Show evaluation boxplots'
    )

    return parser.parse_args()


def main(data_folder, params_path, results_folder, verbose, show):
    """
    Processes a set of patients to evaluate the performance of FLAIR 
    silent lesion segmentation.

    For each patient in the data folder:
    
    ----
    - Runs the main segmentation function of the package.
    - Computes voxel-wise metrics (Sensitivity, Precision, DICE).
    - Computes the number of lesions and total lesion volume.
    - Computes volume similarity between ground truth and segmentation.
    - Generates summary boxplots for metrics, lesion count, total volume, 
      and volume similarity, saving them to the results folder.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing input patient images (.nii).
    params_path : str
        Path to the .yaml file containing segmentation parameters.
    results_folder : str
        Path to the folder where results and boxplots will be saved.
    verbose : bool
        If True, prints detailed log messages during processing.
    show : bool
        If True, displays the generated boxplots.

    Returns
    -------
    None
        All results are saved to files; nothing is returned.

    Notes
    -----
    - If present, a 'test_patients.pkl' file is used to process only the test set patients.
      Otherwise, all patients in the folder are processed.
    - Uses SimpleITK for volume calculations and Seaborn/Matplotlib for visualization.
    - Requires the functions `process_patient`, `evaluate_voxel_wise`, `get_paths_df`,
      and `connected_components` to be importable from the SilentInfarctionSegmentationFLAIR package.
    """

    eval_bp = []
    n_bp = []
    v_bp = []
    vs_bp = []
    thr_metrics = []
    ref_metrics = []
    thr_volsim = []
    ref_volsim = []


    # organize file paths
    paths_df = get_paths_df(data_folder, extensions=".nii")
    os.makedirs(results_folder, exist_ok=True)

    # load train-val-test split if available
    test_split_path = os.path.join(data_folder, "test_patients.pkl")
    if os.path.isfile(test_split_path):
        test_patients = list(pd.read_pickle(test_split_path))
        print(f"Found {len(test_patients)} test patients "\
              "in existing train-validation-test split.")
    else:   # process al files in the folder
        test_patients = [os.path.basename(root) for root in paths_df.index]
        print("No train-validation-test split found. "\
              f"Processing all {len(test_patients)} patients.")

    # iterate over patients
    counts = 0
    for patient_folder in paths_df.index:
        patient = os.path.basename(patient_folder)

        if patient not in test_patients:
            continue
        else:
            counts+=1

        if verbose:
            print(f"\nProcessing patient {patient} "\
                  f"({counts}/{len(test_patients)})...\n")

        # process
        image, thr_mask, ref_mask, gt = process_patient(
            patient_folder,
            params_path,
            results_folder,
            verbose=False,
            show=False
        )
    
        # evaluate
        if gt is None:
            print(f"WARNING!!! No ground truth found for patient {patient}")
        else:
            if verbose:
                print("Computing evaluation metrics AFTER THRESHOLDING...")
            
            thr_results = evaluate_voxel_wise(thr_mask, gt)
            thr_metrics.append(pd.DataFrame(thr_results, index=[patient]))
            
            if verbose:
                print(f"  - True positives fraction: "
                    f"{thr_results['vw-TPF']:.3g}")
                print(f"  - False positives fraction: "
                    f"{thr_results['vw-FPF']:.3g}")
                print(f"  - Precision: "
                    f"{thr_results['vw-PPV']:.3g}")
                print(f"  - DICE coefficient: "
                    f"{thr_results['vw-DSC']:.3g}")
                print("Computing evaluation metrics AFTER REFINEMENT STEP:")
            
            ref_results = evaluate_voxel_wise(ref_mask, gt)
            ref_metrics.append(pd.DataFrame(ref_results, index=[patient]))

            if verbose:
                print(f"  - True positives fraction: "
                    f"{ref_results['vw-TPF']:.3g}")
                print(f"  - False positives fraction: "
                    f"{ref_results['vw-FPF']:.3g}")
                print(f"  - Precision: "
                    f"{ref_results['vw-PPV']:.3g}")
                print(f"  - DICE coefficient: "
                    f"{ref_results['vw-DSC']:.3g}")

        # preparing boxplot data
        n_bp.append(pd.DataFrame({
            "patient": [patient] * 2,
            "metric": [
                "Number of lesions",
                "Number of lesions"
            ],
            "value": [
                connected_components(gt)[1],
                connected_components(ref_mask)[1],
            ],
            "type": [
                "Ground truth",
                "Segmented",
            ]
        }))

        spacing = image.GetSpacing()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]

        stats = sitk.StatisticsImageFilter()
        stats.Execute(gt)
        v_gt = stats.GetSum()*voxel_volume

        stats = sitk.StatisticsImageFilter()
        stats.Execute(thr_mask)
        v_thr = stats.GetSum()*voxel_volume

        stats = sitk.StatisticsImageFilter()
        stats.Execute(ref_mask)
        v_ref = stats.GetSum()*voxel_volume

        v_bp.append(pd.DataFrame({
            "patient": [patient] * 2,
            "metric": [
                "Lesion volume (mm³)",
                "Lesion volume (mm³)"
            ],
            "value": [
                v_gt,
                v_ref

            ],
            "type": [
                "Ground truth",
                "Segmented",
                ]
        }))

        thr_vs = 1 - np.abs(v_thr - v_gt)/(v_thr + v_gt)
        thr_volsim.append(thr_vs)
        ref_vs = 1 - np.abs(v_ref - v_gt)/(v_ref + v_gt)
        ref_volsim.append(ref_vs)

        vs_bp.append(pd.DataFrame({
            "patient": [patient] * 2,
            "metric": [
                "Volume similarity",
                "Volume similarity"
            ],
            "value": [
                thr_vs,
                ref_vs
            ],
            "type": [
                "After threshold",
                "After refinement step",
                ]
        }))

        eval_bp.append(pd.DataFrame({
                "patient": [patient] * 3,
                "metric": ["Sensitivity",
                           "Precision",
                           "DICE"],
                "value": [
                    thr_results["vw-TPF"],
                    thr_results["vw-PPV"],
                    thr_results["vw-DSC"]
                ],
                "type": ["After threshold"] * 3
        }))

        eval_bp.append(pd.DataFrame({
                "patient": [patient] * 3,
                "metric": ["Sensitivity",
                           "Precision",
                           "DICE"],
                "value": [
                    ref_results["vw-TPF"],
                    ref_results["vw-PPV"],
                    ref_results["vw-DSC"]
                ],
                "type": ["After refinement step"] * 3
            }))


    # boxplots
    eval_bp = pd.concat(eval_bp, ignore_index=True)
    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=eval_bp,
        x="metric",
        y="value",
        hue="type",
        palette=["#1f77b4", "#ff7f0e"]
    )
    plt.title("Evaluation metrics distribution on test set")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_folder, "evaluation_metrics.png"), dpi=200)
    if show:
        plt.show()
    plt.close()

    n_bp = pd.concat(n_bp, ignore_index=True)
    plt.figure(figsize=(5, 4))
    sns.boxplot(
        data=n_bp,
        x="metric",
        y="value",
        hue="type",
        palette=["#3db41f", "#ff7f0e"]
    )
    plt.title("Number of lesions distribution on test set")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_folder, "number_of_lesions.png"), dpi=200)
    if show:
        plt.show()
    plt.close()

    v_bp = pd.concat(v_bp, ignore_index=True)
    plt.figure(figsize=(5, 4))
    sns.boxplot(
        data=v_bp,
        x="metric",
        y="value",
        hue="type",
        palette=["#3db41f", "#ff7f0e"]
    )
    plt.title("Total lesion load distribution on test set")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_folder, "lesion_load.png"), dpi=200)
    if show:
        plt.show()
    plt.close()

    vs_bp = pd.concat(vs_bp, ignore_index=True)
    plt.figure(figsize=(5, 4))
    sns.boxplot(
        data=vs_bp,
        x="metric",
        y="value",
        hue="type",
        palette=["#1f77b4", "#ff7f0e"]
    )
    plt.title("Volume similarity distribution on test set")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_folder, "volume_similarity.png"), dpi=200)
    if show:
        plt.show()
    plt.close()


    # final summary
    thr_metrics = pd.concat(thr_metrics)
    ref_metrics = pd.concat(ref_metrics)

    print(
        "\nAverage SENSITIVITY on test set after threshold: "
        f"{np.mean(thr_metrics['vw-TPF']):.3g} ± "
        f"{np.std(thr_metrics['vw-TPF']):.3g}\n"
        f"MEDIAN = {np.median(thr_metrics['vw-TPF']):.3g}\n"
        f"IQR = [{np.quantile(thr_metrics['vw-TPF'], 0.25):.3g}, "
        f"{np.quantile(thr_metrics['vw-TPF'], 0.75):.3g}]"
    )

    print(
        "\nAverage SENSITIVITY on test set after refinement step: "
        f"{np.mean(ref_metrics['vw-TPF']):.3g} ± "
        f"{np.std(ref_metrics['vw-TPF']):.3g}\n"
        f"MEDIAN = {np.median(ref_metrics['vw-TPF']):.3g}\n"
        f"IQR = [{np.quantile(ref_metrics['vw-TPF'], 0.25):.3g}, "
        f"{np.quantile(ref_metrics['vw-TPF'], 0.75):.3g}]"
    )

    print(
        "\nAverage SPECIFICITY on test set after threshold: "
        f"{np.mean(1 - thr_metrics['vw-FPF']):.4f} ± "
        f"{np.std(1 - thr_metrics['vw-FPF']):.3g}\n"
        f"MEDIAN = {np.median(1 - thr_metrics['vw-FPF']):.3g}\n"
        f"IQR = [{np.quantile(1 - thr_metrics['vw-FPF'], 0.25):.3g}, "
        f"{np.quantile(1 - thr_metrics['vw-FPF'], 0.75):.3g}]"
    )

    print(
        "\nAverage SPECIFICITY on test set after refinement step: "
        f"{np.mean(1 - ref_metrics['vw-FPF']):.4f} ± "
        f"{np.std(1 - ref_metrics['vw-FPF']):.3g}\n"
        f"MEDIAN = {np.median(1 - ref_metrics['vw-FPF']):.3g}\n"
        f"IQR = [{np.quantile(1 - ref_metrics['vw-FPF'], 0.25):.3g}, "
        f"{np.quantile(1 - ref_metrics['vw-FPF'], 0.75):.3g}]"
    )

    print(
        "\nAverage DICE on test set after threshold: "
        f"{np.mean(thr_metrics['vw-DSC']):.3g} ± "
        f"{np.std(thr_metrics['vw-DSC']):.3g}\n"
        f"MEDIAN = {np.median(thr_metrics['vw-DSC']):.3g}\n"
        f"IQR = [{np.quantile(thr_metrics['vw-DSC'], 0.25):.3g}, "
        f"{np.quantile(thr_metrics['vw-DSC'], 0.75):.3g}]"
    )

    print(
        "\nAverage DICE on test set after refinement step: "
        f"{np.mean(ref_metrics['vw-DSC']):.3g} ± "
        f"{np.std(ref_metrics['vw-DSC']):.3g}\n"
        f"MEDIAN = {np.median(ref_metrics['vw-DSC']):.3g}\n"
        f"IQR = [{np.quantile(ref_metrics['vw-DSC'], 0.25):.3g}, "
        f"{np.quantile(ref_metrics['vw-DSC'], 0.75):.3g}]"
    )

    print(
        "\nAverage volume similarity on test set after threshold: "
        f"{np.mean(thr_volsim):.3g} ± "
        f"{np.std(thr_volsim):.3g}\n"
        f"MEDIAN = {np.median(thr_volsim):.3g}\n"
        f"IQR = [{np.quantile(thr_volsim, 0.25):.3g}, "
        f"{np.quantile(thr_volsim, 0.75):.3g}]"
    )

    print(
        "\nAverage volume similarity on test set after refinement step: "
        f"{np.mean(ref_volsim):.3g} ± "
        f"{np.std(ref_volsim):.3g}\n"
        f"MEDIAN = {np.median(ref_volsim):.3g}\n"
        f"IQR = [{np.quantile(ref_volsim, 0.25):.3g}, "
        f"{np.quantile(ref_volsim, 0.75):.3g}]"
    )


if __name__ == "__main__":
    
    start_time = time.time()

    args = parse_args()
    main(args.data_folder, args.params_path, args.results_folder,
         args.verbose, args.show)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.1f} s")
