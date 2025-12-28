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
import matplotlib
matplotlib.use("Agg")

from SilentInfarctionSegmentationFLAIR.__main__ import main as process_patient
from SilentInfarctionSegmentationFLAIR.segmentation import evaluate_voxel_wise
from SilentInfarctionSegmentationFLAIR.utils import get_paths_df


def parse_args():
    parser = argparse.ArgumentParser(description="!!!!")

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
        help='Output directory where results will be saved'
    )
    _ = parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    _ = parser.add_argument(
        '--show',
        action='store_true',
        help='Show evaluation boxplot'
    )

    return parser.parse_args()


def main(data_folder, params_path, results_folder, verbose, show):
   
    to_plot = []
    thr_metrics = []
    ref_metrics = []

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
        _, thr_mask, ref_mask, gt = process_patient(
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
            to_plot.append(pd.DataFrame({
                "patient": [patient] * 3,
                "metric": ["Sensitivity (TPF)", "1 - specificity (FPF)", "DICE"],
                "value": [
                    thr_results["vw-TPF"],
                    thr_results["vw-FPF"],
                    thr_results["vw-DSC"]
                ],
                "type": ["After threshold"] * 3
            }))
            
            if verbose:
                print(f"  - True positives fraction: "
                    f"{thr_results['vw-TPF']:.3g}")
                print(f"  - False positives fraction: "
                    f"{thr_results['vw-FPF']:.3g}")
                print(f"  - DICE coefficient: "
                    f"{thr_results['vw-DSC']:.3g}")
            
            if verbose:
                print("Computing evaluation metrics AFTER REFINEMENT STEP:")
            
            ref_results = evaluate_voxel_wise(ref_mask, gt)
            ref_metrics.append(pd.DataFrame(ref_results, index=[patient]))
            to_plot.append(pd.DataFrame({
                "patient": [patient] * 3,
                "metric": ["Sensitivity (TPF)", "1 - specificity (FPF)", "DICE"],
                "value": [
                    ref_results["vw-TPF"],
                    ref_results["vw-FPF"],
                    ref_results["vw-DSC"]
                ],
                "type": ["After refinement step"] * 3
            }))

            if verbose:
                print(f"  - True positives fraction: "
                    f"{ref_results['vw-TPF']:.3g}")
                print(f"  - False positives fraction: "
                    f"{ref_results['vw-FPF']:.3g}")
                print(f"  - DICE coefficient: "
                    f"{ref_results['vw-DSC']:.3g}")
                

    # save metrics
    thr_metrics = pd.concat(thr_metrics)
    thr_metrics.to_csv(
        os.path.join(
            results_folder, "evaluation_metrics_after_threshold.csv")
    )

    ref_metrics = pd.concat(ref_metrics)
    ref_metrics.to_csv(
        os.path.join(
            results_folder, "evaluation_metrics_after_refinement.csv")
    )

    # boxplot
    to_plot = pd.concat(to_plot, ignore_index=True)
    plt.figure(figsize=(5, 4))
    sns.boxplot(
        data=to_plot,
        x="metric",
        y="value",
        hue="type",
        palette=["#1f77b4", "#ff7f0e"]
    )
    plt.title("Evaluation metrics distribution on test set")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()

    # save and show
    plt.savefig(
        os.path.join(results_folder, "validation_dice_boxplot.png"), dpi=200)
    if show:
        plt.show()
    plt.close()

    # final summary
    print(
        "Average DICE on test set after threshold: "
        f"{np.mean(thr_metrics["vw-DSC"]):.3g} ± "
        f"{np.std(thr_metrics["vw-DSC"]):.3g}\n"
        f"MEDIAN = {np.median(thr_metrics["vw-DSC"]):.3g}\n"
        f"IQR = [{np.quantile(thr_metrics["vw-DSC"], 0.25):.3g}, "
        f"{np.quantile(thr_metrics["vw-DSC"], 0.75):.3g}]"
    )

    print(
        "Average DICE on test set after refinement step: "
        f"{np.mean(ref_metrics["vw-DSC"]):.3g} ± "
        f"{np.std(ref_metrics["vw-DSC"]):.3g}\n"
        f"MEDIAN = {np.median(thr_metrics["vw-DSC"]):.3g}\n"
        f"IQR = [{np.quantile(ref_metrics["vw-DSC"], 0.25):.3g}, "
        f"{np.quantile(ref_metrics["vw-DSC"], 0.75):.3g}]"
    )




if __name__ == "__main__":
    
    start_time = time.time()

    args = parse_args()
    main(args.data_folder, args.params_path, args.results_folder,
         args.verbose, args.show)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.1f} s")
