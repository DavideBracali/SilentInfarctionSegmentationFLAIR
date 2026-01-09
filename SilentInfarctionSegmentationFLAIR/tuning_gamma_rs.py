#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tune gamma and refinement-step parameters.

This script optimizes refinement-step parameters for thresholded
FLAIR+T1 images. It supports image precomputation, threshold mask
generation for several `gamma` values, Bayesian optimization of the
refinement step, and validation on a held-out set.
"""

import SimpleITK as sitk
import os
import pandas as pd
import numpy as np
import argparse
import gc
import yaml
from bayes_opt import BayesianOptimization
import time
import multiprocessing as mp
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

from SilentInfarctionSegmentationFLAIR.utils import (
    orient_image,
    resample_to_reference,
    train_val_test_split,
)
from SilentInfarctionSegmentationFLAIR.segmentation import (
    get_mask_from_segmentation,
    evaluate_voxel_wise,
)
from SilentInfarctionSegmentationFLAIR import (
    flair_t1_sum,
    threshold,
    refinement_step,
)


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: `data_folder`, `results_folder`,
        `init_points`, `n_iter`, `n_cores`, `alpha`, `beta`, `gammas`.
    """

    description="Optimizes gamma and refinement parameters on a training set. "\
        "Maximises DICE coefficient after refinement step, then chooses gamma "\
        "maximizing DICE over validation set."
    
    parser = argparse.ArgumentParser(description=description)

    _ = parser.add_argument(
        "--data_folder",
        dest="data_folder",
        action="store",
        type=str,
        required=True,
        help="Path of the folder where data is located",
    )

    _ = parser.add_argument(
        "--results_folder",
        dest="results_folder",
        action="store",
        type=str,
        required=False,
        default="tuning_gamma_rs",
        help="Path of the folder where intermediate results are stored",
    )

    _ = parser.add_argument(
        "--init_points",
        dest="init_points",
        action="store",
        type=int,
        required=False,
        default=5,
        help="Number of initial points for random search",
    )

    _ = parser.add_argument(
        "--n_iter",
        dest="n_iter",
        action="store",
        type=int,
        required=False,
        default=20,
        help="Number of Bayesian optimization iterations",
    )

    _ = parser.add_argument(
        "--n_cores",
        dest="n_cores",
        action="store",
        type=int,
        required=False,
        default=1,
        help=(
            "Number of CPU cores to use during optimization (improves "
            "computational time but increases RAM and CPU usage)"
        ),
    )

    _ = parser.add_argument(
        "--alpha",
        dest="alpha",
        action="store",
        type=float,
        required=False,
        default=None,
        help="Segmentation parameter alpha",
    )

    _ = parser.add_argument(
        "--beta",
        dest="beta",
        action="store",
        type=float,
        required=False,
        default=None,
        help="Segmentation parameter beta",
    )

    _ = parser.add_argument(
        "--gammas",
        dest="gammas",
        nargs="+",
        type=float,
        required=False,
        default=[1.0, 2.0, 3.0, 4.0],
        help=(
            "List of gamma values to evaluate (e.g. --gammas 1 2 3 4)"
        ),
    )

    args = parser.parse_args()
    return args

pbounds = {
    "extend_dilation_radius": (1, 10),
    "n_std": (1, 10),
    "min_diameter": (1, 5),
    "surround_dilation_radius": (1, 10),
    "min_points": (1, 5)
}

def load_alpha_beta_from_yaml():
    yaml_path = "params_alpha_beta.yaml"
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(
            f"{yaml_path} not found but alpha/beta were not provided via CLI"
        )

    with open(yaml_path, "r") as f:
        params = yaml.safe_load(f)

    if params.get("alpha") is None or params.get("beta") is None:
        raise ValueError("params_alpha_beta.yaml must contain alpha and beta")

    return params["alpha"], params["beta"]

def load_subjects(data_folder, patients=None, paths_only=False):
    """
    Load imaging data for multiple subjects and organize into a dict.

    Parameters
    ----------
    data_folder : str
        Path to folder containing patient subfolders with .nii files.
    patients : list of str, optional
        List of patient IDs to load; if None, loads all patients.
    paths_only : bool, optional
        If True, only returns file paths without loading images. Default
        is False.

    Returns
    -------
    data : dict
        Keys are patient IDs and values are dicts with the following
        images (SimpleITK.Image): ``flair``, ``t1``, ``gt``, ``gm_mask``,
        ``wm_mask``, ``segm``, ``wm_pve``, ``gm_pve``, ``csf_pve``.
    paths_df : pandas.DataFrame
        DataFrame mapping each patient to their corresponding file
        paths.
    """
    # organize all images in a df
    paths_list = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".nii"):
                paths_list.append((root, file))

    paths_df = pd.DataFrame()
    for root, file in paths_list:
        patient = os.path.basename(root)
        paths_df.loc[patient, file] = os.path.join(root, file)

    # load only specified patients
    if patients is not None:
        paths_df = paths_df.loc[patients]

    # drop NaN
    dropped_patients = paths_df.index[paths_df.isna().any(axis=1)]
    if len(dropped_patients) > 0:
        msg = (
            "WARNING!!! The following patients will be removed because "
            f"one of the required images is missing: {list(dropped_patients)}"
        )
        print(msg)
    paths_df = paths_df.dropna(how="any")

    data = {}
    if paths_only == False:
        for patient in paths_df.index:
            flair = sitk.ReadImage(paths_df.loc[patient, flair_file])
            flair = orient_image(flair, "RAS")

            t1 = sitk.ReadImage(paths_df.loc[patient, t1_file])
            t1 = resample_to_reference(t1, flair,
                                       sitk.sitkLinear)

            gt = resample_to_reference(
                sitk.ReadImage(paths_df.loc[patient, gt_file]),
                flair,
                sitk.sitkNearestNeighbor,
            )
            gt = sitk.Cast(gt, sitk.sitkUInt8)

            segm = resample_to_reference(
                sitk.ReadImage(paths_df.loc[patient, segm_file]),
                flair,
                sitk.sitkNearestNeighbor,
            )
            gm_mask = get_mask_from_segmentation(segm, gm_labels)
            wm_mask = get_mask_from_segmentation(segm, wm_labels)

            wm_pve = sitk.ReadImage(paths_df.loc[patient, wm_pve_file])
            wm_pve = resample_to_reference(wm_pve, flair, sitk.sitkLinear)
            gm_pve = sitk.ReadImage(paths_df.loc[patient, gm_pve_file])
            gm_pve = resample_to_reference(gm_pve, flair, sitk.sitkLinear)
            csf_pve = sitk.ReadImage(paths_df.loc[patient, csf_pve_file])
            csf_pve = resample_to_reference(csf_pve, flair, sitk.sitkLinear)

            data[patient] = {
                "flair": flair,
                "t1": t1,
                "gt": gt,
                "gm_mask": gm_mask,
                "wm_mask": wm_mask,
                "segm": segm,
                "wm_pve": wm_pve,
                "gm_pve": gm_pve,
                "csf_pve": csf_pve,
            }

    return data, paths_df

def process_patient_images(args_tuple, results_folder):
    """
    Process a single patient's images to compute the combined FLAIR+T1.

    Parameters
    ----------
    args_tuple : tuple
        (data, patient, alpha, beta) where ``data`` contains patient
        imaging data and masks.
    results_folder : str
        Directory where output images are written.

    Returns
    -------
    None
        Saves the computed image to disk if not already present.
    """
    data, patient, alpha, beta = args_tuple

    flair = data["flair"]
    t1 = data["t1"]
    gt = data["gt"]
    gm_mask = data["gm_mask"]
    wm_mask = data["wm_mask"]

    image_path = os.path.join(results_folder, "images", f"{patient}.nii")
    if not os.path.isfile(image_path):
        image = flair_t1_sum.main(
            flair,
            t1,
            alpha,
            beta,
            wm_mask=wm_mask,
            gm_mask=gm_mask,
            gt=gt,
            verbose=False,
        )
        sitk.WriteImage(image, image_path)

def process_patient_thr(args_tuple, results_folder):
    """
    Process a single patient's image to compute threshold mask.

    Parameters
    ----------
    args_tuple : tuple
        (data, patient, gamma) where ``data`` contains patient images and
        masks.
    results_folder : str
        Directory where threshold masks are written.

    Returns
    -------
    None
        Saves threshold mask to disk.
    """
    data, patient, gamma = args_tuple

    thr_path = os.path.join(
        results_folder, f"thr_mask_g{gamma}", f"{patient}.nii"
    )
    if not os.path.isfile(thr_path):
        image = sitk.ReadImage(
            os.path.join(results_folder, "images", f"{patient}.nii")
        )
        gm_mask = resample_to_reference(
            data["gm_mask"], image, sitk.sitkNearestNeighbor
        )
        thr_mask = threshold.main(
            image, gm_mask, gamma=gamma, show=False, verbose=False
        )
        sitk.WriteImage(thr_mask, thr_path)
        
def process_patient_rs(args_tuple, results_folder):
    """
    Run refinement step on a patient's image and evaluate DICE.

    Parameters
    ----------
    args_tuple : tuple
        (data, patient, gamma, extend_dilation_radius, n_std,
         min_diameter, surround_dilation_radius, min_points)
        where ``data`` contains patient imaging, masks and PVEs.
    results_folder : str
        Directory where intermediate images and masks are stored.

    Returns
    -------
    float
        Voxel-wise DICE coefficient between refined mask and ground
        truth.
    """

    (
        data,
        patient,
        gamma,
        extend_dilation_radius,
        n_std,
        min_diameter,
        surround_dilation_radius,
        min_points,
    ) = args_tuple

    image = sitk.ReadImage(
        os.path.join(results_folder, "images", f"{patient}.nii")
    )
    thr_mask = sitk.ReadImage(
        os.path.join(results_folder, f"thr_mask_g{gamma}", f"{patient}.nii")
    )
    segm = resample_to_reference(data["segm"], image, sitk.sitkNearestNeighbor)
    thr_mask = resample_to_reference(
        thr_mask, image, sitk.sitkNearestNeighbor
    )
    pves = [data["wm_pve"], data["gm_pve"], data["csf_pve"]]
    pves = [resample_to_reference(p, image, sitk.sitkLinear) for p in pves]
    ref_mask = refinement_step.main(
        thr_mask,
        image,
        pves=pves,
        segm=segm,
        min_diameter=min_diameter,
        surround_dilation_radius=surround_dilation_radius,
        n_std=n_std,
        extend_dilation_radius=extend_dilation_radius,
        min_points=int(min_points),  # truncate to integer
        keywords_to_remove=keywords_to_remove,
        label_name_file=label_name_file,
        verbose=False,
        show=False,
    )

    dsc = evaluate_voxel_wise(ref_mask, data["gt"])["vw-DSC"]
    return dsc

def plot_validation_distributions(gammas, results_folder):

    all_thr = {}
    all_ref = {}

    for gamma in gammas:
        rs_dir = os.path.join(results_folder, f"rs_g{gamma}")

        thr_path = os.path.join(rs_dir, "val_dice_thr.pkl")
        ref_path = os.path.join(rs_dir, "val_dice.pkl")

        if os.path.isfile(thr_path):
            all_thr[gamma] = pd.read_pickle(thr_path)
        else:
            print(f"WARNING!!! Missing: {thr_path}")

        if os.path.isfile(ref_path):
            all_ref[gamma] = pd.read_pickle(ref_path)
        else:
            print(f"WARNING!!! Missing: {ref_path}")

    df_plot = []

    for gamma in gammas:
        if gamma in all_thr:
            df_plot.append(pd.DataFrame({
                "gamma": gamma,
                "DICE": all_thr[gamma],
                "type": "After threshold"
            }))

        if gamma in all_ref:
            df_plot.append(pd.DataFrame({
                "gamma": gamma,
                "DICE": all_ref[gamma],
                "type": "After refinement"
            }))

    df_plot = pd.concat(df_plot, ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_plot,
        x="gamma",
        y="DICE",
        hue="type",
        palette=["#1f77b4", "#ff7f0e"]
    )

    plt.title("DICE coefficient distribution over validation set", fontsize=20)
    plt.xlabel("γ", fontsize=14)
    plt.ylabel("DICE", fontsize=14)
    plt.legend(title="")
    plt.tight_layout()

    out_path = os.path.join(results_folder, "validation_dice_boxplot.png")
    plt.savefig(out_path, dpi=200)
    plt.close()



def process_patient_thr_dice(args_tuple, results_folder):
    """
    Compute voxel-wise DICE between threshold mask and ground truth.

    Parameters
    ----------
    args_tuple : tuple
        (data, patient, gamma) where ``data`` contains the ground truth
        mask under key ``gt``.
    results_folder : str
        Directory where threshold masks are stored.

    Returns
    -------
    float
        Voxel-wise DICE coefficient.
    """
    data, patient, gamma = args_tuple

    thr_mask = sitk.ReadImage(
        os.path.join(results_folder, f"thr_mask_g{gamma}", f"{patient}.nii")
    )

    gt = data["gt"]
    thr_mask = resample_to_reference(thr_mask, gt, sitk.sitkNearestNeighbor)
    dsc = evaluate_voxel_wise(thr_mask, gt)["vw-DSC"]
    return dsc

def main(data_folder, alpha, beta, gammas, results_folder,
         init_points, n_iter, n_cores):
    """
    Main pipeline for Bayesian optimization of segmentation settings.

    Performs:
    - Train-validation split
    - Image precomputation
    - Threshold mask computation
    - Refinement step optimization with Bayesian Optimization
    - Validation of best parameters

    Parameters
    ----------
    data_folder : str
        Path to the dataset directory.
    alpha : float
        Segmentation parameter alpha.
    beta : float
        Segmentation parameter beta.
    gammas : sequence of float
        Gamma values to evaluate.
    results_folder : str
        Directory to save outputs, logs, and metrics.
    init_points : int
        Number of initial random exploration steps for BO.
    n_iter : int
        Number of BO iterations.
    n_cores : int
        Number of CPU cores to use for multiprocessing.

    Returns
    -------
    None
    """

    def compute_images(alpha, beta, data=None, train_patients=[]):
        """
        Compute preprocessed FLAIR+T1 images for patients in chunks.

        Parameters
        ----------
        alpha : float
            Segmentation parameter alpha.
        beta : float
            Segmentation parameter beta.
        data : dict, optional
            Preloaded data dictionary. If None, data is loaded per chunk.
        train_patients : list of str
            List of patient IDs to process.

        Returns
        -------
        None
            Saves images to disk.
        """

        start = time.time()
        gc.collect()

        # chunk structure to avoid RAM overload
        cores = min(n_cores, len(train_patients))    
        n_chunks = (len(train_patients) + cores - 1) // cores

        for chunk in range(n_chunks):
            gc.collect()

            chunk_patients = train_patients[chunk*cores:(chunk+1)*cores]

            if data is None:
                chunk_data, _ = load_subjects(
                    data_folder, patients=chunk_patients
                )
            else:
                chunk_data = data
            
            args_list = [
                (chunk_data[patient], patient, alpha, beta)
                for patient in chunk_patients
            ]

            if cores > 1:
                with mp.Pool(cores) as pool:
                    _ = pool.starmap(
                        process_patient_images,
                        [(a, results_folder) for a in args_list],
                    )
            else:
                _ = [
                    process_patient_images(a, results_folder)
                    for a in args_list
                ]
            
        print(f"Elapsed time: {(time.time()-start):.3g} s")
              
    def compute_thr(gamma, data=None, train_patients=[]):
        """
        Compute threshold masks for a list of patients in parallel chunks.

        Parameters
        ----------
        gamma : int
            Thresholding parameter
        data : dict, optional
            Preloaded data dictionary. If None, data is loaded per chunk.
        train_patients : list of str
            List of patient IDs to process.

        Returns
        -------
        None
            Saves threshold masks to disk.
        """

        start = time.time()
        gc.collect()

        # chunk structure to avoid RAM overload
        cores = min(n_cores, len(train_patients))    
        n_chunks = (len(train_patients) + cores - 1) // cores

        for chunk in range(n_chunks):
            gc.collect()

            chunk_patients = train_patients[chunk*cores:(chunk+1)*cores]

            if data is None:
                chunk_data, _ = load_subjects(
                    data_folder, patients=chunk_patients
                )
            else:
                chunk_data = data
            
            args_list = [
                (chunk_data[patient], patient, gamma)
                for patient in chunk_patients
            ]

            if cores > 1:
                with mp.Pool(cores) as pool:
                    _ = pool.starmap(
                        process_patient_thr,
                        [(a, results_folder) for a in args_list],
                    )
            else:
                _ = [process_patient_thr(a, results_folder) for a in args_list]
            
        print(f"Elapsed time: {(time.time()-start):.3g} s")

    def evaluate_thr_dice(gamma, data=None, val_patients=[]):

        gc.collect()
        dice_list = []

        cores = min(n_cores, len(val_patients))
        n_chunks = (len(val_patients) + cores - 1) // cores

        for chunk in range(n_chunks):
            gc.collect()

            chunk_patients = val_patients[chunk*cores:(chunk+1)*cores]

            if data is None:
                chunk_data, _ = load_subjects(
                    data_folder, patients=chunk_patients
                )
            else:
                chunk_data = data

            args_list = [
                (chunk_data[patient], patient, gamma)
                for patient in chunk_patients
            ]

            if cores > 1:
                with mp.Pool(cores) as pool:
                    results = pool.starmap(
                        process_patient_thr_dice,
                        [(a, results_folder) for a in args_list],
                    )
            else:
                results = [
                    process_patient_thr_dice(a, results_folder)
                    for a in args_list
                ]

            dice_list.extend(results)

        return dice_list

    def dice_obj(extend_dilation_radius, n_std, min_diameter,
                 surround_dilation_radius, min_points,
                 gamma, data=None, train_patients=[]):
        """
        Compute DICE scores for multiple patients using refinement parameters.

        Parameters
        ----------
        extend_dilation_radius : float
        n_std : float
        min_diameter : float
        surround_dilation_radius : float
        min_points : float
        gamma : int
            Threshold parameter
        data : dict, optional
            Preloaded patient data
        train_patients : list of str
            Patient IDs to process

        Returns
        -------
        list of float
            DICE scores per patient.
        """

        start = time.time()
        gc.collect()
        dice_list = []

        # chunk structure to avoid RAM overload
        cores = min(n_cores, len(train_patients))
        n_chunks = (len(train_patients) + cores - 1) // cores

        for chunk in range(n_chunks):
            gc.collect()

            chunk_patients = train_patients[
                chunk * cores:(chunk + 1) * cores
            ]

            if data is None:
                chunk_data, _ = load_subjects(
                    data_folder, patients=chunk_patients
                )
            else:
                chunk_data = data

            args_list = [
                (
                    chunk_data[patient],
                    patient,
                    gamma,
                    extend_dilation_radius,
                    n_std,
                    min_diameter,
                    surround_dilation_radius,
                    min_points,
                )
                for patient in chunk_patients
            ]

            if cores > 1:
                with mp.Pool(cores) as pool:
                    results = pool.starmap(
                        process_patient_rs, [(a, results_folder)
                                             for a in args_list]
                    )
            else:
                results = [process_patient_rs(a, results_folder)
                           for a in args_list]

            dice_list.extend(results)

        print(f"Elapsed time: {(time.time()-start):.3g} s")
        return dice_list


    def dice_obj_mean(*args, **kwargs):
        """
        Compute the mean DICE score for multiple patients.

        This is the objective used by the Bayesian optimizer.

        Parameters
        ----------
        *args, **kwargs : passed to ``dice_obj``

        Returns
        -------
        float
            Mean DICE score across all patients.
        """
        tr_dice_list = dice_obj(*args,
                                train_patients=tr_patients,
                                data=preloaded_data,
                                **kwargs)
        return np.mean(tr_dice_list)

    # resolve alpha / beta
    if alpha is None or beta is None:
        yaml_alpha, yaml_beta = load_alpha_beta_from_yaml()

        if alpha is None:
            alpha = yaml_alpha
        if beta is None:
            beta = yaml_beta

    # create results folder
    os.makedirs(results_folder, exist_ok=True)
    
    # train-val-test split
    if (
        os.path.isfile(os.path.join(data_folder,
                                    "train_patients.pkl"))
        and os.path.isfile(os.path.join(data_folder,
                                        "validation_patients.pkl"))
    ):
        tr_patients = list(
            pd.read_pickle(os.path.join(data_folder,
                                        "train_patients.pkl"))
        )
        val_patients = list(
            pd.read_pickle(os.path.join(data_folder,
                                        "validation_patients.pkl"))
        )
        print(
            f"Found train-validation split in '{data_folder}'. "
            f"{len(tr_patients)} patients will be used for training, "
            f"{len(val_patients)} for validation."
        )
    else:
        tr_patients, val_patients, _ = train_val_test_split(
            data_folder,
            validation_fraction=0.3,
            test_fraction=0.1,
            show=False,
            title=(
                "Stratification of train–validation–test split according to "
                "positive-to-total ratio"
            ),
        )

    # optimization
    if not os.path.isfile(os.path.join(results_folder, "best_params.pkl")):

        # compute images for ALL patients        
        patients_to_do = []
        for patient in tr_patients + val_patients:
            if not os.path.isfile(os.path.join(results_folder,
                                               "images",
                                               f"{patient}.nii")):
                patients_to_do.append(patient)

        if len(patients_to_do) != 0:
            if len(patients_to_do) <= n_cores:
                preloaded_data, _ = load_subjects(data_folder,
                                                  patients=patients_to_do)
            else:   # light on RAM, but slower in time
                preloaded_data = None
            
            os.makedirs(os.path.join(results_folder, "images"), exist_ok=True)
            print("Computing FLAIR + T1 images for all patients...")
            compute_images(alpha, beta, data=preloaded_data,
                        train_patients=patients_to_do)
        
        # compute threshold masks for ALL patients
        for gamma in gammas:
            patients_to_do = []
            for patient in tr_patients + val_patients:
                thr_path = os.path.join(
                    results_folder, f"thr_mask_g{gamma}", f"{patient}.nii"
                )
                if not os.path.isfile(thr_path):
                    patients_to_do.append(patient)

            if len(patients_to_do) != 0:
                if len(patients_to_do) <= n_cores:
                    preloaded_data, _ = load_subjects(
                        data_folder, patients=patients_to_do
                    )
                else:  # light on RAM, but slower in time
                    preloaded_data = None

                os.makedirs(
                    os.path.join(results_folder, f"thr_mask_g{gamma}"),
                    exist_ok=True,
                )
                msg = (
                    "Computing threshold masks for all patients "
                    "(gamma = {})...".format(gamma)
                )
                print(msg)
                compute_thr(
                    gamma, data=preloaded_data, train_patients=patients_to_do
                )
                
        # optimize refinement step
        for gamma in gammas:
            rs_dir = os.path.join(results_folder, f"rs_g{gamma}")

            if os.path.isfile(os.path.join(rs_dir, "best_params.pkl")):
                print(f"Found optimized parameters for gamma = {gamma}")
            else:
                print(f"Optimizing parameters for gamma = {gamma}...")
                os.makedirs(rs_dir, exist_ok=True)

                if len(tr_patients) <= n_cores:
                    preloaded_data, _ = load_subjects(
                        data_folder, patients=tr_patients
                    )
                else:  # light on RAM, but slower in time
                    preloaded_data = None

                optimizer = BayesianOptimization(
                    partial(dice_obj_mean, gamma=gamma), pbounds=pbounds,
                    random_state=42
                )
                optimizer.maximize(init_points=init_points, n_iter=n_iter)

                best_params = optimizer.max["params"]
                pd.DataFrame(best_params, index=[0]).to_pickle(
                    os.path.join(rs_dir, "best_params.pkl")
                )
                history = optimizer.res
                pd.DataFrame(history).to_pickle(
                    os.path.join(rs_dir, "history.pkl")
                )
        
        # choose best parameters from validation set
        if os.path.isfile(os.path.join(results_folder, "val_dices.pkl")):
            val_dices = pd.read_pickle(
                os.path.join(results_folder, "val_dices.pkl")
            )
        else:
            val_dices = pd.DataFrame(
                index=gammas, columns=["mean", "std", "q1", "q3"]
            )
            for gamma in gammas:
                print(f"Evaluating gamma = {gamma} on validation set...")
                
                rs_dir = os.path.join(results_folder, f"rs_g{gamma}")
                best_params = pd.read_pickle(
                    os.path.join(rs_dir, "best_params.pkl")
                ).iloc[0].to_dict()
                
                val_dice_list = dice_obj(
                    extend_dilation_radius=best_params[
                        "extend_dilation_radius"
                    ],
                    n_std=best_params["n_std"],
                    min_diameter=best_params["min_diameter"],
                    surround_dilation_radius=best_params[
                        "surround_dilation_radius"
                    ],
                    min_points=best_params["min_points"],
                    gamma=gamma,
                    train_patients=val_patients,
                )
                pd.Series(val_dice_list).to_pickle(
                    os.path.join(rs_dir, f"val_dice.pkl"))

                val_dices.loc[gamma, "mean"] = np.mean(val_dice_list)
                val_dices.loc[gamma, "std"]  = np.std(val_dice_list)
                val_dices.loc[gamma, "q1"]   = np.quantile(val_dice_list, 0.25)
                val_dices.loc[gamma, "q3"]   = np.quantile(val_dice_list, 0.75)

                print(f"Average DICE on validation set for gamma = {gamma}: "\
                    f"{val_dices.loc[gamma, 'mean']:.3g} ± "\
                    f"{val_dices.loc[gamma, 'std']:.3g}\n"\
                    f"IQR = [{val_dices.loc[gamma, 'q1']:.3g}, "\
                    f"{val_dices.loc[gamma, 'q3']:.3g}]")
            
            val_dices.to_pickle(
                os.path.join(results_folder, "val_dices.pkl")
            )
            

        best_gamma = val_dices["mean"].idxmax()
        print(
            f"Best gamma on validation set: {best_gamma} "
            f"(mean DICE = {val_dices.loc[best_gamma, 'mean']:.3g})"
        )
        best_params = pd.read_pickle(
            os.path.join(
                results_folder,
                f"rs_g{best_gamma}",
                "best_params.pkl",
            )
        ).iloc[0].to_dict()
        best_params["gamma"] = best_gamma
        pd.DataFrame(best_params, index=[0]).to_pickle(
            os.path.join(results_folder, "best_params.pkl")
        )
    
    else:
        best_params = pd.read_pickle(
            os.path.join(results_folder, "best_params.pkl")
        ).iloc[0].to_dict()
        best_gamma = best_params["gamma"]
    
    # training performance
    if os.path.isfile(os.path.join(results_folder, "tr_dices.pkl")):
        tr_dices = pd.read_pickle(
            os.path.join(results_folder, "tr_dices.pkl")
        )
    else:
        tr_dices = pd.DataFrame(
            index=gammas, columns=["mean", "std", "q1", "q3"]
        )
        for gamma in gammas:
            print(f"Evaluating gamma={gamma} on training set...")

            rs_dir = os.path.join(results_folder, f"rs_g{gamma}")
            best_params = pd.read_pickle(
                os.path.join(rs_dir, "best_params.pkl")
            ).iloc[0].to_dict()

            tr_dice_list = dice_obj(
                extend_dilation_radius=best_params[
                    "extend_dilation_radius"
                ],
                n_std=best_params["n_std"],
                min_diameter=best_params["min_diameter"],
                surround_dilation_radius=best_params[
                    "surround_dilation_radius"
                ],
                min_points=best_params["min_points"],
                gamma=gamma,
                train_patients=tr_patients,
            )

            tr_dices.loc[gamma, "mean"] = np.mean(tr_dice_list)
            tr_dices.loc[gamma, "std"] = np.std(tr_dice_list)
            tr_dices.loc[gamma, "q1"] = np.quantile(tr_dice_list, 0.25)
            tr_dices.loc[gamma, "q3"] = np.quantile(tr_dice_list, 0.75)

            print(
                f"Average DICE on training set for gamma = {gamma}: "
                f"{tr_dices.loc[gamma, 'mean']:.3g} ± "
                f"{tr_dices.loc[gamma, 'std']:.3g}\n"
                f"IQR = [{tr_dices.loc[gamma,'q1']:.3g}, "
                f"{tr_dices.loc[gamma,'q3']:.3g}]"
            )

        tr_dices.to_pickle(os.path.join(results_folder, "tr_dices.pkl"))

    # evaluate on validation set after threshold (no refinement)
    thr_val_path = os.path.join(results_folder, "val_dices_thr.pkl")
    if os.path.isfile(thr_val_path):
        val_dices_thr = pd.read_pickle(thr_val_path)
    else:
        val_dices_thr = pd.DataFrame(
            index=gammas, columns=["mean", "std", "q1", "q3"]
        )

        for gamma in gammas:
            rs_dir = os.path.join(results_folder, f"rs_g{gamma}")
            print(
                f"Evaluating DICE after threshold on validation set "
                f"(gamma = {gamma})..."
            )

            if len(val_patients) <= n_cores:
                preloaded_data, _ = load_subjects(
                    data_folder, patients=val_patients
                )
            else:
                preloaded_data = None

            val_dice_list_thr = evaluate_thr_dice(
                gamma=gamma, val_patients=val_patients, data=preloaded_data
            )

            pd.Series(val_dice_list_thr).to_pickle(
                os.path.join(rs_dir, f"val_dice_thr.pkl"))

            val_dices_thr.loc[gamma, "mean"] = np.mean(val_dice_list_thr)
            val_dices_thr.loc[gamma, "std"] = np.std(val_dice_list_thr)
            val_dices_thr.loc[gamma, "q1"] = np.quantile(val_dice_list_thr, 0.25)
            val_dices_thr.loc[gamma, "q3"] = np.quantile(val_dice_list_thr, 0.75)

            print(
                f"THR DICE (gamma={gamma}): "
                f"{val_dices_thr.loc[gamma,'mean']:.3g} ± "
                f"{val_dices_thr.loc[gamma,'std']:.3g} | "
                f"IQR = [{val_dices_thr.loc[gamma,'q1']:.3g}, "
                f"{val_dices_thr.loc[gamma,'q3']:.3g}]"
            )

        val_dices_thr.to_pickle(thr_val_path)

    print("\nBest parameters: ")
    for k, v in best_params.items():
        if k == "min_points":
            print(f"{k} = {int(v)}")       # truncate to integer
        else:
            print(f"{k} = {v:.3g}")
    print(
        f"Average DICE on training set for gamma = {best_gamma}: "
        f"{tr_dices.loc[best_gamma, 'mean']:.3g} ± "
        f"{tr_dices.loc[best_gamma, 'std']:.3g}\n"
        f"IQR = [{tr_dices.loc[best_gamma, 'q1']:.3g}, "
        f"{tr_dices.loc[best_gamma, 'q3']:.3g}]"
    )

    val_dices = pd.read_pickle(os.path.join(results_folder, "val_dices.pkl"))
    print(
        f"Average DICE on validation set for gamma = {best_gamma}: "
        f"{val_dices.loc[best_gamma, 'mean']:.3g} ± "
        f"{val_dices.loc[best_gamma, 'std']:.3g}\n"
        f"IQR = [{val_dices.loc[best_gamma, 'q1']:.3g}, "
        f"{val_dices.loc[best_gamma, 'q3']:.3g}]"
    )
    
    # save parameters
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    yaml_path = f"params_{timestamp}.yaml"

    params_yaml = {
        "alpha": alpha,
        "beta": beta,
        "gamma": float(best_gamma),
        "extend_dilation_radius": best_params["extend_dilation_radius"],
        "n_std": best_params["n_std"],
        "min_diameter": best_params["min_diameter"],
        "surround_dilation_radius": best_params["surround_dilation_radius"],
        "min_points": int(best_params["min_points"]),
    }

    with open(yaml_path, "w") as f:
        yaml.safe_dump(params_yaml, f, sort_keys=False)

    print(f"Saved parameters to {yaml_path}")
    
    # plot validation dices before and after refinement
    plot_validation_distributions(gammas, results_folder)


if __name__ == '__main__':

    args = parse_args()

    start_time = time.time()

    # load constants
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    gm_labels = config["labels"]["gm"]
    wm_labels = config["labels"]["wm"]
    keywords_to_remove = config["labels"]["keywords_to_remove"]

    flair_file = config["files"]["flair"]
    t1_file = config["files"]["t1"]
    segm_file = config["files"]["segmentation"]
    gm_pve_file = config["files"]["gm_pve"]
    wm_pve_file = config["files"]["wm_pve"]
    csf_pve_file = config["files"]["csf_pve"]
    gt_file = config["files"]["gt"]
    label_name_file = config["files"]["label_name"]
    
    main(args.data_folder, args.alpha, args.beta, args.gammas,
         args.results_folder, args.init_points, args.n_iter, args.n_cores)
    print(f"Elapsed time: {(time.time()-start_time):.3g} s")