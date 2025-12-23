#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tune alpha and beta segmentation parameters.

This module provides a CLI and helper functions to optimize segmentation
parameters with Bayesian optimization. It loads subject images, computes
a per-patient separation score and runs optimization on a training set,
then validates the best parameters on a validation set.

Notes
-----
The script writes best parameters to `params_alpha_beta.yaml` but leaves
other parameters unset for further tuning.
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

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

from SilentInfarctionSegmentationFLAIR.utils import (
    orient_image,
    resample_to_reference,
    get_array_from_image,
    get_image_from_array,
    label_names,
    train_val_test_split,
    cliffs_delta,
)
from SilentInfarctionSegmentationFLAIR.segmentation import (
    get_mask_from_segmentation,
    evaluate_voxel_wise,
)
from SilentInfarctionSegmentationFLAIR.histograms import (
    plot_multiple_histograms,
)
from SilentInfarctionSegmentationFLAIR import flair_t1_sum


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: `data_folder`, `results_folder`,
        `init_points`, `n_iter`, `n_cores`.
    """

    description = (
        "Optimizes all segmentation parameters (except gamma) on a "
        "training set, maximizing ROC-AUC using Bayesian optimization. "
        "Then chooses gamma maximizing average DICE coefficient on a "
        "validation set."
    )

    parser = argparse.ArgumentParser(description=description)

    _ = parser.add_argument(
        "--data_folder",
        dest="data_folder",
        action="store",
        type=str,
        required=False,
        default="data",
        help="Path of the folder where data is located",
    )

    _ = parser.add_argument(
        "--results_folder",
        dest="results_folder",
        action="store",
        type=str,
        required=False,
        default="tuning_alpha_beta",
        help="Path of the folder where data is located",
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

    args = parser.parse_args()
    return args


pbounds={"alpha": (1, 10),
        "beta": (0, 5)}

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
        If True, only returns file paths without loading images into
        memory. Default is False.

    Returns
    -------
    data : dict
        Keys are patient IDs and values are dicts with the following
        images (SimpleITK.Image): ``flair``, ``t1``, ``gt``, ``gm_mask``,
        ``wm_mask``. If ``paths_only`` is True, ``data`` will be empty.
    paths_df : pandas.DataFrame
        DataFrame mapping each patient to their corresponding file
        paths. Rows with missing images are dropped.
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

            gt = resample_to_reference(sitk.ReadImage(paths_df.loc[patient, gt_file]), flair,
                                       sitk.sitkNearestNeighbor)
            gt = sitk.Cast(gt, sitk.sitkUInt8)

            segm = resample_to_reference(sitk.ReadImage(paths_df.loc[patient, segm_file]), flair,
                                         sitk.sitkNearestNeighbor)
            gm_mask = get_mask_from_segmentation(segm, gm_labels)
            wm_mask = get_mask_from_segmentation(segm, wm_labels)

            data[patient] = {
                "flair": flair,
                "t1": t1,
                "gt": gt,
                "gm_mask": gm_mask,
                "wm_mask": wm_mask,
            }

    return data, paths_df


def process_patient(args_tuple):
    """
    Compute the separation score for one patient.

    The score rewards separation between gray matter (GM) and lesions
    while penalizing overlap between white matter (WM) and GM. Higher
    values indicate better separation and fewer false positives.

    Parameters
    ----------
    args_tuple : tuple
        (data, alpha, beta) where ``data`` is a dict with keys:
        ``flair``, ``t1``, ``gt``, ``gm_mask``, ``wm_mask``. ``alpha``
        and ``beta`` are floats controlling the `flair_t1_sum` behavior.

    Returns
    -------
    float
        Separation score (reward minus penalty). Larger is better.
    """
    data, alpha, beta = args_tuple

    flair = data["flair"]
    t1 = data["t1"]
    gt = data["gt"]
    gm_mask = data["gm_mask"]
    wm_mask = data["wm_mask"]

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

    gm_arr = get_array_from_image(sitk.Mask(image, gm_mask))
    wm_arr = get_array_from_image(sitk.Mask(image, wm_mask))
    lesions_arr = get_array_from_image(sitk.Mask(image, gt))
    gm_gl = gm_arr[gm_arr > 0]
    wm_gl = wm_arr[wm_arr > 0]
    lesions_gl = lesions_arr[lesions_arr > 0]
    
    les_low = np.quantile(lesions_gl, 0.25)
    gm_high = np.quantile(gm_gl, 0.75)
    scale = np.quantile(gm_gl, 0.75) - np.quantile(gm_gl, 0.25) + 1e-6
    reward = (les_low - gm_high) / scale

    gm_mode = np.argmax(np.bincount(gm_gl))
    mean_wm_above_gm = np.mean(wm_gl > gm_mode) 
    penalty = np.exp(5*mean_wm_above_gm) - 1

    # maximization logic:
    # if gm << lesions this is good (true positives)
    # but if wm and gl overlap this is very bad
    # because in that case we have some wm > gm_mode (false positives)
    # so we try to separate lesions and gm
    # as long wm and gm don't overlap too much
    # so we try to maximize:

    return reward - max(penalty, 0)


def main(data_folder, results_folder, init_points, n_iter, n_cores):
    """
    Run Bayesian optimization to tune ``alpha`` and ``beta``.

    Workflow
    --------
    - Load patient file paths or images
    - Create a train/validation split
    - Optimize parameters on the training set using Bayesian
      optimization
    - Evaluate best parameters on training and validation sets
    - Save results and a minimal `params_alpha_beta.yaml` file with
      ``alpha`` and ``beta`` set.

    Parameters
    ----------
    data_folder : str
        Path to the dataset directory.
    results_folder : str
        Directory where output files will be saved.
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
    iteration = 0

    def separation_obj(alpha, beta, data=None, train_patients=[]):

        nonlocal n_cores
        nonlocal iteration
        iteration += 1

        start = time.time()
        gc.collect()
        separation_list = []

        # chunk structure to avoid RAM overload
        if n_cores > len(train_patients):
            n_cores = len(train_patients)
            
        n_chunks = (len(train_patients) + n_cores - 1) // n_cores

        for chunk in range(n_chunks):
            gc.collect()

            chunk_patients = train_patients[chunk*n_cores:(chunk+1)*n_cores]

            if data is None:
                chunk_data, _ = load_subjects(data_folder, patients=chunk_patients)
            else:
                chunk_data = data
            
            args_list = [
                (chunk_data[patient], alpha, beta) for patient in chunk_patients]

            if n_cores > 1:
                with mp.Pool(n_cores) as pool:
                    results = pool.map(process_patient, args_list)
            else:
                results = [process_patient(a) for a in args_list]

            separation_list.extend(results)  
            
        print(f"Elapsed time: {(time.time()-start):.3g} s")
        return separation_list

    def separation_obj_mean(*args, **kwargs):
        tr_separation_list = separation_obj(*args,
                                train_patients=tr_patients,
                                data=preloaded_data,
                                **kwargs)
        return np.mean(tr_separation_list)

    # create results folder
    os.makedirs(results_folder, exist_ok=True)
    
    # train-val-test split
    if (
        os.path.isfile(os.path.join(data_folder, "train_patients.pkl"))
        and os.path.isfile(os.path.join(data_folder, "validation_patients.pkl"))
    ):
        tr_patients = list(
            pd.read_pickle(os.path.join(data_folder, "train_patients.pkl"))
        )
        val_patients = list(
            pd.read_pickle(os.path.join(data_folder, "validation_patients.pkl"))
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

        if len(tr_patients) <= n_cores:  # preload data (faster but heavy in RAM)
            print(
                f"{len(tr_patients)} training patients and {n_cores} "
                "avalible cores. Preloading data in advance..."
            )
            preloaded_data, _ = load_subjects(
                data_folder, patients=tr_patients
            )
        else:  # light on RAM, but slower in time
            preloaded_data = None
            print(
                f"{len(tr_patients)} training patients and {n_cores} "
                "avalible cores. Data will be loaded at each iteration..."
            )

        # maximize training separation
        optimizer = BayesianOptimization(
            f=separation_obj_mean, pbounds=pbounds, random_state=42
        )
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        best_params = optimizer.max["params"]
        pd.DataFrame(best_params, index=[0]).to_pickle(
            os.path.join(results_folder, "best_params.pkl")
        )
        history = optimizer.res
        pd.DataFrame(history).to_pickle(os.path.join(results_folder, "history.pkl"))

    else:
        best_params = pd.read_pickle(
            os.path.join(results_folder, "best_params.pkl")
        ).iloc[0].to_dict()
    
    # training performance
    if os.path.isfile(os.path.join(results_folder, "tr_separation.npy")):
        tr_separation_list = np.load(
            os.path.join(results_folder, "tr_separation.npy")
        )
    else:
        print("Evaluating best parameters on training set...")
        tr_separation_list = separation_obj(
            alpha=best_params["alpha"],
            beta=best_params["beta"],
            train_patients=tr_patients,
            data=None,
        )
        np.save(
            os.path.join(results_folder, "tr_separation.npy"),
            tr_separation_list,
        )

            

    # validation
    print("Validating best parameters on validation set...")
    if os.path.isfile(os.path.join(results_folder, "val_separation.npy")):
        val_separation_list = np.load(
            os.path.join(results_folder, "val_separation.npy")
        )
    else:
        val_data, _ = load_subjects(data_folder, patients=val_patients)
        val_separation_list = [
            process_patient(
                (val_data[patient],)
                + tuple(best_params[k] for k in list(pbounds.keys()))
            )
            for patient in val_patients
        ]
        np.save(
            os.path.join(results_folder, "val_separation.npy"),
            np.array(val_separation_list),
        )


    print("\nBest parameters: ")
    for k, v in best_params.items():
        print(f"{k} = {v:.3g}")
    
    print(
        "Average separation on training set: "
        f"{np.mean(tr_separation_list):.3g} ± {np.std(tr_separation_list):.3g}"
    )

    print(
        "Average separation on validation set: "
        f"{np.mean(val_separation_list):.3g} ± {np.std(val_separation_list):.3g}"
    )
    
    # save alpha and beta
    params_yaml = {
        "alpha": float(best_params["alpha"]),
        "beta": float(best_params["beta"]),
        "gamma": None,
        "extend_dilation_radius": None,
        "n_std": None,
        "min_diameter": None,
        "surround_dilation_radius": None,
        "min_points": None,
    }
    
    yaml_path = "params_alpha_beta.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(params_yaml, f, sort_keys=False)

    print(f"Saved parameters to {yaml_path} (still not ready for use, "
          "please tune the other parameters by running "
          "SilentInfarctionSegmentationFLAIR/tuning_gamma_rs.py")


if __name__ == '__main__':

    args = parse_args()

    start_time = time.time()
    main(args.data_folder, args.results_folder, args.init_points,
         args.n_iter, args.n_cores)
    print(f"Elapsed time: {(time.time()-start_time):.3g} s")