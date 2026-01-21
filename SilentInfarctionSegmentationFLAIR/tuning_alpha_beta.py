#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-11-01 13:21:28

@author: david
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
import pathlib
import matplotlib
matplotlib.use("Agg")
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

from SilentInfarctionSegmentationFLAIR.utils import (orient_image,
                                                     resample_to_reference,
                                                     get_array_from_image,
                                                     get_image_from_array,
                                                     label_names,
                                                     train_val_test_split,
                                                     cliffs_delta)
from SilentInfarctionSegmentationFLAIR.segmentation import (
    get_mask_from_segmentation
)
from SilentInfarctionSegmentationFLAIR.utils import (
    get_settings_path
)
from SilentInfarctionSegmentationFLAIR import flair_t1_sum

CONFIG_PATH = get_settings_path("config.yaml")

def parse_args():
    """
    Parse command-line arguments for optimizing alpha and beta parameters.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: `data_folder`, `results_folder`,
        `init_points`, `n_iter`, `n_cores`.
    """
    description="Optimizes alpha and beta parameters on a training set, "\
        "maximizing separation between histograms of different tissues."

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
        default="tuning_alpha_beta",
        help="Path of the folder where intermediate results are stored",
    )

    _ = parser.add_argument(
        "--init_points",
        dest="init_points",
        action="store",
        type=int,
        required=False,
        default=30,
        help="Number of initial points for random search",
    )

    _ = parser.add_argument(
        "--n_iter",
        dest="n_iter",
        action="store",
        type=int,
        required=False,
        default=70,
        help="Number of Bayesian optimization iterations",
    )

    _ = parser.add_argument('--n_cores',
                            dest='n_cores',
                            action='store',
                            type=int,
                            required=False,
                            default=1,
                            help='Number of CPU cores to use during ' \
                                'optimization (improves computational ' \
                                'time but increases RAM and CPU usage)')
    
    args = parser.parse_args()

    return args


pbounds={"alpha": (1, 10),
        "beta": (0, 5)}

def load_subjects(data_folder, patients=None, paths_only=False):
    """
    Load imaging data for multiple subjects and organize into a dictionary.

    Parameters
    ----------
    data_folder : str
        Path to folder containing patient subfolders with .nii files.
    patients : list of str, optional
        List of patient IDs to load; if None, loads all patients.
    paths_only : bool, default=False
        If True, only returns file paths without loading images into memory.

    Returns
    -------
    data : dict
        Dictionary where keys are patient IDs and values are dictionaries containing:
            - flair : SimpleITK.Image, FLAIR image
            - t1 : SimpleITK.Image, T1-weighted image
            - gt : SimpleITK.Image, ground truth mask
            - gm_mask : SimpleITK.Image, GM mask from segmentation
            - wm_mask : SimpleITK.Image, WM mask from segmentation
    paths_df : pandas.DataFrame
        DataFrame mapping each patient to their corresponding file paths.
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
       print("WARNING!!! The following patients will be removed because " \
                      f"one of the required images is missing: {list(dropped_patients)}")
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
                "flair": flair, "t1": t1, "gt": gt,
                "gm_mask": gm_mask, "wm_mask": wm_mask}

    return data, paths_df


def process_patient(args_tuple):
    """
    Process a single patient's data and compute the separation score.

    The score is designed to:
        - Maximize separation between GM and lesions (true positives)
        - Penalize overlap between WM and GM (false positives)

    Parameters
    ----------
    args_tuple : tuple
        Tuple containing:
            - data : dict
                Dictionary with patient images and masks.
            - alpha : float
                Segmentation parameter alpha.
            - beta : float
                Segmentation parameter beta.

    Returns
    -------
    float
        Separation score: larger values indicate better GM-lesion separation
        and minimal WM overlap.
    """
    data, alpha, beta = args_tuple

    flair = data["flair"]
    t1 = data["t1"]
    gt = data["gt"]
    gm_mask = data["gm_mask"]
    wm_mask = data["wm_mask"]

    image = flair_t1_sum.main(flair, t1, alpha, beta,
                                                      wm_mask=wm_mask,
                                                      gm_mask=gm_mask,
                                                      gt=gt,
                                                      verbose=False)
    
    gm_arr = get_array_from_image(sitk.Mask(image, gm_mask))
    wm_arr = get_array_from_image(sitk.Mask(image, wm_mask))
    lesions_arr = get_array_from_image(sitk.Mask(image, gt))
    gm_gl = gm_arr[gm_arr>0]
    wm_gl = wm_arr[wm_arr>0]
    lesions_gl = lesions_arr[lesions_arr>0]
    
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
    Main pipeline for Bayesian optimization of segmentation parameters.

    Steps:
    - Load patient data
    - Split into training and validation sets
    - Optimize alpha and beta using Bayesian Optimization
    - Save best parameters and separation metrics
    - Validate the best parameters on the validation set

    Parameters
    ----------
    data_folder : str
        Path to the dataset directory.
    results_folder : str
        Path to save outputs, logs, and metrics.
    init_points : int
        Number of initial random exploration steps for Bayesian Optimization.
    n_iter : int
        Number of BO iterations.
    n_cores : int
        Number of CPU cores to use for multiprocessing.

    Returns
    -------
    None
    """

    def separation_obj(alpha, beta, data=None, train_patients=[]):

        start = time.time()
        gc.collect()
        separation_list = []

        # chunk structure to avoid RAM overload
        cores = min(n_cores, len(train_patients))    
        n_chunks = (len(train_patients) + cores - 1) // cores

        for chunk in range(n_chunks):
            gc.collect()

            chunk_patients = train_patients[chunk*cores:(chunk+1)*cores]

            if data is None:
                chunk_data, _ = load_subjects(data_folder,
                                              patients=chunk_patients)
            else:
                chunk_data = data
            
            args_list = [
                (chunk_data[patient], alpha, beta)
                    for patient in chunk_patients]

            if cores > 1:
                with mp.Pool(cores) as pool:
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
    all_patients = list(load_subjects(data_folder, paths_only=True)[1].index)
    single_patient_mode = len(all_patients) == 1
    if single_patient_mode:
        print(f"WARNING!! Single patient detected: {all_patients[0]}. The " \
              "same patient will be used BOTH for TRAINING and VALIDATION.")
        tr_patients = all_patients
        val_patients = all_patients
    else:
        if os.path.isfile(os.path.join(data_folder,
                                    "train_patients.pkl")
            ) and os.path.isfile(os.path.join(
                data_folder, "validation_patients.pkl")):
            tr_patients = list(pd.read_pickle(os.path.join(
                data_folder, "train_patients.pkl")))
            val_patients = list(pd.read_pickle(os.path.join(
                data_folder, "validation_patients.pkl")))
            print(f"Found train-validation split in '{data_folder}'. " \
                f"{len(tr_patients)} patients will be used for training, \
                {len(val_patients)} for validation.")
        else:
            tr_patients, val_patients, _ = train_val_test_split(
                data_folder,
                validation_fraction=0.3,
                test_fraction=0.1,
                show=False,
                title="Stratification of train–validation–test split" \
                    "according to positive-to-total ratio")

    # optimization
    if not os.path.isfile(os.path.join(results_folder, "best_params.pkl")):
        
        if len(tr_patients) <= n_cores: # preload data (faster but heavy in RAM) 
            print(f"{len(tr_patients)} training patients and {n_cores} avalible cores. " \
                        "Preloading data in advance...")
            preloaded_data, _ = load_subjects(data_folder, patients=tr_patients)
        else:   # light on RAM, but slower in time
            preloaded_data = None
            print(f"{len(tr_patients)} training patients and {n_cores} avalible cores. " \
                    "Data will be loaded at each iteration...")

        # maximize training separation    
        optimizer = BayesianOptimization(f=separation_obj_mean, pbounds=pbounds, random_state=42)
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        best_params = optimizer.max["params"]
        pd.DataFrame(best_params, index=[0]).to_pickle(os.path.join(results_folder, "best_params.pkl"))
        history = optimizer.res
        pd.DataFrame(history).to_pickle(os.path.join(results_folder, "history.pkl"))

    else:
        best_params = pd.read_pickle(os.path.join(results_folder, "best_params.pkl")).iloc[0].to_dict()
    
    # training performance
    if os.path.isfile(os.path.join(results_folder, "tr_separation.npy")):
        tr_separation_list = np.load(os.path.join(results_folder, "tr_separation.npy"))
    else:
        print("Evaluating best parameters on training set...")
        tr_separation_list = separation_obj(alpha=best_params["alpha"],
                                            beta=best_params["beta"],
                                            train_patients=tr_patients,
                                            data=None)
        np.save(os.path.join(results_folder, "tr_separation.npy"), tr_separation_list)

            

    # validation
    print("Validating best parameters on validation set...")
    if os.path.isfile(os.path.join(results_folder, "val_separation.npy")):
        val_separation_list = np.load(os.path.join(results_folder, "val_separation.npy"))
    else:
        val_data, _ = load_subjects(data_folder, patients=val_patients)
        val_separation_list = [process_patient((val_data[patient],) + tuple(best_params[k]
                                                                for k in list(pbounds.keys())))
                        for patient in val_patients]
        np.save(os.path.join(results_folder, "val_separation.npy"), np.array(val_separation_list))



    print("\nBest parameters: ")
    for k, v in best_params.items():
        print(f"{k} = {v:.3g}")
    
    print("Average separation on training set: "\
          f"{np.mean(tr_separation_list):.3g} ± {np.std(tr_separation_list):.3g}")
    
    if not single_patient_mode:
        print("Average separation on validation set: "\
              f"{np.mean(val_separation_list):.3g} ± {np.std(val_separation_list):.3g}")
    
    # save alpha and beta
    params_yaml = {
        "alpha": float(best_params["alpha"]),
        "beta": float(best_params["beta"]),
        "gamma": None,
        "extend_dilation_radius": None,
        "n_std": None,
        "min_diameter": None,
        "surround_dilation_radius": None,
        "min_points": None}
    
    yaml_path = get_settings_path("params_alpha_beta.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(params_yaml, f, sort_keys=False)
    
    script = os.path.join("SilentInfarctionSegmentationFLAIR",
        "tuning_gamma_rs.py")
    print(
        f"Saved parameters to {yaml_path} (still not ready for use, "
        f"please tune the other parameters by running {script})"
    )


if __name__ == '__main__':

    start_time = time.time()

    args = parse_args()

    # load constants
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    gm_labels = config["labels"]["gm"]
    wm_labels = config["labels"]["wm"]
    flair_file = config["files"]["flair"]
    t1_file = config["files"]["t1"]
    segm_file = config["files"]["segmentation"]
    gt_file = config["files"]["gt"]

    main(args.data_folder, args.results_folder, args.init_points,
         args.n_iter, args.n_cores)
    print(f"Elapsed time: {(time.time()-start_time):.3g} s")