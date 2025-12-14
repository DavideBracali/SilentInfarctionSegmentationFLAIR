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
import random
from bayes_opt import BayesianOptimization
import time
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns
import psutil
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

from SilentInfarctionSegmentationFLAIR.utils import (to_n_bit,
                                                     orient_image,
                                                     resample_to_reference,
                                                     get_array_from_image,
                                                     get_image_from_array,
                                                     label_names,
                                                     train_val_test_split,
                                                     cliffs_delta)
from SilentInfarctionSegmentationFLAIR.segmentation import (get_mask_from_segmentation,
                                                            evaluate_voxel_wise)
from SilentInfarctionSegmentationFLAIR.histograms import plot_multiple_histograms
from SilentInfarctionSegmentationFLAIR import flair_t1_sum


def parse_args():
    description="Optimizes all segmentation parameters (except gamma) on a training set, "\
        "maximizing ROC-AUC using Bayesian optimization. Then chooses gamma maximizing "\
        "average DICE coefficient on a validation set."

    parser = argparse.ArgumentParser(description=description)

    _ = parser.add_argument('--data_folder',
                            dest='data_folder',
                            action='store',
                            type=str,
                            required=False,
                            default="data",
                            help='Path of the folder where data is located')
    
    _ = parser.add_argument('--results_folder',
                            dest='results_folder',
                            action='store',
                            type=str,
                            required=False,
                            default="training",
                            help='Path of the folder where data is located')

    _ = parser.add_argument('--init_points',
                            dest='init_points',
                            action='store',
                            type=int,
                            required=False,
                            default=5,
                            help='Number of initial points for random search')
    
    _ = parser.add_argument('--n_iter',
                            dest='n_iter',
                            action='store',
                            type=int,
                            required=False,
                            default=20,
                            help='Number of Bayesian optimization iterations')

    _ = parser.add_argument('--n_cores',
                            dest='n_cores',
                            action='store',
                            type=int,
                            required=False,
                            default=1,
                            help='Number of CPU cores to use during optimization \
                                (improves computational time but increases RAM and CPU usage)')


    args = parser.parse_args()

    return args


pbounds={"alpha": (1, 10),
        "beta": (0, 5)}

gm_labels = [3, 8, 10, 11, 12, 13, 17, 18, 26,
              42, 47, 49, 50, 51, 52, 53, 54, 62]
wm_labels = [2, 7, 41, 46]
flair_file = "FLAIR.nii"
t1_file = "T1ontoFLAIR.nii"
segm_file = "aseg.auto_noCCseg.nii"
gm_pve_file = "pve_gm.nii"
wm_pve_file = "pve_wm.nii"
csf_pve_file = "pve_csf.nii"
gt_file = "GT.nii"
label_name_file = "data/FreeSurferColorLUT.txt"
keywords_to_remove = [
    "Bone", "Teeth", "Cranium", "Skull", "Table", "Diploe", "Endosteum", "Periosteum",
    "CSF", "Ventricle", "Humor", "Lens", "Globe", "Choroid",
    "Epidermis", "Scalp", "Skin", "Dura",
    "Air", "Nasal"
]
labels_dict = label_names(label_name_file)

gammas = np.linspace(0.5, 5.0, 10)


def load_subjects(data_folder, patients=None, paths_only=False):
    """
    Load imaging data for multiple subjects and organize them into a dictionary.


    Parameters
    ----------
    data_folder : str
        Path to the folder containing patient subfolders with .nii files.
    patients : list of str, optional
        List of patient IDs to load. If None, all available patients are loaded.
    paths_only : bool, default=False
        If True, only the file paths are returned without loading images into memory.


    Returns
    -------
    data : dict
        Dictionary where keys are patient IDs and values are dictionaries containing
        SimpleITK images and masks.
    paths_df : pandas.DataFrame
        DataFrame mapping each patient to the corresponding file paths of required images.
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
            flair = to_n_bit(
                orient_image(sitk.ReadImage(paths_df.loc[patient, flair_file]), "RAS"), 8
            )

            t1 = to_n_bit(sitk.ReadImage(paths_df.loc[patient, t1_file]), 8)
            t1 = resample_to_reference(t1, flair)

            gt = resample_to_reference(sitk.ReadImage(paths_df.loc[patient, gt_file]), flair)
            gt = sitk.Cast(gt, sitk.sitkUInt8)

            segm = resample_to_reference(sitk.ReadImage(paths_df.loc[patient, segm_file]), flair)
            gm_mask = get_mask_from_segmentation(segm, gm_labels)
            wm_mask = get_mask_from_segmentation(segm, wm_labels)

            data[patient] = {
                "flair": flair, "t1": t1, "gt": gt,
                "gm_mask": gm_mask, "wm_mask": wm_mask}

    return data, paths_df


def process_patient(args_tuple):
    """
    Process a single patient's imaging data and compute histogram separation metrics.

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
        Sum of cliffs delta between lesions/GM and GM/WM
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

    if len(gm_gl) == 0 or len(wm_gl) == 0 or len(lesions_gl) == 0:
        return -100000       # penalty for configurations that produce empty masks

    downsample_size = 1000

    if len(gm_gl) > downsample_size:
        idx = np.linspace(0, len(gm_gl)-1, downsample_size, dtype=int)
        gm_gl =  np.sort(gm_gl)[idx]

    if len(lesions_gl) > downsample_size:
        idx = np.linspace(0, len(lesions_gl)-1, downsample_size, dtype=int)
        lesions_gl = np.sort(lesions_gl)[idx]

    delta_l_gm = cliffs_delta(lesions_gl, gm_gl)    # big if gm < lesions  (more true positives)
    delta_gm_wm = cliffs_delta(gm_gl, wm_gl)        # big if wm < gm (less false positives)

    return delta_l_gm + delta_gm_wm


def main(data_folder, results_folder, init_points, n_iter, n_cores):
    """
    Main execution pipeline for Bayesian optimization of segmentation parameters.

    Steps:
    - Load patient data
    - Split into training and validation sets
    - Perform Bayesian optimization on training patients
    - Save best parameters and training separation
    - Validate best parameters on the validation set

    Parameters
    ----------
    data_folder : str
        Path to the input dataset directory.
    results_folder : str
        Path where outputs, logs, and metrics will be saved.
    init_points : int
        Number of initial random exploration steps for Bayesian Optimization.
    n_iter : int
        Number of optimization iterations.
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
            
        print("Elapsed time:", time.time()-start)
        return separation_list

    def separation_obj_mean(*args, **kwargs):
        global tr_separation_list
        tr_separation_list = separation_obj(*args,
                                train_patients=tr_patients,
                                data=preloaded_data,
                                **kwargs)
        return np.mean(tr_separation_list)

    # create results folder
    os.makedirs(results_folder, exist_ok=True)
    
    # train-val-test split
    if os.path.isfile(os.path.join(results_folder, "tr_separation.npy")):
        best_params = pd.read_pickle(os.path.join(results_folder, "best_params.pkl")).to_dict()
        tr_separation_list = np.load(os.path.join(results_folder, "tr_separation.npy"))
    else:
        # check if there is an existing train/val split
        if os.path.isfile(os.path.join(data_folder, "train_patients.pkl")
            ) and os.path.isfile(os.path.join(data_folder, "validation_patients.pkl")):
            tr_patients = list(pd.read_pickle(os.path.join(data_folder, "train_patients.pkl")))
            val_patients = list(pd.read_pickle(os.path.join(data_folder, "validation_patients.pkl")))
            print(f"Found train-validation split in '{data_folder}'. " \
                f"{len(tr_patients)} patients will be used for training, {len(val_patients)} for validation.")
        else:
            tr_patients, val_patients, _ = train_val_test_split(data_folder,
                                                    validation_fraction=0.3, test_fraction=0.1,
                                                    pos_neg_stratify=True, show=False,
                title="Stratification of train–validation–test split according to positive-to-total ratio")
            
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
        # tr_separation_list is a global variable defined inside auc_obj_global
        np.save(os.path.join(results_folder, "tr_separation.npy"), np.array(tr_separation_list))
        history = optimizer.res
        pd.DataFrame(history).to_pickle(os.path.join(results_folder, "history.pkl"))

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


    

if __name__ == '__main__':

    args = parse_args()

    start_time = time.time()
    main(args.data_folder, args.results_folder, args.init_points, args.n_iter, args.n_cores)
    print(f"Elapsed time: {(time.time()-start_time):.2f} s")