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
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

from SilentInfarctionSegmentationFLAIR.utils import (to_n_bit,
                                                     orient_image,
                                                     resample_to_reference,
                                                     get_array_from_image,
                                                     get_image_from_array,
                                                     label_names,
                                                     train_val_test_split)
from SilentInfarctionSegmentationFLAIR.segmentation import (get_mask_from_segmentation,
                                                            evaluate_voxel_wise)
from SilentInfarctionSegmentationFLAIR.refinement import gaussian_transform
from SilentInfarctionSegmentationFLAIR import threshold
from SilentInfarctionSegmentationFLAIR import refinement_step


def parse_args():
    description="!!!!!!!!! scriverla"

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


pbounds={"gamma": (0, 10),
        "alpha": (1, 10),
        "beta": (0, 5),
        "extend_dilation_radius": (1, 5),
        "n_std": (0, 5),
        "min_diameter": (1, 3),
        "surround_dilation_radius": (1, 5),
        "min_points": (1, 4)}

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

def print_ram(prefix="RAM"):
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024**2
    print(f"{prefix}: {ram_mb:.2f} MB")

def load_subjects(data_folder, patients=None, paths_only=False):

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

            wm_pve = resample_to_reference(sitk.ReadImage(paths_df.loc[patient, wm_pve_file]), flair)
            gm_pve = resample_to_reference(sitk.ReadImage(paths_df.loc[patient, gm_pve_file]), flair)
            csf_pve = resample_to_reference(sitk.ReadImage(paths_df.loc[patient, csf_pve_file]), flair)

            data[patient] = {
                "flair": flair, "t1": t1, "gt": gt,
                "gm_mask": gm_mask, "wm_mask": wm_mask,
                "wm_pve": wm_pve, "gm_pve": gm_pve, "csf_pve": csf_pve
            }

    return data, paths_df


def process_patient(args_tuple):
    data, gamma, alpha, beta, extend_dilation_radius, n_std, \
    min_diameter, surround_dilation_radius, min_points = args_tuple

    flair = data["flair"]
    t1 = data["t1"]
    gt = data["gt"]
    gm_mask = data["gm_mask"]
    wm_mask = data["wm_mask"]
    wm_pve = data["wm_pve"]
    gm_pve = data["gm_pve"]
    csf_pve = data["csf_pve"]

    # weighted sum of flair and gaussian transformed t1
    wm_t1 = sitk.Mask(t1, wm_mask)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(wm_t1)
    wm_mean = stats.GetMean()
    wm_std = stats.GetSigma()

    t1_gauss = gaussian_transform(t1, mean=wm_mean, std=alpha*wm_std)
    t1_gauss_arr = beta * get_array_from_image(t1_gauss)
    t1_gauss_scaled = get_image_from_array(
        t1_gauss_arr, sitk.Cast(t1_gauss, sitk.sitkUInt16)
    )

    image = sitk.Cast(flair, sitk.sitkUInt16) + t1_gauss_scaled
    image = to_n_bit(image, 8)

    # threshold
    gm = sitk.Mask(image, gm_mask)
    thr_mask = threshold.main(image, gm, gamma=gamma, show=False, verbose=False)

    # refinement
    ref_mask = refinement_step.main(
        thr_mask,
        image=image,
        pves=[wm_pve, gm_pve, csf_pve],
        labels_dict=labels_dict,
        keywords_to_remove=keywords_to_remove,
        min_diameter=min_diameter,
        extend_dilation_radius=extend_dilation_radius,
        n_std=n_std,
        surround_dilation_radius=surround_dilation_radius,
        min_points=int(min_points),
        verbose=False
    )

    dice = evaluate_voxel_wise(ref_mask, gt)["vw-DSC"]
    return dice

def plot_fold_distributions(tr_dice, val_dice):
    data = []
    K = len(tr_dice)

    for k in range(K):
        for v in tr_dice[k]:
            data.append({
                "Fold": f"Fold {k+1}",
                "Type": "Train",
                "DICE": v
            })
        for v in val_dice[k]:
            data.append({
                "Fold": f"Fold {k+1}",
                "Type": "Validation",
                "DICE": v
            })

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Fold", y="DICE", hue="Type", palette="Set2")

    # aggiungiamo i punti veri
    sns.stripplot(data=df, x="Fold", y="DICE", hue="Type",
                  dodge=True, jitter=0.15, alpha=0.7, color="black")

    # doppia legenda → rimuoviamone una
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2])

    plt.title("Train vs Validation DICE per fold")
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def main(data_folder, results_folder, init_points, n_iter, n_cores):
 
    def dice_obj(gamma, alpha, beta, extend_dilation_radius, n_std,
                 min_diameter, surround_dilation_radius, min_points,
                 data=None, train_patients=[]):

        start = time.time()
        gc.collect()
        nonlocal n_cores
        dice_list = []

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
                (chunk_data[patient], gamma, alpha, beta, extend_dilation_radius,
                 n_std, min_diameter, surround_dilation_radius, min_points)
                for patient in chunk_patients]

            if n_cores > 1:
                with mp.Pool(n_cores) as pool:
                    results = pool.map(process_patient, args_list)
            else:
                results = [process_patient(a) for a in args_list]

            dice_list.extend([r for r in results if r is not None])

        #print(f"Elapsed time: {(time.time()-start):.2f} s")
        return dice_list

    def dice_obj_mean(*args, **kwargs):
        global tr_dice_list
        tr_dice_list = dice_obj(*args,
                                train_patients=tr_patients,
                                data=preloaded_data,
                                **kwargs)
        return np.mean(tr_dice_list)

    # create results folder
    os.makedirs(results_folder, exist_ok=True)

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
                                                pos_neg_stratify=True, show=True)
    

    if len(tr_patients) <= n_cores: # preload data (faster but heavy in RAM) 
        print(f"{len(tr_patients)} training patients and {n_cores} avalible cores. " \
                      "Preloading data in advance...")
        preloaded_data, _ = load_subjects(data_folder, patients=tr_patients)
    else:   # light on RAM, but slower in time
        preloaded_data = None
        print(f"{len(tr_patients)} training patients and {n_cores} avalible cores. " \
                "Data will be loaded at each iteration...")

    # maximize training DICE    
    optimizer = BayesianOptimization(f=dice_obj_mean, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    params = optimizer.max["params"]
    # tr_dice_list is a global variable defined inside dice_obj_mean
    np.save(os.path.join(results_folder, "tr_dice.npy"), np.array(tr_dice_list))

    # validation
    val_data, _ = load_subjects(data_folder, patients=val_patients)
    val_dice_list = [process_patient((val_data[patient],) + tuple(params[k]
                                                             for k in list(pbounds.keys())))
                    for patient in val_patients]
    np.save(os.path.join(results_folder, "val_dice.npy"), np.array(val_dice_list))



    

if __name__ == '__main__':

    args = parse_args()

    start_time = time.time()
    main(args.data_folder, args.results_folder, args.init_points, args.n_iter, args.n_cores)
    print(f"Elapsed time: {(time.time()-start_time):.2f} s")