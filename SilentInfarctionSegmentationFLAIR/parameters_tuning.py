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
from bayes_opt import BayesianOptimization
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

from SilentInfarctionSegmentationFLAIR.utils import (to_n_bit,
                                                     orient_image,
                                                     resample_to_reference,
                                                     get_array_from_image,
                                                     get_image_from_array,
                                                     label_names)
from SilentInfarctionSegmentationFLAIR.segmentation import (get_mask_from_segmentation,
                                                            evaluate_voxel_wise)
from SilentInfarctionSegmentationFLAIR.refinement import gaussian_transform
from SilentInfarctionSegmentationFLAIR import threshold
from SilentInfarctionSegmentationFLAIR import refinement_step


def parse_args():
    description="Evaluates DICE coefficient over all images contained in the 'data' folder"

    parser = argparse.ArgumentParser(description=description)

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

    args = parser.parse_args()

    return args


pbounds={"gamma": (0.1, 5),
        "alpha": (1, 10),
        "beta": (0, 2),
        "extend_dilation_radius": (0, 5),
        "n_std": (0, 5),
        "min_diameter": (0, 3),
        "surround_dilation_radius": (0, 5),
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

def main(paths_df, init_points, n_iter):

    def load_subjects(paths_df):
        data = {}
        for root in paths_df.index:
            flair = to_n_bit(orient_image(sitk.ReadImage(paths_df.loc[root, flair_file]), "RAS"), 8)
            t1 = to_n_bit(sitk.ReadImage(paths_df.loc[root, t1_file]), 8)
            t1 = resample_to_reference(t1, flair)
            gt = resample_to_reference(sitk.ReadImage(paths_df.loc[root, gt_file]), flair)
            gt = sitk.Cast(gt, sitk.sitkUInt8)
            segm = resample_to_reference(sitk.ReadImage(paths_df.loc[root, segm_file]), flair)

            gm_mask = get_mask_from_segmentation(segm, gm_labels)
            wm_mask = get_mask_from_segmentation(segm, wm_labels)
            wm_pve = resample_to_reference(sitk.ReadImage(paths_df.loc[root, wm_pve_file]), flair)
            gm_pve = resample_to_reference(sitk.ReadImage(paths_df.loc[root, gm_pve_file]), flair)
            csf_pve = resample_to_reference(sitk.ReadImage(paths_df.loc[root, csf_pve_file]), flair)

            data[root] = {
                "flair": flair, "t1": t1, "gt": gt,
                "gm_mask": gm_mask, "wm_mask": wm_mask,
                "wm_pve": wm_pve, "gm_pve": gm_pve, "csf_pve": csf_pve}
        return data

    def dice_obj(gamma, alpha, beta, extend_dilation_radius, n_std,
                 min_diameter, surround_dilation_radius, min_points):
        gc.collect()
        dice_list = []
        for root in paths_df.index:
            flair = data[root]["flair"]
            t1 = data[root]["t1"]
            gt = data[root]["gt"]
            gm_mask = data[root]["gm_mask"]
            wm_mask = data[root]["wm_mask"]
            wm_pve = data[root]["wm_pve"]
            gm_pve = data[root]["gm_pve"]
            csf_pve = data[root]["csf_pve"]

            # weighted sum of flair and gaussian transformed t1
            wm_t1 = sitk.Mask(t1, wm_mask)
            stats = sitk.StatisticsImageFilter()
            stats.Execute(wm_t1)
            wm_mean = stats.GetMean()
            wm_std = stats.GetSigma()
            t1_gauss = gaussian_transform(t1, mean=wm_mean, std=alpha*wm_std)
            t1_gauss_arr = beta * get_array_from_image(t1_gauss)
            t1_gauss_scaled = get_image_from_array(t1_gauss_arr,
                                                    sitk.Cast(t1_gauss, sitk.sitkUInt16))
            image = sitk.Cast(flair, sitk.sitkUInt16) + t1_gauss_scaled
            image = to_n_bit(image, 8)

            # process
            gm = sitk.Mask(image, gm_mask)
            mask = threshold.main(image, gm,
                                gamma=gamma, show=False, verbose=False)
            mask = refinement_step.main(
                                mask,
                                image=image,
                                pves=[wm_pve, gm_pve, csf_pve],
                                labels_dict=labels_dict,
                                keywords_to_remove=keywords_to_remove,
                                min_diameter=min_diameter,
                                extend_dilation_radius=extend_dilation_radius,
                                n_std=n_std,
                                surround_dilation_radius=surround_dilation_radius,
                                min_points=int(min_points),
                                verbose=False)
            
            # evaluate
            dice = evaluate_voxel_wise(mask, gt)["vw-DSC"]
            dice_list.append(dice)

        return np.mean(dice_list)

    print("Loading data...")
    data = load_subjects(paths_df)
    optimizer = BayesianOptimization(f=dice_obj, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)


if __name__ == '__main__':

    args = parse_args()

    # organize all images in a df
    paths_list = []
    for root, _, files in os.walk("data"):
        for file in files:
            if file.endswith(".nii"):
                paths_list.append((root,file))
    paths_df = pd.DataFrame()
    for root, file in paths_list:
        paths_df.loc[root,file] = os.path.join(root,file)

    # extract label names
    labels_dict = label_names(label_name_file)


    main(paths_df, args.init_points, args.n_iter)
