# Silent Infarction FLAIR Segmentation

## Overview
!! Fare

## Contents

The **SilentInfarctionSegmentationFLAIR** package is organized into two main components:

- **Scripts** — provide a structured and modular interface to run the complete segmentation workflow, handle I/O operations, and expose the command-line interface (CLI):
  - [flair_t1_sum](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/SilentInfarctionSegmentationFLAIR/flair_t1_sum.py): Combines FLAIR and T1 images using a gaussian-transformed T1 weighted sum.
  - [threshold](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/SilentInfarctionSegmentationFLAIR/threshold.py): Applies an adaptive threshold using histogram mode and rHWHM.
  - [refinement_step](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/SilentInfarctionSegmentationFLAIR/refinement_step.py): Post-processes a threshold-based lesion mask applying connected components, geometric, PVE-based and anatomical filters.
  - [tuning_alpha_beta](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/SilentInfarctionSegmentationFLAIR/tuning_alpha_beta.py): Optimizes the two parameters necessary to integrate FLAIR and T1 information.
  - [tuning_gamma_rs](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/SilentInfarctionSegmentationFLAIR/tuning_gamma_rs.py): Optimizes gamma, the main segmentation parameter, and the five parameters necessary to run the refinement step.

- **Modules** — contain the core functionality of the pipeline:
  - [histograms](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/SilentInfarctionSegmentationFLAIR/histograms.py): Methods to compute and extract information from gray level histograms.
  - [segmentation](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/SilentInfarctionSegmentationFLAIR/segmentation.py): Methods to apply and evaluate segmentation algorithms.
  - [refinement](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/SilentInfarctionSegmentationFLAIR/refinement.py): Methods to refine the initial threshold segmentation.
  - [utils](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/SilentInfarctionSegmentationFLAIR/utils.py): Methods to perform general operations such as image plotting, gray level transformations and train-validation-test splits.

This structure keeps the computational logic clean making the package easier to maintain and modify.

!! Qui descrivi nel dettaglio la pipeline

## Prerequisites

Supported python versions:   ![Python version](https://img.shields.io/badge/python-3.10|3.11|3.12|3.13-blue.svg)

This package requires:
- ```numpy```
- ```pandas```
- ```SimpleITK```
- ```matplotlib```
- ```seaborn```
- ```scipy```
- ```bayesian-optimization```
- ```pyyaml```

This package is equipped with test routines. To run them, ```pytest``` and ```hypothesis``` need to be installed.
Installation instructions are available at: [PyTest](https://docs.pytest.org/en/6.2.x/getting-started.html), [Hypothesis](https://docs.pytest.org/en/6.2.x/getting-started.html)



## Installation
Using bash, change directory to the folder where you want to store the package. Then clone the repository from GitHub:

```bash
git clone https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR.git
```

The package will be downloaded in a subfolder named ```../SilentInfarctionSegmentationFLAIR/```. Install the package using ```pip```:

```bash
pip install SilentInfarctionSegmentationFLAIR/
```
or
```bash
pip install -e SilentInfarctionSegmentationFLAIR/
```
in case you want to install the package in editable mode.

### Testing

First, please install ```pytest``` and ```hypothesis```. On bash:
```bash
pip install pytest hypothesis
```

## Usage

### Download example data

!!! Fare

### Prepare your own data
To process your own data, images need to satisfy the following requisites:
- All images that refer to the same imaging session must be contained in the same folder.
- Each imaging session must provide a set of *.nii* images:
  - A FLAIR image (named *FLAIR.nii* by default, if you use a different name please modify the ```files/flair``` entry in the *config.yaml* file).
  - An anatomical segmentation labeled image of integer (named *aseg.auto_noCCseg.nii* by default, if you use a different name please modify the ```files/segmentation``` entry in the *config.yaml* file). Each anatomical tissue is identified by an integer label. By default FastSurfer labels are used, however different labels can be used to identify gray matter and white matter. In such case, please modify the ```labels/gm``` and ```labels/wm``` entries in the *config.yaml* file
  - A partial volume map for gray matter (named *gm_pve.nii* by default, if you use a different name please modify the ```files/gm_pve``` entry in the *config.yaml* file).
  - A partial volume map for white matter (named *wm_pve.nii* by default, if you use a different name please modify the ```files/wm_pve``` entry in the *config.yaml* file).
  - A partial volume map for cerebrospinal fluid (named *csf_pve.nii* by default, if you use a different name please modify the ```files/csf_pve``` entry in the *config.yaml* file).
  - Optionally, a T1-weighted image (named *T1ontoFLAIR.nii* by default, if you use a different name please modify the ```files/t1``` entry in the *config.yaml* file). If not provided, the package will use the FLAIR image.
  - Optionally, a ground truth binary image (named *GT.nii* by default, if you use a different name please modify the ```files/gt``` entry in the *config.yaml* file). If provided, the package will evaluate the segmentation and print evaluation metrics.
- Multiple imaging sessions must be stored as subfolders of the same folder (e.g. *data/patient1* and *data/patient2*). Images of the same type must have the same file name (e.g. FLAIR images must be named ```FLAIR.nii``` for all patients).
- Optionally, the algorithm will remove voxels where it is anatomically impossible to find lesions. It does so by eliminating segmentation voxels containing certain keywords. However segmentation labels are integer. To use this feature, you need to specify a relationship between integer labels and tissue name. By default, this information is contained in *data/FreeSurferColorLUT.txt* and is valid for both FastSurfer and FreeSurfer segmentations. Feel free to specify a different file by changing the ```files/label_name``` entry in the *config.yaml* file.

If don't want to modify the *config.yaml* file, then your data directory should look like:
```
data/
│
├── Patient001/
│   ├── FLAIR.nii
│   ├── T1ontoFLAIR.nii
│   ├── aseg.auto_noCCseg.nii
│   ├── pve_gm.nii
│   ├── pve_wm.nii
│   ├── pve_csf.nii
│   └── GT.nii
│
├── Patient002/
│   ├── FLAIR.nii
│   ├── T1ontoFLAIR.nii
│   ├── aseg.auto_noCCseg.nii
│   ├── pve_gm.nii
│   ├── pve_wm.nii
│   ├── pve_csf.nii
│   └── GT.nii
|
└── FreeSurferColorLUT.txt 
```
### Process one or more imaging sessions

To process one or multiple imaging session, use the CLI (command-line interface) by running the following line:
```bash
SilentInfarctionSegmentationFLAIR/ --data_folder='path_to_data_directory'
``` 
or
```bash
SilentInfarctionSegmentationFLAIR/ -i 'path_to_data_directory'
``` 
This will process each individual subfolder of the input directory as a separate case. Additional parameters can be provided:
- `--params_path`  
  Path to the YAML file containing segmentation parameters  
  *(default: params.yaml)*

- `--results_folder`  
  Output directory where results will be saved  
  *(default: results/)*

- `--verbose`  
  Print detailed progress information.

- `--show`  
  Display intermediate plots.

- `--version`  
  Print the installed package version.

### Running individual modules

The recommended workflow is to run the full pipeline through the main `SilentInfarctionSegmentationFLAIR` command. However, each processing step can also be executed independently.

The following modules can be run as standalone components:

- [flair_t1_sum](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/SilentInfarctionSegmentationFLAIR/flair_t1_sum.py) — computes the weighted sum of FLAIR and
  Gaussian‑transformed T1.
- [threshold](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/SilentInfarctionSegmentationFLAIR/threshold.py) — applies the GM‑guided adaptive thresholding.
- [refinement_step](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/SilentInfarctionSegmentationFLAIR/refinement_step.py) — refines the thresholded mask using geometric,
  PVE‑based and anatomical constraints.

To run any of these modules individually, execute:

```bash
python -m SilentInfarctionSegmentationFLAIR.<the_script_you_want_to_execute>
```

Each script provides its own ```--help``` message describing the required
inputs and optional parameters.


### Testing
To execute all test routines (located in the [testing](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/testing) directory), simply run:
```bash
pytest
```
or
```bash
python -m pytest
```

## Output description
The **SilentInfarctionSegmentationFLAIR** package will save the following outputs in the directory specified by the ```--results_folder``` argument:
- *histogram_FLAIR.png*: Tissue histograms (normalized to unit area) for gray matter, white matter and cerebrospinal fluid in the FLAIR image.
- *histogram_T1.png*: Tissue histograms (normalized to unit area) for gray matter, white matter and cerebrospinal fluid in the T1-weighted image.
- *histogram_image.png*: Tissue histograms (normalized to unit area) for gray matter, white matter and cerebrospinal fluid in the FLAIR and gaussian-transformed T1 weighted sum.
- *image.nii*: NIfTI image of FLAIR and gaussian-transformed T1 weighted sum.
- *image.png*: 2-dimensional sections of the FLAIR and gaussian-transformed T1 weighted sum.
- *thr.png*: gray matter histogram where mode, rHWHM (!!! hai spiegato cos'è?) and gray level threshold are highlighted.
- *thr_mask.nii*: NIFTI image of the thresholded input image.
- *thr_mask.png*: 2-dimensional sections of the thresholded input image.
- *segmentation.nii*: NIfTI image of the final refined segmentation.
- *segmentation.png*: 2-dimensional sections of the thresholded input image.

## Parameters
### Parameters description
This segmentation algorithm requires 8 parameters to specify:
- *alpha* and *beta* (floats): to regulate the weighted sum between FLAIR and gaussian-transformed T1 images. *alpha* controls the width of the Gaussian applied to the T1 intensities, while *beta* controls how strongly the transformed T1 contributes to the final image.
- *gamma* (float): this is the most important segmentation parameter, as it defines the value of the initial threshold (!!! come descritto nella parte dove spiego l'algoritmo).
- *extend_dilation_radius* and *n_std* (floats): to control the lesion extension in the refinement step. After the initial segmentation each connected component is dilated with a box kernel with radius equal to *extend_dilation_radius* millimeters. Gray levels that are *n_std* standard deviations within the mean of each component are included in the initial thresholded mask.
- *min_diameter* (float): specifies, in millimeters, the minimum diameter of the connected components to be considered as lesions in the refined step.
- *surround_dilation_radius* (float): in the refinement step each connected components receives a certain score based on the prevalent tissue in the neighborhood of each component. This parameter specifies, in millimeters, the radius of the box kernel of the dilation applied to each candidate lesion.
- *min_points* (int): the minimum score that a candidate lesion must have collected during the refinement step to be present in the final segmentation.

### Parameter optimization
An initial set of optimized parameters is available in the [params.yaml](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/params.yaml) file. This set of parameters was optimized over a training set of 33 sets of images, a validation set of 16 sets of images and evaluated on a test set of 6 images, obtaining a mean DICE of !!!! statistiche!! Anche sensitivity e specificity e precision.

All parameters were optimized using [bayesian-optimization](https://github.com/bayesian-optimization/BayesianOptimization):
-*alpha* and *beta* were tuned by maximizing an objective function over the training set that rewards separation between gray matter and lesions histograms (increasing true positives), while punishing overlap between gray matter and white matter (potentially increasing false negatives).
- *extend_dilation_radius*, *n_std*, *min_diameter*, *surround_dilation_radius* and *min_points* were tuned by maximizing the average DICE coefficient (after the refinement step) over the training set. The initial thresholded mask depends on *gamma*, so this optimization was performed separately for each candidate value of *gamma*.
- *gamma*, being the most influential parameter of the algorithm, was chosen as the value that produced the maximum average DICE coefficient over the validation set. The proposed parameters in [params.yaml](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/params.yaml) returned a validation average DICE of !!! aggiungere media e IQR. !! Immagini??? forse non qui

If you prefer to tune your own parameters, it is possible to do so using the [tuning_alpha_beta](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/SilentInfarctionSegmentationFLAIR/tuning_alpha_beta.py) and [tuning_gamma_rs](https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR/tree/main/SilentInfarctionSegmentationFLAIR/tuning_gamma_rs.py) scripts. Be aware that both scripts will require several hours or even days depending on the performances your computer. 

To optimize *alpha* and *beta* run:

```bash
python -m SilentInfarctionSegmentationFLAIR.tuning_alpha_beta --data_folder='path_to_data_directory'
```

Then you can optimize all the other parameters by running:

```bash
python -m SilentInfarctionSegmentationFLAIR.tuning_gamma_rs --data_folder='path_to_data_directory'
```

The last script can be run for specific values of *alpha* and *beta* by specifying the ```--alpha``` and ```--beta``` arguments. If not provided, the script will automatically use the previously optimized *params_alpha_beta.yaml* file created by the first script.

Both scripts will search for an existing train-validation-test split. If not found, a 60% - 30% - 10% split will be created, stratified according to the positives / total voxel ratios in the ground truth file.

Both scripts can exploit parallel processing by specifying the ```n_cores``` argument. Specifying an high value will decrease computation time, however the required RAM space will linearly increase as multiple images will be loaded at the same time.

A ```--help``` argument can be used in both scripts to describe the optional parameters.

## Evaluate test set

To evaluate the algorithm over a test set, run:

```bash
python -m SilentInfarctionSegmentationFLAIR.evaluate_test_set --data_folder='path_to_data_directory'
```

This script will search for an existing train-validatoin-test split. If not found, every imaging session in the data folder will be evaluated. 
A ```--help``` argument can be used to describe the optional parameters.

## License
!! Fare