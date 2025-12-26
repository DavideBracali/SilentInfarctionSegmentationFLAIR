# Silent Infarction FLAIR Segmentation

## Overview
!! Fare

## Contents

The **SilentInfarctionSegmentationFLAIR** package is organized into two main components:

- **Scripts** — provide a structured and modular interface to run the complete segmentation workflow, handle I/O operations, and expose the command-line interface (CLI):
  - *flair_t1_sum*: Combines FLAIR and T1 images using a gaussian-transformed T1 weighted sum.
  - *threshold*: Applies an adaptive threshold using histogram mode and rHWHM.
  - *refinement_step*: Post-processes a threshold-based lesion mask applying connected components, geometric, PVE-based and anatomical filters.
  - *tuning_alpha_beta*: Optimizes the two parameters necessary to integrate FLAIR and T1 information.
  - *tuning_gamma_rs*: Optimizes gamma, the main segmentation parameter, and the five parameters necessary to run the refinement step.

- **Modules** — contain the core functionality of the pipeline:
  - *histograms*: Methods to compute and extract information from gray level histograms.
  - *segmentation*: Methods to apply and evaluate segmentation algorithms.
  - *refinement*: Methods to refine the initial threshold segmentation.
  - *utils*: Methods to perform general operations such as image plotting, gray level transformations and train-validation-test splits.

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

To execute test routines, please install ```pytest``` and ```hypothesis```. On bash:
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
│   ├── flair.nii
│   ├── t1.nii
│   ├── segmentation.nii
│   ├── gm_pve.nii
│   ├── wm_pve.nii
│   ├── csf_pve.nii
│   └── gt.nii
│
├── Patient002/
│   ├── flair.nii
│   ├── t1.nii
│   ├── segmentation.nii
│   ├── gm_pve.nii
│   ├── wm_pve.nii
│   ├── csf_pve.nii
│   └── gt.nii
|
├──FreeSurferColorLUT.txt 
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

- **flair_t1_sum** — computes the weighted sum of FLAIR and
  Gaussian‑transformed T1.
- **threshold** — applies the GM‑guided adaptive thresholding.
- **refinement_step** — refines the thresholded mask using geometric,
  PVE‑based and anatomical constraints.

To run any of these modules individually, execute:

```bash
python -m SilentInfarctionSegmentationFLAIR.<the_module_you_want_to_execute>
```

Each module provides its own ```--help``` message describing the required
inputs and optional parameters.


### Testing
To execute all test routines, move to the project folder and simply run:
```bash
pytest
```
or
```bash
python -m pytest
```

## Output description
!! Fare

## Parameter description
!! Fare

## Parameter optimization
!! Fare

## Evaluate test set
!! Fare

## License
!! Fare