#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:17:21 2025

@author: david
"""

import pytest
import hypothesis.strategies as st
from hypothesis import given, settings
import hypothesis.extra.numpy as stnp

import sys
import os
import numpy as np
import SimpleITK as sitk
import matplotlib
from pathlib import Path
import tempfile
import pandas as pd
from contextlib import redirect_stdout
import io

matplotlib.use('Agg')
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                            '..', 'SilentInfarctionSegmentationFLAIR')))


from SilentInfarctionSegmentationFLAIR.utils import DimensionError
from SilentInfarctionSegmentationFLAIR.utils import check_3d
from SilentInfarctionSegmentationFLAIR.utils import get_info
from SilentInfarctionSegmentationFLAIR.utils import get_array_from_image
from SilentInfarctionSegmentationFLAIR.utils import get_image_from_array
from SilentInfarctionSegmentationFLAIR.utils import plot_image
from SilentInfarctionSegmentationFLAIR.utils import orient_image
from SilentInfarctionSegmentationFLAIR.utils import resample_to_reference
from SilentInfarctionSegmentationFLAIR.utils import label_names
from SilentInfarctionSegmentationFLAIR.utils import get_paths_df
from SilentInfarctionSegmentationFLAIR.utils import progress_bar
from SilentInfarctionSegmentationFLAIR.utils import train_val_test_split
from SilentInfarctionSegmentationFLAIR.utils import gaussian_transform
from SilentInfarctionSegmentationFLAIR.utils import cliffs_delta







####################
###  STRATEGIES  ###
####################


@st.composite
def random_orientation_strategy(draw):
    """
    Generate a random orientation string like 'RAS', 'LPI', ...
    """
    axis_directions = {
        'X': ['L', 'R'],
        'Y': ['P', 'A'],
        'Z': ['I', 'S']
    }

    axis_order = draw(st.permutations(['X', 'Y', 'Z']))
    orientation = ''.join(draw(st.sampled_from(axis_directions[ax])) for ax in axis_order)
    
    return orientation


@st.composite
def gauss_noise_strategy_3D(draw):    
    """
    Creates a 3D image with gaussian noise for testing
    """
    origin = draw(st.tuples(*[st.floats(0., 100.)] * 3))
    spacing = draw(st.tuples(*[st.floats(.1, 1.)] * 3))
    direction = tuple([0., 0., 1., 1., 0., 0., 0., 1., 0.])
    size = (draw(st.integers(10, 100)), draw(st.integers(10, 100)), draw(st.integers(10, 100)))
    
    #draw image    
    filter_ = sitk.GaussianImageSource()
    filter_.SetSize(size)
    filter_.SetOrigin(origin)
    filter_.SetSpacing(spacing)
    filter_.SetDirection(direction)
    image = filter_.Execute()

    # random orientation
    orientation = draw(random_orientation_strategy())
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(orientation)
    image = orienter.Execute(image)


    return image


@st.composite
def gauss_noise_strategy_2D(draw):    
    """
    Creates a 2D image with gaussian noise for testing
    """
    origin = draw(st.tuples(*[st.floats(0., 100.)] * 2))
    spacing = draw(st.tuples(*[st.floats(.1, 1.)] * 2))
    direction = tuple([0., 1., 1., 0.])
    size = (draw(st.integers(10, 100)), draw(st.integers(10, 100)))
    
    #draw image    
    filter_ = sitk.GaussianImageSource()
    filter_.SetSize(size)
    filter_.SetOrigin(origin)
    filter_.SetSpacing(spacing)
    filter_.SetDirection(direction)
    image = filter_.Execute()

    return image

@pytest.fixture
def fake_data(tmp_path):
    """
    Creates a temporary data folder for 4 patients containing 3 random images each
    """
    root = tmp_path / "data"
    root.mkdir()

    # create 4 fake patients
    for pid, r in zip(["P1", "P2", "P3", "P4"], [0.1, 0.4, 0.6, 0.9]):
        d = root / pid
        d.mkdir()

        img = sitk.Image(5, 5, sitk.sitkUInt8)
        arr = np.where(np.random.rand(5, 5) < r, 1, 0).astype(np.uint8)
        img = sitk.GetImageFromArray(arr)

        sitk.WriteImage(img, str(d / "GT.nii"))
        sitk.WriteImage(img, str(d / "A.nii"))
        sitk.WriteImage(img, str(d / "B.nii"))

    return root

@st.composite
def non_binary_image_strategy(draw):
    """
    Generate a non-binary SimpleITK image with gray levels from 10 to 15.
    """
    shape = draw(st.tuples(st.integers(5, 20), st.integers(5, 20), st.integers(5, 20)))
    arr = draw(stnp.arrays(dtype=np.int32, shape=shape, elements=st.integers(10, 15)))
    img = sitk.GetImageFromArray(arr)

    return img

@st.composite
def float_array_1d(draw, min_size=5, max_size=200):
    """
    Generate a uniformly distributed random 1D array of floats.
    """
    size = draw(st.integers(min_size, max_size))
    arr = draw(stnp.arrays(
        dtype=np.float64,
        shape=(size,),
        elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)
    ))
    return arr






##################
###  TESTING   ###
##################


@given(gauss_noise_strategy_2D())
@settings(max_examples=5, deadline=None)
def test_check_3d_raise_dimensionerror(image2d):
    """
    Given:
        - gaussian noise SimpleITK 2D image
    Then:
        - apply check_3d 
    Assert that:
        - image2d raises DimensionError
        """
    with pytest.raises(DimensionError):
        check_3d(image2d)
        
        
@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_check_3d_doesnt_raise_exception(image3d):
    """
    Given:
        - gaussian noise SimpleITK 3D image
    Then:
        - apply check_3d 
    Assert that:
        - image3d doesn't raise any exception
        """
    try:
        check_3d(image3d)
    except Exception:
        assert False        
        

@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_get_info_valid_return(image):
    """
    Given:
        - gaussian noise SimpleITK 3D image
    Then:
        - apply get_info 
    Assert that:
        - a 4-items non-empty dict is returned
        - returned size, spacing, origin are 3-elements tuple each
        - returned direction matrix is a 9-element tuple 
    """
    info = get_info(image)
    
    assert isinstance(info, dict)
    assert len(info) == 4
    
    for k, i in info.items():
        assert isinstance(i, tuple)
    
    assert len(info["size"]) == 3
    assert len(info["spacing"]) == 3
    assert len(info["origin"]) == 3
    assert len(info["direction"]) == 9
    
    
@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_get_array_from_image_not_uniform(image):
    """
    Given:
        - gaussian noise SimpleITK 3D image
    Then:
        - apply get_array_from_image 
    Assert that:
        - array is not uniform (gaussian noise)
    """
    
    array = get_array_from_image(image)
    
    assert not np.all(array == array[0,0,0])
    
    
@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_get_array_from_image_same_size(image):
    """
    Given:
        - gaussian noise SimpleITK 3D image
    Then:
        - apply get_array_from_image 
    Assert that:
        - array size is consistent with image size
    """
    
    array = get_array_from_image(image)
    
    assert array.shape == image.GetSize()


@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_get_image_from_array_valid_return(image):
    """
    Given:
        - gaussian noise SimpleITK image
    Then:
        - get array from image
        - get image from array
    Assert:
        - the combined effect of the two functions, with image as reference, returns exactly image
    """
    assert get_image_from_array(get_array_from_image(image))


@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_plot_image_valid_return(image):
    """
    Given:
        - gaussian noise SimpleITK 3D image
    Then:
        - plot the image with plot_image
    Assert that:
        - a 3-items dict is returned
        - returned size, spacing, aspects are 3-elements tuple each
    """
    plot_info = plot_image(image)
    
    assert isinstance(plot_info, dict)
    assert len(plot_info) == 3
    
    for k, i in plot_info.items():
        assert isinstance(i, tuple)
    
    assert len(plot_info["size"]) == 3
    assert len(plot_info["spacing"]) == 3
    assert len(plot_info["aspects"]) == 3
    
    
    
@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_plot_image_spacings_aspects(image):
    """
    Given:
        - gaussian noise SimpleITK 3D image
    Then:
        - plot the image with plot_image
    Assert that:
        - if all spacings are the same then all aspects are the same
        - if all spacings are not the same then all aspects are not the same
    """
    plot_info = plot_image(image)
        
    if len(set(plot_info["spacing"])) == 1:
        assert len(set(plot_info["aspects"])) == 1
    else:
        assert len(set(plot_info["aspects"])) != 1
          
        
@given(gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_plot_image_raise_indexerror(image):
    """
    Given:
        - gaussian noise SimpleITK 3D image
    Then:
        - apply plot_image with out of bounds coordinates
    Assert that:
        - IndexError is raised
    """
    oob = [(image.GetSize()[0]+1, 0, 0),
           (0, image.GetSize()[1]+1, 0),
           (0, 0, image.GetSize()[2]+1),
           (-image.GetSize()[0]-1, 0, 0),
           (0, -image.GetSize()[1]-1, 0),
           (0, 0, -image.GetSize()[2]-1)]
    
    for i in range(3):
        with pytest.raises(IndexError):
            plot_image(image, xyz=oob[i])
           
            
@given(gauss_noise_strategy_3D(), random_orientation_strategy())
@settings(max_examples=5, deadline=None)
def test_orient_image_valid_return(image, new_orientation):
    """
    Given:
        - gaussian noise SimpleITK 3D image
        - a random orientation string
    Then:
        - orient the image with orient_image
    Assert that:
        - a SimpleITK.Image is returned
        - if the new orientation is equal to the old then nothing changes
        - if the new orientation is not equal to the old one then a different
        image is returned
    """
    old_orientation = sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(image.GetDirection())
    oriented_image_old = orient_image(image, old_orientation)
    oriented_image_new = orient_image(image, new_orientation)

    assert isinstance(oriented_image_old, sitk.Image)  
    assert get_info(oriented_image_old) == get_info(image)
    if new_orientation != old_orientation:
        assert get_info(oriented_image_new) != get_info(image)

        
@given(gauss_noise_strategy_3D(), gauss_noise_strategy_3D())
@settings(max_examples=5, deadline=None)
def test_resample_to_reference_valid_return(image, reference):
    """
    Given:
        - 2 gaussian noise SimpleITK images
    Then:
        - resample image to reference
    Assert that:
        - a SimpleITK.Image is returned
        - size, spacing, origin, direction of the resampled image are the same
        as the reference
    """
    image_rs = resample_to_reference(image, reference)
    
    assert isinstance(image_rs, sitk.Image)
    assert get_info(image_rs) == get_info(reference)


def test_get_paths_df_expected_result(tmp_path):
    """
    Given:
        - a temporary folder containing some files with different extensions
    Then:
        - apply get_paths_df with a filter of extensions
    Assert that:
        - DataFrame contains exactly the expected files with correct paths
    """
    f1 = tmp_path / "a.txt"
    f1.write_text("hello")
    f2 = tmp_path / "b.csv"
    f2.write_text("world")
    f3 = tmp_path / "c.txt"
    f3.write_text("!")

    df = get_paths_df(str(tmp_path), extensions=[".txt"])

    assert isinstance(df, pd.DataFrame)
    assert str(tmp_path) in df.index

    assert "a.txt" in df.columns
    assert "c.txt" in df.columns
    assert "b.csv" not in df.columns 
    
    assert df.loc[str(tmp_path), "a.txt"] == str(f1)
    assert df.loc[str(tmp_path), "c.txt"] == str(f3)


def test_get_paths_df_empty_folder(tmp_path):
    """
    Given:
        - an empty folder
    Then:
        - apply get_paths_df
    Assert that:
        - returned DataFrame is empty
    """
    df = get_paths_df(str(tmp_path), extensions=[".txt"])
    assert df.empty


def test_get_paths_df_nonlist_extension(tmp_path):
    """
    Given:
        - a folder with one .txt file
        - extension passed as a string instead of list
    Then:
        - apply get_paths_df
    Assert that:
        - function still works (converts to list internally)
    """
    f = tmp_path / "file.txt"
    f.write_text("data")

    df = get_paths_df(str(tmp_path), extensions=".txt")
    assert "file.txt" in df.columns
    assert df.loc[str(tmp_path), "file.txt"] == str(f)


def test_label_names_expected_result():
    """
    Given:
        - a temporary text file
    Then:
        - read the labels from the file
    Assert that:
        - a dict is returned
        - the returned dict matches the expected results
    """
    test_content = """1 Left-Thalamus
2 Left Thalamus
# comment line
3 Right-Hippocampus
invalid line
"""

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(test_content)
        tmp_path = tmp.name

    result = label_names(tmp_path)

    expected = {
        1: "Left-Thalamus",
        2: "Left",
        3: "Right-Hippocampus"
    }

    assert isinstance(result, dict)
    assert result == expected

    Path(tmp_path).unlink()


@given(st.integers(0, 99), st.integers(2,99))
@settings(max_examples=5, deadline=None)
def test_progress_bar_basic_output(i, n):
    """
    Given:
        - a call to progress_bar with iteration = i
            and total = i + n > i + 1
    Then:
        - capture the printed output
    Assert that:
        - prefix and iteration counters appear in the output
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        progress_bar(iteration=i, total=i+n, prefix="Test")
    captured = buf.getvalue()

    assert "Test" in captured
    assert f"{i+1}/{i+n}" in captured


def test_train_val_test_split_valid_return(fake_data):
    """
    Given:
        - a temporary dataset containing multiple patients
        - a call to train_val_test_split with validation_fraction > 0 and test_fraction > 0
    Then:
        - the function returns three collections of patient IDs
    Assert that:
        - the returned outputs are lists of strings
    """
    tr, val, ts = train_val_test_split(
        data_folder=str(fake_data),
        validation_fraction=0.25,
        test_fraction=0.25)

    assert isinstance(tr, list)
    assert isinstance(val, list)
    assert isinstance(ts, list)

    assert all(isinstance(p, str) for p in tr)
    assert all(isinstance(p, str) for p in val)
    assert all(isinstance(p, str) for p in ts)


def test_no_overlap(fake_data):
    """
    Given:
        - a temporary dataset containing multiple patients
        - a call to train_val_test_split with validation_fraction > 0 and test_fraction > 0
    Then:
        - the function returns three collections of patient IDs
    Assert that:
        - there is no overlap between the three sets
    """
    tr, val, ts = train_val_test_split(
        data_folder=str(fake_data),
        validation_fraction=0.25,
        test_fraction=0.25)

    set_tr = set(tr)
    set_val = set(val)
    set_ts = set(ts)

    assert set_tr.isdisjoint(set_val)
    assert set_tr.isdisjoint(set_ts)
    assert set_val.isdisjoint(set_ts)


def test_all_patients_used(fake_data):
    """
    Given:
        - a temporary dataset containing multiple patients
        - a call to train_val_test_split with validation_fraction > 0 and test_fraction > 0
    Then:
        - the function returns three collections of patient IDs
    Assert that:
        - every patient in the folder is assigned to at least one set
    """
    tr, val, ts = train_val_test_split(
        data_folder=str(fake_data),
        validation_fraction=0.25,
        test_fraction=0.25)

    assigned = set(tr + val + ts)

    existing = set([d for d in os.listdir(fake_data)
                    if os.path.isdir(os.path.join(fake_data, d))])

    assert existing == assigned


def test_raise_all_errors():
    """
    Given:
        - invalid folder
        - invalid validation and test fractions
        - a folder without .nii files
    Then:
        - repeatedly call train_val_test_split with bad parameters
    Assert that:
        - correct exceptions are raised each time
    """

    with pytest.raises(FileNotFoundError):
        train_val_test_split("folder_that_does_not_exist")


    dummy_folder = Path("dummy_test_folder")
    dummy_folder.mkdir(exist_ok=True)

    try:
        with pytest.raises(ValueError):
            train_val_test_split(str(dummy_folder), validation_fraction=-0.1)

        with pytest.raises(ValueError):
            train_val_test_split(str(dummy_folder), test_fraction=-0.5)

        with pytest.raises(ValueError):
            train_val_test_split(str(dummy_folder),
                                 validation_fraction=0.7,
                                 test_fraction=0.4)

        with pytest.raises(RuntimeError):
            train_val_test_split(str(dummy_folder), validation_fraction=0.1)

    finally:
        dummy_folder.rmdir()



@given(
    non_binary_image_strategy(),
    st.floats(min_value=0, max_value=20),   # mean
    st.floats(min_value=0.1, max_value=10)  # std > 0
)
@settings(max_examples=5, deadline=None)
def test_gaussian_transform_valid_return(img, mean, std):
    """
    Given:
        - a non-binary SimpleITK image (gray levels 10-15)
        - mean and std for the Gaussian weighting
    Then:
        - apply gaussian_transform
    Assert:
        - output is a SimpleITK image
        - output has same size, spacing, origin, direction as input
        - all output pixel values are non-negative
    """

    enhanced = gaussian_transform(img, mean, std)
    
    arr_in = sitk.GetArrayFromImage(img)
    arr_out = sitk.GetArrayFromImage(enhanced)
    
    assert isinstance(enhanced, sitk.Image)
    assert enhanced.GetSize() == img.GetSize()
    assert enhanced.GetSpacing() == img.GetSpacing()
    assert enhanced.GetOrigin() == img.GetOrigin()
    assert enhanced.GetDirection() == img.GetDirection()
    assert np.all(arr_out >= 0)
    assert arr_out.shape == arr_in.shape


@given(non_binary_image_strategy(),
    st.floats(min_value=0, max_value=20),
    st.floats(min_value=0.1, max_value=10))
@settings(max_examples=5, deadline=None)
def test_gaussian_transform_peak_behavior(img, mean, std):
    """
    Given:
        - a non-binary SimpleITK image
        - mean and std for Gaussian weighting
    Then:
        - apply gaussian_transform
    Assert:
        - pixels close to 'mean' are enhanced more than pixels far from 'mean'
    """

    arr_in = sitk.GetArrayFromImage(img)
    enhanced = gaussian_transform(img, mean, std, return_float=True)
    arr_out = sitk.GetArrayFromImage(enhanced)
    
    # Pick one pixel close to mean, one far
    diff = np.abs(arr_in - mean)
    close_idx = np.unravel_index(np.argmin(diff), diff.shape)
    far_idx = np.unravel_index(np.argmax(diff), diff.shape)
    
    assert arr_out[close_idx] >= arr_out[far_idx]


@given(float_array_1d(), float_array_1d())
@settings(max_examples=10, deadline=None)
def test_cliffs_delta_valid_return(x, y):
    """
    Given:
        - two 1D numeric arrays
    Then:
        - compute Cliff's Delta
    Assert that:
        - a float is returned
        - result is bounded in [-1, 1]
    """
    delta = cliffs_delta(x, y)

    assert isinstance(delta, float)
    assert -1.0 <= delta <= 1.0


@given(float_array_1d(), float_array_1d())
@settings(max_examples=10, deadline=None)
def test_cliffs_delta_antisymmetry(x, y):
    """
    Given:
        - two 1D numeric arrays
    Then:
        - compute Cliff's Delta in both orders
    Assert that:
        - delta(x, y) == -delta(y, x)
    """
    d_xy = cliffs_delta(x, y)
    d_yx = cliffs_delta(y, x)
    assert np.isclose(d_xy, -d_yx)


@given(float_array_1d())
@settings(max_examples=10, deadline=None)
def test_cliffs_delta_same_distribution_zero(x):
    """
    Given:
        - a 1D numeric array
    Then:
        - compute Cliff's Delta with itself
    Assert that:
        - delta is approximately zero
    """
    delta = cliffs_delta(x, x)
    assert np.isclose(delta, 0.0)


@given(float_array_1d())
@settings(max_examples=10, deadline=None)
def test_cliffs_delta_perfect_separation(x):
    """
    Given:
        - an array x
    Then:
        - create y strictly smaller than x
    Assert that:
        - Cliff's Delta is close to +1
    """
    y = x - 1000
    delta = cliffs_delta(x, y)
    assert np.isclose(delta, 1.0)

