#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2025-08-11 17:18:09

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter1d

from SilentInfarctionSegmentationFLAIR.utils import get_array_from_image
from SilentInfarctionSegmentationFLAIR.utils import downsample_array


def plot_histogram(
    image,
    bins=None,
    title="Gray level histogram",
    save_path=None,
    no_bkg=False,
    show=True,
    normalize=False,
    alpha=1,
    downsampling=None,
    ax=None,
):
    """
    Plot the histogram of a gray‑scale 3D SimpleITK image.

    Parameters
    ----------
    image : SimpleITK.Image
        Input image.
    bins : int or str, optional
        Number of bins or binning strategy. If None, uses the maximum
        gray level.
    title : str, optional
        Title of the plot.
    save_path : str, optional
        Path where the figure is saved.
    no_bkg : bool, optional
        If True, exclude gray level 0.
    show : bool, optional
        Whether to display the plot.
    normalize : bool, optional
        If True, normalize histogram counts.
    alpha : float, optional
        Transparency of the bars.
    downsampling : float or None, optional
        Percentage of voxels to sample.
    ax : matplotlib.axes.Axes, optional
        Axis on which to draw.

    Returns
    -------
    hist : tuple
        Tuple ``(counts, bin_edges)`` from ``np.histogram``.
    """
    if ax is None:
        fig, ax = plt.subplots()

    arr = get_array_from_image(image).flatten()

    if no_bkg:
        arr = arr[arr != 0]

    if downsampling is not None:
        arr = downsample_array(arr, perc=downsampling)

    if bins is None:
        bins = int(max(np.unique(arr)))

    hist = np.histogram(arr, bins=bins, density=normalize)
    counts, bin_edges = hist

    width = bin_edges[1] - bin_edges[0]
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if arr.size > 0:
        ax.bar(
            centers,
            counts,
            alpha=alpha,
            width=width,
            align="center"
        )
        xlabel = "Gray level (excluding black)" if no_bkg else "Gray level"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.set_title(title)

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

    return hist


def plot_multiple_histograms(
    images,
    bins=None,
    labels=None,
    title="Gray level histograms",
    legend_title="",
    save_path=None,
    no_bkg=False,
    alpha=0.5,
    normalize=False,
    downsampling=None,
    show=True,
    ax=None,
):
    """
    Plot histograms for multiple 3D SimpleITK images.

    Parameters
    ----------
    images : list of SimpleITK.Image
        Images to process.
    bins : int, list, or None
        Number of bins. If list, must match ``images`` length.
    labels : list of str, optional
        Labels for each histogram.
    title : str, optional
        Plot title.
    legend_title : str, optional
        Title of the legend.
    save_path : str, optional
        Path where the figure is saved.
    no_bkg : bool, optional
        If True, exclude gray level 0.
    alpha : float, optional
        Transparency of bars.
    normalize : bool, optional
        If True, normalize histogram counts.
    downsampling : float or None, optional
        Percentage of voxels to sample.
    show : bool, optional
        Whether to display the plot.
    ax : matplotlib.axes.Axes, optional
        Axis on which to draw.

    Returns
    -------
    hist_tables : list of tuple
        List of histograms ``(counts, bin_edges)``.
    """
    if labels is None:
        labels = [f"Image {i+1}" for i in range(len(images))]

    if ax is None and (show or save_path is not None):
        fig, ax = plt.subplots()

    hist_tables = []

    for idx, (image, label) in enumerate(zip(images, labels)):
        arr = get_array_from_image(image).flatten()

        if no_bkg:
            arr = arr[arr != 0]

        if downsampling is not None:
            arr = downsample_array(arr, perc=downsampling)

        if isinstance(bins, list):
            my_bins = bins[idx]
        else:
            my_bins = bins

        if my_bins is None:
            my_bins = int(max(np.unique(arr)))

        hist = np.histogram(arr, bins=my_bins, density=normalize)
        counts, bin_edges = hist

        width = bin_edges[1] - bin_edges[0]
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if arr.size > 0 and (show or save_path is not None):
            ax.bar(
                centers,
                counts,
                alpha=alpha,
                label=label,
                width=width,
                align="center"
            )
            ax.set_xlabel(
                "Gray level (excluding black)" if no_bkg else "Gray level"
            )
            ax.set_ylabel("Density" if normalize else "Counts")
            ax.set_title(title)
            ax.legend(title=legend_title)

        hist_tables.append(hist)

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.close()
    return hist_tables


def histogram_stats(hist, q1=25, q2=75):
    """
    Compute descriptive statistics from a histogram.

    Parameters
    ----------
    hist : tuple
        Tuple ``(counts, bin_edges)`` from ``np.histogram``.
    q1 : float, optional
        First percentile (default 25).
    q2 : float, optional
        Second percentile (default 75).

    Returns
    -------
    tuple
        ``(mean, p1, p2, variance, skewness, kurtosis)``.
    """
    counts, bin_edges = hist
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mean = np.average(centers, weights=counts)

    cdf = np.cumsum(counts).astype(float)
    cdf /= cdf[-1]

    p1 = np.interp(q1 / 100, cdf, centers)
    p2 = np.interp(q2 / 100, cdf, centers)

    variance = np.average((centers - mean) ** 2, weights=counts)

    expanded = np.repeat(centers, counts.astype(int))
    skewness = skew(expanded)
    kurt = kurtosis(expanded)

    return mean, p1, p2, variance, skewness, kurt


def gaussian_smooth_histogram(hist, sigma=3, ax=None):
    """
    Smooth a histogram using a Gaussian kernel.

    Parameters
    ----------
    hist : tuple
        Tuple ``(counts, bin_edges)`` from ``np.histogram``.
    sigma : float, optional
        Standard deviation of the Gaussian kernel.
    ax : matplotlib.axes.Axes, optional
        Axis on which to draw.

    Returns
    -------
    smooth_hist : tuple
        Smoothed histogram ``(counts, bin_edges)``.
    """
    counts, bins = hist
    centers = (bins[:-1] + bins[1:]) / 2

    smooth_counts = gaussian_filter1d(counts, sigma=sigma)

    if ax is not None:
        ax.plot(
            centers,
            smooth_counts,
            "r-",
            linewidth=2,
            alpha=0.7,
            label=f"Gaussian smoothing (σ={sigma})"
        )

    return smooth_counts, bins


def mode_and_rhwhm(hist, ax=None):
    """
    Compute the mode and right-side half-width at half-maximum (HWHM).

    Parameters
    ----------
    hist : tuple
        Tuple ``(counts, bin_edges)`` from ``np.histogram``.
    ax : matplotlib.axes.Axes, optional
        Axis on which to draw.

    Returns
    -------
    mode : float
        Most frequent gray level.
    rhwhm : float
        Right-side half-width at half-maximum.
    """
    counts, bins = hist
    centers = (bins[:-1] + bins[1:]) / 2

    mode = centers[np.argmax(counts)]
    half_max = np.max(counts) / 2

    below = centers[counts < half_max]
    right = below[below > mode]
    right_hm = np.min(right)

    rhwhm = right_hm - mode

    if ax is not None:
        ax.axvline(
            mode,
            linestyle="--",
            color="red",
            linewidth=2,
            label=f"Mode ({mode:.1f})"
        )
        ax.annotate(
            "",
            xy=(mode, half_max),
            xytext=(right_hm, half_max),
            arrowprops=dict(
                arrowstyle="<->",
                color="red",
                lw=2,
                shrinkA=0,
                shrinkB=0,
            ),
        )

        x_text = (mode + right_hm) / 2
        y_text = half_max - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])

        ax.text(
            x_text,
            y_text,
            f"Right-side HWHM\n({rhwhm:.1f})",
            color="black",
            ha="center",
            va="top",
        )

    return mode, rhwhm
