from typing import List
from examples.seismic import Model

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np


def plot_density(model: Model, show: bool = True):
    """
    Plot a two-dimensional density field from a seismic `Model`
    object.

    Parameters
    ----------
    model : Model
        Object that holds the density model.
    """
    rho = 1 / model.b.data
    rho = rho[model.nbl : -model.nbl, model.nbl : -model.nbl]
    domain_size = 1.0e-3 * np.array(model.domain_size)
    extent = [
        model.origin[0],
        model.origin[0] + domain_size[0],
        model.origin[1] + domain_size[1],
        model.origin[1],
    ]

    plt.figure()
    plot = plt.imshow(
        np.transpose(rho),
        cmap="jet",
        vmin=np.min(rho),
        vmax=np.max(rho),
        extent=extent,
    )
    plt.xlabel("X position (km)")
    plt.ylabel("Depth (km)")

    ax = plt.gca()
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", 0.05)
    cbar = plt.colorbar(plot, cax=cax)
    cbar.set_label("Density ($g/$cm$^3$)")

    if show:
        plt.show()


def plot_image2(
    data,
    model: Model,
    vmin=None,
    vmax=None,
    colorbar=True,
    cmap="gray",
    show: bool = True,
):
    """
    Works like `plot_image`, but adjusts the image aspect ratio.

    - Original description:

        Plot image data, such as RTM images or FWI gradients.

    Parameters
    ----------
    data : ndarray
        Image data to plot.
    model: SeismicModel
        Reference model to get the aspect ratio.
    cmap : str
        Choice of colormap. Defaults to gray scale for images as a
        seismic convention.
    """
    plt.figure()

    domain_size = 1.0e-3 * np.array(model.domain_size)
    extent = [
        model.origin[0],
        model.origin[0] + domain_size[0],
        model.origin[1] + domain_size[1],
        model.origin[1],
    ]

    plot = plt.imshow(
        np.transpose(data),
        vmin=vmin or 0.9 * np.min(data),
        vmax=vmax or 1.1 * np.max(data),
        cmap=cmap,
        extent=extent,
    )

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)

    if show:
        plt.show()


def plot_graph(
    xdata: np.ndarray,
    ydata: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str = "",
    show: bool = True,
):
    plt.figure()
    plt.plot(xdata, ydata)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if show:
        plt.show()


def plot_array(
    arr: np.ndarray, xlabel: str, ylabel: str, title: str = "", show: bool = True
):
    x = np.arange(1, 1 + len(arr))
    plot_graph(x, arr, xlabel, ylabel, title, show)
    plt.xticks(x[:: int(np.ceil(len(arr) / 10))])


def plot_trace(
    traces: List[np.ndarray],
    ymin,
    ymax,
    styles: List[str],
    legends: List[str],
    xlabel: str,
    ylabel: str,
    show: bool = True,
):
    plt.figure(figsize=(8, 9))

    for i in range(len(traces)):
        y = np.linspace(ymin, ymax, traces[i].size)
        plt.plot(traces[i], y, styles[i], linewidth=2)

    plt.legend(legends, fontsize=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().invert_yaxis()

    if show:
        plt.show()
