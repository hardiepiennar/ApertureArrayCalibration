"""
Nearfield and farfield plotting functions. These functions will be untested.

Hardie Pienaar
4 April 2016
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_phi_cut(theta, gain, title, ylim=[-80, 10]):
    """
    Cartesian plot of the thea values for a given phi
    :param theta: theta vector associated with gain vector
    :param gain: gain vector on constant phi
    :param title: title
    :param ylim: y limits
    :return:
    """
    plt.figure()
    plt.title(title)
    plt.xlabel("Theta [degrees]")
    plt.ylabel("Gain [dB]")
    plt.plot(theta, gain)
    plt.xlim(0, 180)
    plt.ylim(ylim[0], ylim[1])
    plt.grid(True)


def plot_farfield_2d(theta, phi, gain_grid, title, zlim=[-1, -1], only_top_hemisphere=False):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.set_title(title)
    ax.set_xlabel("Phi [degrees]")
    ax.set_ylabel("Theta [degrees]")
    if only_top_hemisphere:
        ax.set_ylim(90, 180)
    else:
        ax.set_ylim(0, 180)
    ax.set_xlim(0, 360)
    extents = (np.min(phi), np.max(phi), np.min(theta), np.max(theta))
    if zlim[0] == -1 and zlim[1] == -1:
        v_limits = (np.min(gain_grid), np.max(gain_grid))
    else:
        v_limits = (zlim[0], zlim[1])
    v_ticks = np.linspace(v_limits[0], v_limits[1], 11)
    data = np.transpose(gain_grid)

    cax = ax.imshow(data, extent=extents, vmin=v_limits)
    ax.contour(data, v_ticks, extent=extents, vmin=v_limits, colors='k', origin='upper')
    fig.colorbar(cax, ticks=v_ticks, orientation='horizontal')

    ax.set_aspect("auto")
    fig.set_tight_layout(True)


def plot_nearfield_2d(x, y, e, title, zlim=[-1, -1]):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_xlim(np.min(x), np.max(x))
    extents = (np.min(x), np.max(x), np.min(y), np.max(y))
    if zlim[0] == -1 and zlim[1] == -1:
        v_limits = (np.min(e), np.max(e))
    else:
        v_limits = (zlim[0], zlim[1])
    v_ticks = np.linspace(v_limits[0], v_limits[1], 11)
    data = np.abs(np.transpose(e))

    cax = ax.imshow(data, extent=extents, vmin=v_limits)
    ax.contour(data, v_ticks, extent=extents, vmin=v_limits, colors='k', origin='upper')
    fig.colorbar(cax, ticks=v_ticks, orientation='horizontal')

    ax.set_aspect("auto")
    fig.set_tight_layout(True)

