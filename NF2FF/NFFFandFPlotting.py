"""
Nearfield and farfield plotting functions. These functions will be untested.

Hardie Pienaar
4 April 2016
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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
        ax.set_ylim(0, 90)
    else:
        ax.set_ylim(-90, 90)
    ax.set_xlim(-180, 180)
    extents = (np.min(phi), np.max(phi), np.min(theta), np.max(theta))
    if zlim[0] == -1 and zlim[1] == -1:
        v_limits = (np.min(gain_grid), np.max(gain_grid))
    else:
        v_limits = (zlim[0], zlim[1])
    v_ticks = np.linspace(v_limits[0], v_limits[1], 11)
    data = gain_grid

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
    data = e

    cax = ax.imshow(data, extent=extents, vmin=v_limits)
    ax.contour(data, v_ticks, extent=extents, vmin=v_limits, colors='k', origin='upper')
    fig.colorbar(cax, ticks=v_ticks, orientation='horizontal')

    ax.set_aspect("auto")
    fig.set_tight_layout(True)


def plot_farfield_kspace_2d(kx, ky, e, title, zlim=[-1, -1]):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.set_title(title)
    ax.set_xlabel("kx [rad/m]")
    ax.set_ylabel("ky [rad/m]")
    ax.set_ylim(np.min(ky), np.max(ky))
    ax.set_xlim(np.min(kx), np.max(kx))
    extents = (np.min(kx), np.max(kx), np.min(ky), np.max(ky))
    if zlim[0] == -1 and zlim[1] == -1:
        v_limits = (np.min(e), np.max(e))
    else:
        v_limits = (zlim[0], zlim[1])
    v_ticks = np.linspace(v_limits[0], v_limits[1], 11)
    data = np.abs(e)

    cax = ax.imshow(data, extent=extents, vmin=v_limits)
    ax.contour(data, v_ticks, extent=extents, vmin=v_limits, colors='k', origin='upper')
    fig.colorbar(cax, ticks=v_ticks, orientation='horizontal')

    ax.set_aspect("auto")
    fig.set_tight_layout(True)


def plot_nearfield_2d_all(x, y, ex, ey, ez, title, zlim=[-1, -1], xlim=[-1, -1], ylim=[-1, -1]):
    s = 1.1
    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(6*s, 8*s))

    data = np.abs(ex)
    ax[0][0].set_title("Mag: "+title)
    ax[0][0].set_ylim(ylim)
    ax[0][0].set_xlim(xlim)
    extents = (np.min(x), np.max(x), np.min(y), np.max(y))
    if zlim[0] == -1 and zlim[1] == -1:
        v_limits = (np.min(data), np.max(data))
    else:
        v_limits = (zlim[0], zlim[1])
    v_ticks = np.linspace(v_limits[0], v_limits[1], 6)

    cax = ax[0][0].imshow(data, extent=extents, vmin=v_limits)
    ax[0][0].contour(data, v_ticks, extent=extents, vmin=v_limits, colors='k', origin='upper')

    data = np.abs(ey)
    ax[1][0].set_ylabel("y [m]")
    ax[1][0].set_ylim(ylim)
    ax[1][0].set_xlim(xlim)
    extents = (np.min(x), np.max(x), np.min(y), np.max(y))
    if zlim[0] == -1 and zlim[1] == -1:
        v_limits = (np.min(data), np.max(data))
    else:
        v_limits = (zlim[0], zlim[1])
    v_ticks = np.linspace(v_limits[0], v_limits[1], 6)

    cax = ax[1][0].imshow(data, extent=extents, vmin=v_limits)
    ax[1][0].contour(data, v_ticks, extent=extents, vmin=v_limits, colors='k', origin='upper')

    data = np.abs(ez)
    ax[2][0].set_xlabel("x [m]")
    ax[2][0].set_ylim(ylim)
    ax[2][0].set_xlim(xlim)
    extents = (np.min(x), np.max(x), np.min(y), np.max(y))
    if zlim[0] == -1 and zlim[1] == -1:
        v_limits = (np.min(data), np.max(data))
    else:
        v_limits = (zlim[0], zlim[1])
    v_ticks = np.linspace(v_limits[0], v_limits[1], 6)

    cax = ax[2][0].imshow(data, extent=extents, vmin=v_limits)
    ax[2][0].contour(data, v_ticks, extent=extents, vmin=v_limits, colors='k', origin='upper')

    data = np.angle(ex)
    ax[0][1].set_title("Ang: "+title)
    ax[0][1].set_ylim(ylim)
    ax[0][1].set_xlim(xlim)
    extents = (np.min(x), np.max(x), np.min(y), np.max(y))
    v_limits = (-np.pi/2, np.pi/2)
    v_ticks = np.linspace(v_limits[0], v_limits[1], 6)

    cax = ax[0][1].imshow(data, extent=extents, vmin=v_limits)
    #ax[0].contour(data, v_ticks, extent=extents, vmin=v_limits, colors='k', origin='upper')

    data = np.angle(ey)
    ax[1][1].set_ylabel("y [m]")
    ax[1][1].set_ylim(ylim)
    ax[1][1].set_xlim(xlim)
    extents = (np.min(x), np.max(x), np.min(y), np.max(y))
    v_limits = (-np.pi/2, np.pi/2)
    v_ticks = np.linspace(v_limits[0], v_limits[1], 6)

    cax = ax[1][1].imshow(data, extent=extents, vmin=v_limits)
    #ax[1].contour(data, v_ticks, extent=extents, vmin=v_limits, colors='k', origin='upper')

    data = np.angle(ez)
    ax[2][1].set_xlabel("x [m]")
    ax[2][1].set_ylim(ylim)
    ax[2][1].set_xlim(xlim)
    extents = (np.min(x), np.max(x), np.min(y), np.max(y))
    v_limits = (-np.pi/2, np.pi/2)
    v_ticks = np.linspace(v_limits[0], v_limits[1], 6)

    cax = ax[2][1].imshow(data, extent=extents, vmin=v_limits)
    #ax[2].contour(data, v_ticks, extent=extents, vmin=v_limits, colors='k', origin='upper')

    fig.set_tight_layout(True)


def plot_farfield_3d_cartesian(theta, phi, ampl, title, zlim=[-1,-1]):
    """
    Plot the data in a 3D cartesian environment
    :param theta: theta coordinate grid
    :param phi: phi coordinate grid
    :param ampl: amplitudes to plot
    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title(title)

    if zlim[0] == -1 and zlim[1] == -1:
        ax.set_zlim(np.min(ampl), np.max(ampl))
    else:
        ampl[ampl < zlim[0]] = zlim[0]
        ampl[ampl > zlim[1]] = zlim[1]
        ax.set_zlim(zlim[0], zlim[1])

    surf = ax.plot_surface(np.rad2deg(theta), np.rad2deg(phi), ampl, rstride=1, cstride=1, cmap=cm.jet,
                           linewidth=0.5, antialiased=True, shade=False)

    ax.set_xlabel("Theta [deg]")
    ax.set_ylabel("Phi [deg]")
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)


def plot_farfield_3d_spherical(theta, phi, ampl, title, zlim=[-1,-1]):
    """
    Plot the data in a 3D spherical environment
    :param theta: theta coordinate grid
    :param phi: phi coordinate grid
    :param ampl: amplitudes to plot
    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax._axis3don = False

    scale = 1.2
    ax.set_xlim(-0.5*scale, 0.5*scale)
    ax.set_ylim(-0.5*scale, 0.5*scale)
    ax.set_zlim(-0.65*scale, -0.15*scale)

    ax.set_title(title)

    ampl -= np.min(ampl)
    ampl /= np.max(ampl)
    x = ampl*np.sin(theta)*np.cos(phi)
    y = ampl*np.sin(theta)*np.sin(phi)
    z = ampl*np.cos(theta)
    z -= np.max(z)
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet,
                           linewidth=0.5, antialiased=True, shade=False)

