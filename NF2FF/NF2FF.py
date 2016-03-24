"""
Nearfield to farfield conversion methods
Nearfield Antenna measurements - Dan Slater

Hardie Pienaar
29 Feb 2016
"""

import numpy as np
from scipy.interpolate import griddata


def calc_recommended_sampling_param(d, m, p, z, lambd):
    """
    Calculate the recommended sampling parameters for a nearfield measurement
    (Not sure if sample delta and no of samples per ray can be trusted)
    :param d: AUT aperture diameter
    :param m: AUT maximum farfield angle from bore-sight in rad
    :param p: Probe antenna aperture
    :param z: Probe to AUT separation
    :param lambd: wavelength
    :return: probe_gain, scan_width, no_samples_per_ray, sample_delta
    """

    probe_gain = 0.5* (np.pi / (np.tan(m / 1.03))) ** 2
    scan_width = d + p + 2 * z * np.tan(m)
    no_samples_per_ray = 0.5*lambd/(np.sin(m) * scan_width)
    sample_delta = 0.5*(no_samples_per_ray/(no_samples_per_ray+1))*lambd/np.sin(m)

    return probe_gain, scan_width, no_samples_per_ray, sample_delta


def calc_nearfield_bounds(f, d):
    """
    Caclulate the start and stop distances of the radiating nearfield
    :param f: frequency in Hz
    :param d: aut aperture
    :return: start, stop distance of the radiating nearfield
    """

    c = 3e8
    lambd = c/f
    start = lambd*3
    stop = (2*d**2)/lambd

    return start, stop


def transform_data_coord_to_grid(coordinates, values, resolution):
    """
    Transforms the given coordinated values to a grid format
    :param coordinates: (x, y) coordinates of the values
    :param values: values at (x,y) coordinates
    :param resolution: number of points in return grid [x_axis, y_axis]
    :return grid_x, grid_y, grid_data: data in a grid format
    """
    grid_y, grid_x = np.mgrid[np.min(coordinates[1]):np.max(coordinates[1]):1j*resolution[1],
                              np.min(coordinates[0]):np.max(coordinates[0]):1j*resolution[0]]

    grid_data = griddata(np.transpose(coordinates), values, (grid_x, grid_y), method='linear')

    return grid_x, grid_y, grid_data
