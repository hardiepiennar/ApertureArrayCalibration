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

    grid_data = griddata(np.transpose(coordinates), values, (grid_x, grid_y), method='nearest')

    return grid_x, grid_y, grid_data


def calculate_total_gain(gain_theta, gain_phi):
    """
    Calculates the total gain in dB from its theta and phi components
    :param gain_theta: theta gain in dB
    :param gain_phi: phi gain in dB
    :return: total gain in dB
    """
    gain = 20*np.log10(np.sqrt((10**(gain_theta/20))**2 + (10**(gain_phi/20))**2))
    return gain


def calculate_total_e_field(ex, ey, ez):
    """
    Calculates the total complex e-field from its ex, ey and ez components
    :param ex: complex E-field in x direction
    :param ey: complex E-field in y direction
    :param ez: complex E-field in z direction
    :return: total complex e-field
    """
    e = np.sqrt(ex**2 + ey**2 + ez**2)
    return e


def calc_nf2ff(grid_x, grid_y, grid_z, nearfield_x, nearfield_y):
    """
    Calculates the farfield given the nearfield data
    :param grid_x: 2D matrix with x coords
    :param grid_y: 2D matrix with y coords
    :param grid_z: 2D matrix with z coords
    :param nearfield_x: 2D matrix of complex nearfield_x values
    :param nearfield_y: 2D matrix of complex nearfield_y values
    :param nearfield_z: 2D matrix of complex nearfield_z values
    :return farfield_x, farfield_y, farfield_z: complex farfield 2D matrices
    """

    """Calculate the x and y farfields with the fourier transform"""
    farfield_x = np.fft.fftshift(np.fft.fft2(nearfield_x))
    farfield_y = np.fft.fftshift(np.fft.fft2(nearfield_y))

    """Calculate the z component of the farfield to make a fields transverse to propogation direction"""
    farfield_z = (farfield_x*grid_x + farfield_y*grid_y)/grid_z

    return farfield_x, farfield_y, farfield_z


def transform_cartesian_to_spherical(grid_x, grid_y, grid_z, data_x, data_y, data_z, theta, phi):
    """
    Transform cartesian data to spherical data
    :param grid_x: 2D matrix of data x coordinates
    :param grid_y: 2D matrix of data y coordinates
    :param grid_z: 2D matrix of data z coordinates
    :param data_x: 2D matrix of complex x cartesian data
    :param data_y: 2D matrix of complex y cartesian data
    :param data_z: 2D matrix of complex z cartesian data
    :return data_theta, data_phi: spherical transformed theta and phi directed 2D matrices
    """

    """Calculate theta and phi coordinates"""
    data_theta = []
    data_phi = []
    return data_theta, data_phi


def calc_dft2(x, y, z, data, kx, ky):
    """
    Calculates the disctrete fourier transform for the given 2D data
    :param x: constant spaced position vector in m
    :param y: constant spaced position vector in m
    :param z: position vector in m
    :param data: 2D matrix of complex values to transform
    :param kx: angular spectrum to calculate in deg
    :param ky: angular spectrum to calculate in deg
    :return transformed_data: transformed 2D complex matrix
    """

    # Calculate the dimensions of the kx, ky coordinate space
    start_coord = kx[0]
    num_x_coords = 0
    for i in np.arange(1, len(kx)):
        if kx[i] == start_coord:
            num_x_coords = i
            break
    kx_size = num_x_coords
    ky_size = len(ky)/num_x_coords

    # Calculate the dimensions of the x, y coordinate space
    start_coord = x[0]
    num_x_coords = 0
    for i in np.arange(1, len(x)):
        if x[i] == start_coord:
            num_x_coords = i
            break
    x_size = num_x_coords
    y_size = len(y)/num_x_coords

    # Calculate spacing in grid
    delta_x = x[1] - x[0]
    delta_y = y[x_size] - y[0]

    # Create matrix for transformed data
    transformed_data = np.zeros((ky_size, kx_size), dtype=complex)

    # Convert kx, ky from deg to rad
    kx = kx*np.pi/180
    ky = ky*np.pi/180

    x_sum = 0
    y_sum = 0

    for i in np.arange(len(kx)):
        for i_x in np.arange(len(data[0])):
            for i_y in np.arange(len(data)):
                y_sum += data[i_y][i_x]*np.exp(1j*(kx[i]*i_x*delta_x + ky[i]*i_y*delta_y))
            x_sum += y_sum
            y_sum = 0
        transformed_data[np.floor(i/kx_size)][i % kx_size] += delta_x*delta_y*x_sum
        x_sum = 0

    return transformed_data