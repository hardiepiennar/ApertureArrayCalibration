"""
Nearfield to farfield conversion methods
Nearfield Antenna measurements - Dan Slater

Hardie Pienaar
29 Feb 2016
"""

import numpy as np
from scipy import interpolate
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


def calc_nf2ff(nearfield_x, nearfield_y):
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
    farfield_x = np.fft.fftshift(np.fft.ifft2(nearfield_x))
    farfield_y = np.fft.fftshift(np.fft.ifft2(nearfield_y))

    return farfield_x, farfield_y

def calc_kgrid(x_grid, y_grid, padding):
    ksize = len(x_grid) + padding
    kx = np.arange(ksize) - (float(ksize-1)/2)
    ky = np.arange(ksize) - (float(ksize-1)/2)
    kx_grid, ky_grid = np.meshgrid(kx, ky)

    dx = np.abs(x_grid[0][1] - x_grid[0][0])
    N = len(x_grid) + padding
    scaling = 2*np.pi*(float(1)/N)*(float(1)/dx)
    kx_grid *= scaling
    ky_grid *= scaling

    return kx_grid, ky_grid



def transform_cartesian_to_spherical(grid_x, grid_y, grid_z, data_x, data_y, data_z):
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

    print(np.shape(grid_x))
    print(np.shape(grid_y))
    print(np.shape(grid_z))
    print(np.shape(data_x))
    print(np.shape(data_y))
    print(np.shape(data_z))
    r = np.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
    theta = np.arccos(grid_z/r)
    phi = np.arctan2(grid_y, grid_x)

    """Calculate the function values in the ETheta and EPhi directions"""
    farfield_theta = data_x*np.cos(theta)*np.cos(phi) + data_y*np.cos(theta)*np.sin(phi)
    #farfield_theta = spherical_x*np.cos(phi_coords) + spherical_y*np.sin(phi_coords)
    farfield_phi = data_x*(-1*np.sin(phi)) + data_y*np.cos(phi)
    #farfield_phi = np.cos(theta_coords)*(-spherical_x*(np.sin(phi_coords)) + spherical_y*np.cos(phi_coords))

    return theta, phi, farfield_theta, farfield_phi


def calc_dft2(x, y, z, data):
    """
    Calculates the disctrete fourier transform for the given 2D data
    :param x: constant spaced position vector in m
    :param y: constant spaced position vector in m
    :param z: position vector in m
    :param data: 2D matrix of complex values to transform
    :return transformed_data: transformed 2D complex matrix
    """

    # Calculate the dimensions of the x, y coordinate space
    x_size = float(len(x[0]))
    y_size = float(len(x))

    # Calculate spacing in grid
    delta_x = float(x[0][1] - x[0][0])
    delta_y = delta_x

    # Create matrix for transformed data
    transformed_data = np.zeros((y_size, x_size), dtype=complex)

    x_sum = 0
    y_sum = 0

    for ky in np.arange(len(data)):
        for kx in np.arange(len(data[0])):
            transformed_data[ky][kx] = np.sum(delta_x*delta_y*data*np.exp(-2j*np.pi*((kx-(x_size-1)/2)*delta_x/x_size +
                                                                               (ky-(y_size-1)/2)*delta_y/y_size)))
            y_sum = 0

    return transformed_data