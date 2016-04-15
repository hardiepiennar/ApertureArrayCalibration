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


def generate_kspace(grid_x, grid_y, wavenumber):
    """
    Generates kspace for angular spectrum
    :param grid_x: x grid points
    :param grid_y: y grid points
    :param wavenumber: freespace wavenumber in rad/m
    :return kx_grid, ky_grid, kz_grid: wavenumber domain grids
    """

    no_points = len(grid_x)
    delta = np.abs(grid_x[0][1] - grid_x[0][0])

    kx = np.arange(no_points) - float(no_points-1)/2
    ky = np.arange(no_points) - float(no_points-1)/2
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    scaling = 2*np.pi*(float(1)/(delta*no_points))
    kx_grid *= scaling
    ky_grid *= scaling

    kz_grid = np.sqrt(wavenumber**2 - kx_grid**2 - ky_grid**2 + 0j)

    return kx_grid, ky_grid, kz_grid


def calc_angular_spectrum(nearfield):
    """
    Calculates the angular spectrum of a 2D nearfield grid
    :param nearfield:
    :return farfield: angular spectrum
    """

    farfield = np.fft.fftshift(np.fft.ifft2(nearfield))

    return farfield

def generate_spherical_theta_phi_grid(theta_steps, phi_steps, theta_lim, phi_lim):
    """
    Generates a theta phi grid with the given step sizes and limits
    :param dtheta: theta step size in rad
    :param dphi: phi step size in rad
    :param theta_lim: theta limit tuple in rad
    :param phi_lim: phi limit tuple in rad
    :return theta_grid, phi_grid:
    """
    theta = np.linspace(theta_lim[0], theta_lim[1], theta_steps)
    phi = np.linspace(phi_lim[0], phi_lim[1], phi_steps)

    theta_grid, phi_grid = np.meshgrid(theta, phi)

    return theta_grid, phi_grid


def transform_cartesian_to_spherical(theta_grid, phi_grid, data_x, data_y):
    """
    Transform cartesian data to spherical data
    :param theta: 2D matrix of theta coordinates
    :param phi: 2D matrix of phi coordinates
    :param data_x: 2D matrix of complex x cartesian data at theta-phi coordinate
    :param data_y: 2D matrix of complex y cartesian data at theta-phi coordinate
    :return e_theta, e_phi: spherical transformed theta and phi directed 2D matrices
    """

    """Calculate the function values in the ETheta and EPhi directions"""
    e_theta = data_x*np.cos(phi_grid) + data_y*np.sin(phi_grid)
    e_phi = np.cos(theta_grid)*(-1*data_x*np.sin(phi_grid) + data_y*np.cos(phi_grid))

    return e_theta, e_phi


def get_fundamental_constants():
    """
    Simple method to return widely used fundamental constants
    :return c0 m/s, e0 F/m, u0: H/m
    """
    c0 = 299792458       # m/s
    e0 = 8.8541878176e-12  # F/m
    u0 = np.pi*4e-7      # H/m

    return c0, e0, u0


def calc_freespace_wavelength(frequency):
    """
    Calculates the freespace wavelength from the frequency
    :return lambda0 m
    """

    c0, _, _ = get_fundamental_constants()
    lambda0 = c0/frequency

    return lambda0


def calc_freespace_wavenumber(frequency):
    """
    Calculates the freespace wavenumber from the frequency
    :return wavenumber rad/m
    """

    c0, _, _ = get_fundamental_constants()
    lambda0 = calc_freespace_wavelength(frequency)
    wavenumber = 2*np.pi/lambda0

    return wavenumber


def calc_propagation_coef(freq, distance):
    """
    :param freq: frequency in Hz
    :param distance: distance in m
    :return: propagation coefficient
    """
    k0 = calc_freespace_wavenumber(freq)
    C = 1j*(k0*np.exp(-1j*k0*distance))/(2*np.pi*distance)

    return C


def pad_nearfield_grid(grid_x, grid_y, nearfield_x, nearfield_y, nearfield_z, pad_factor):
    """
    :param grid_x: x grid points
    :param grid_y: y grid points
    :param nearfield_x: x nearfield grid points
    :param nearfield_y: y nearfield grid points
    :param pad_factor: the amount of padding rows and columns to add
    :return: padded data
    """
    padding = np.ceil(len(grid_x)*(pad_factor-1))
    grid_x = np.pad(grid_x, (0, padding), 'constant')
    grid_y = np.pad(grid_y, (0, padding), 'constant')
    nearfield_x = np.pad(nearfield_x, (0, padding), 'constant')
    nearfield_y = np.pad(nearfield_y, (0, padding), 'constant')
    nearfield_z = np.pad(nearfield_z, (0, padding), 'constant')

    return grid_x, grid_y, nearfield_x, nearfield_y, nearfield_z

def interpolate_cartesian_to_spherical(kx_grid, ky_grid, fe_grid, wavenumber, theta, phi):
    """
    :param kx_grid: wavenumber x grid
    :param ky_grid: wavenumber y grid
    :param fe_grid: angular spectrum grid
    :param wavenumber: wavenumber
    :param theta: theta points to calculate in rad
    :param phi: phi points to calculate in rad
    :return: angular spectrum in spherical grid
    """
    fe_spherical_real_func = interpolate.RectBivariateSpline(kx_grid[0], ky_grid[:, 0], np.real(fe_grid))
    fe_spherical_imag_func = interpolate.RectBivariateSpline(kx_grid[0], ky_grid[:, 0], np.imag(fe_grid))

    fe_spherical = np.reshape(np.zeros(len(phi)*len(theta), dtype=complex), (len(phi), len(theta)))

    for phi_i in np.arange(len(phi)):
        for theta_i in np.arange(len(theta)):
            x_coord = wavenumber*np.sin(theta[theta_i])*np.cos(phi[phi_i])
            y_coord = wavenumber*np.sin(theta[theta_i])*np.sin(phi[phi_i])
            fe_spherical[phi_i][theta_i] = complex(fe_spherical_real_func(x_coord, y_coord)[0]) - \
                1j*complex(fe_spherical_imag_func(x_coord, y_coord)[0])

    return fe_spherical

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