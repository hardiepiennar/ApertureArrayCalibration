"""
Nearfield to farfield conversion methods
Nearfield Antenna measurements - Dan Slater

Hardie Pienaar
29 Feb 2016
"""

import numpy as np
from scipy import interpolate
from scipy.interpolate import griddata
from scipy import integrate
import FileReading.readFromFile as rff


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


def calc_nf2ff(freq, x_grid, y_grid, ex_grid, ey_grid, distance, theta_grid, phi_grid, verbose=False, pad_factor=4):
    """
    Calculate the farfield e_theta and e_phi at the given distance from grid data
    :param freq: frequency in Hz
    :param x_grid: 2D grid matrix of x coordinates in m
    :param y_grid: 2D grid matrix of y coordinates in m
    :param ex_grid: 2D grid matrix of x directed nearfield values V/m
    :param ey_grid: 2D grid matrix of y directed nearfield values V/m
    :param theta_grid: 2D grid matrix of theta points to calculate in rad
    :param phi_grid: 2D grid matrix of phi points to calculate in rad
    :param verbose: Give information during method
    :param pad_factor: amount of padding to add during angular spectrum calculation
    :return e_theta, e_phi: theta and phi farfield values for given coordinates
    """
    if verbose:
        print("Probing the ether for its fundamental constants")
    frequency = freq
    lambda0 = calc_freespace_wavelength(frequency)
    wavenumber = calc_freespace_wavenumber(frequency)
    if verbose:
        print("Frequency:  "+str(frequency/1e6)+" MHz")
        print("Wavelength: "+str(np.round(lambda0, 3))+" m")
        print("Wavenumber: "+str(np.round(wavenumber, 3))+" rad/m\n")

    if verbose:
        print("Increasing angular spectrum resolution with a zero padding factor of: "+str(pad_factor))
    x_grid_pad, y_grid_pad, ex_grid_pad, ey_grid_pad, _ = pad_nearfield_grid(x_grid, y_grid,
                                                                             ex_grid, ey_grid, [],
                                                                             pad_factor)

    if verbose:
        print("Generating k-space")
    kx_grid, ky_grid, _ = generate_kspace(x_grid_pad, y_grid_pad, wavenumber)

    if verbose:
        print("Calculating angular spectrum of nearfield..."),
    fex_grid = calc_angular_spectrum(ex_grid_pad)
    fey_grid = calc_angular_spectrum(ey_grid_pad)
    if verbose:
        print("[DONE]")

    if verbose:
        print("Interpolating angular spectrum data onto spherical grid..."),
    fex_spherical = interpolate_cartesian_to_spherical(kx_grid, ky_grid, fex_grid, wavenumber,
                                                       theta_grid[0], phi_grid[:, 0])*np.sqrt(2)*len(fex_grid)**2
    fey_spherical = interpolate_cartesian_to_spherical(kx_grid, ky_grid, fey_grid, wavenumber,
                                                       theta_grid[0], phi_grid[:, 0])*np.sqrt(2)*len(fex_grid)**2
    if verbose:
        print("[DONE]")

    if verbose:
        print("Calculating theta and phi components..."),

    C = calc_propagation_coef(frequency, distance)
    e_theta, e_phi = transform_cartesian_to_spherical(theta_grid, phi_grid, fex_spherical, fey_spherical)
    e_theta *= C
    e_phi *= C
    if verbose:
        print("[DONE]\n")

    return e_theta, e_phi


def calc_nf2ff_from_coord_data(freq, x_points, y_points, x, y, ex, ey, theta_steps, phi_steps, theta_lim, phi_lim,
                               verbose=False, pad_factor=4):
    """
    Calculate e_theta and e_phi given coord data
    :param freq: frequency in Hz
    :param x: x coordinates in m
    :param y: y coordinates in m
    :param ex: x directed nearfield values V/m
    :param ey: y directed nearfield values V/m
    :param theta: theta points to calculate in rad
    :param phi: phi points to calculate in rad
    :param verbose: Give information during method
    :param pad_factor: amount of padding to add during angular spectrum calculation
    :return e_theta, e_phi: theta and phi farfield values for given coordinates
    """

    lambda0 = calc_freespace_wavelength(freq)
    wavenumber = calc_freespace_wavenumber(freq)
    if verbose:
        print("Frequency:  "+str(freq/1e6)+" MHz")
        print("Wavelength: "+str(np.round(lambda0, 3))+" m")
        print("Wavenumber: "+str(np.round(wavenumber, 3))+" rad/m\n")

    if verbose:
        print("Transforming data into meshgrid")
    x_grid = rff.transform_data_coord_to_grid(x_points, y_points, x)
    y_grid = rff.transform_data_coord_to_grid(x_points, y_points, y)
    ex_grid = rff.transform_data_coord_to_grid(x_points, y_points, ex)
    ey_grid = rff.transform_data_coord_to_grid(x_points, y_points, ey)

    if verbose:
        print("Generating theta phi spherical grid\n")
    theta_grid, phi_grid = generate_spherical_theta_phi_grid(theta_steps, phi_steps, theta_lim, phi_lim)

    if verbose:
        print("Starting NF2FF transform...")
    r = 10000
    e_theta, e_phi = calc_nf2ff(freq, x_grid, y_grid, ex_grid, ey_grid, r, theta_grid, phi_grid,
                                pad_factor=pad_factor, verbose=verbose)
    return theta_grid, phi_grid, e_theta, e_phi


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

    farfield = np.fft.ifftshift(np.fft.ifft2(nearfield))

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


def calc_radiation_intensity(e_theta, e_phi):
    """
    Calculates the radiation intensity (power radiated per solid angle) W/rad^2
    :param e_theta: e_field in theta direction
    :param e_phi: e_field in phi direction
    :return U, radiation intensity:
    """
    c0, e0, u0 = get_fundamental_constants()
    z0 = np.sqrt(u0/e0)
    U = (float(1)/(2*z0))*(e_theta*np.conj(e_theta)+e_phi*np.conj(e_phi))
    return np.real(U)


def calc_empl(x1, x2):
    """
    Calculates the equivelent multipath level between 2 signals. The default 0.5 factor has been omitted assuming that
    x1 or x2 is absolute correct.
    :param x1: matrix of values
    :param x2: matrix of values
    :return: equivelent multipath level in dB
    """
    empl = np.abs(np.abs(x1) - np.abs(x2))
    empl_db = 20*np.log10(np.abs(empl))
    return empl_db


def calc_radiated_power(theta_grid, phi_grid, U_grid):
    """
    Calculates the total radiated power W over the given theta phi grid
    :param theta_grid: theta 2D matrix grid of angles in rad
    :param phi_grid: phi 2D matrix grid of angles in rad
    :param U_grid: Radiation intensity grid W/rad^2
    :return P_rad: Radiated power W
    """
    I = U_grid
    I_interp = interpolate.interp2d(phi_grid[:, 0], theta_grid[0], I)
    I_func = lambda phi, theta: I_interp(phi, theta)*np.cos(theta)
    theta_start, theta_stop = np.min(theta_grid), np.max(theta_grid)
    phi_start, phi_stop = lambda theta: np.min(phi_grid), lambda theta: np.max(phi_grid)
    P_rad = integrate.dblquad(I_func, theta_start, theta_stop, phi_start, phi_stop)

    return P_rad


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
    fe_spherical_real_func = interpolate.RectBivariateSpline(kx_grid[0], ky_grid[:, 0], np.real(np.conj(fe_grid)))
    fe_spherical_imag_func = interpolate.RectBivariateSpline(kx_grid[0], ky_grid[:, 0], np.imag(np.conj(fe_grid)))

    fe_spherical = np.reshape(np.zeros(len(phi)*len(theta), dtype=complex), (len(phi), len(theta)))

    for phi_i in np.arange(len(phi)):
        for theta_i in np.arange(len(theta)):
            y_coord = wavenumber*np.sin(theta[theta_i])*np.cos(phi[phi_i])
            x_coord = wavenumber*np.sin(theta[theta_i])*np.sin(phi[phi_i])
            fe_spherical[phi_i][theta_i] = complex(fe_spherical_real_func(x_coord, y_coord)) + \
                1j*complex(fe_spherical_imag_func(x_coord, y_coord))
    return fe_spherical


def add_position_noise(x_grid, y_grid, z_grid, x_n_amp, y_n_amp):
    """
    Add noise to data by probing interpolated z-grid at noisy coordinates and rebuilding z grid as an equally spaced
    grid with the noisy probe points
    :param x_grid: 2D matrix grid of x coordinates
    :param y_grid: 2D matrix grid of y coordinates
    :param z_grid: 2D matrix grid of function to make noisy
    :param x_n_amp: max x noise amplitude
    :param y_n_amp: max y noise amplitude
    :return z_n_grid: noisy z_grid
    """

    z_func = interpolate.interp2d(x_grid[0], y_grid[:, 0], z_grid)
    x_n_grid = x_grid + (np.random.rand(len(x_grid), len(x_grid[0]))*2 - 1)*x_n_amp
    y_n_grid = y_grid + (np.random.rand(len(y_grid), len(y_grid[0]))*2 - 1)*y_n_amp
    x_coords = np.reshape(x_n_grid, (1, len(x_n_grid)*len(x_n_grid[0])))[0]
    y_coords = np.reshape(y_n_grid, (1, len(y_n_grid)*len(y_n_grid[0])))[0]
    z_n_coords = np.zeros(len(x_coords))
    for i in np.arange(len(x_coords)):
        z_n_coords[i] = z_func(x_coords[i], y_coords[i])
    z_n_grid = np.reshape(z_n_coords, (len(x_grid), len(x_grid[0])))
    return z_n_grid


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