"""
NF2FF transformation testing script

Dr Hardie Pienaar
16 April 2016
"""

import FileReading.readFromFile as rff
import NF2FF.NF2FF as nf2ff
import numpy as np

"""General settings"""
filename = "Horizontal_Dipole_Above_Plane.efe"
frequency_block = 0  # Select the first frequency block in the file
pad_factor = 1

# Spherical farfield pattern settings
theta_lim = (-np.pi/2, np.pi/2)
phi_lim = (0, np.pi)
theta_steps = 20
phi_steps = 20

"""Start of script"""
print("\nStarting pyNF2FF\n")

print("Importing nearfield data from "+filename)
f, x, y, z, ex, ey, ez, coord_structure = rff.read_fekonearfield_datafile(filename)

# Assuming an equally spaced 2D grid calculate grid properties (only 1 z-layer is currently permitted)
delta = np.abs(x[1]-x[0])
x_points = coord_structure[1]
y_points = coord_structure[2]

print("File imported with following structure:")
print("Frequencies: "+str(coord_structure[0]))
print("X-Points:    "+str(x_points))
print("Y-Points:    "+str(y_points))
print("Z-Points:    "+str(coord_structure[3]))
print("Delta:       "+str(delta)+" m\n")

print("Extracting block: "+str(frequency_block)+" from imported dataset\n")
x, y, z, ex, ey, ez = rff.read_frequency_block_from_nearfield_dataset(frequency_block, coord_structure, x, y, z,
                                                                      ex, ey, ez)

print("Probing the ether for its fundamental constants")
c0, e0, u0 = nf2ff.get_fundamental_constants()
frequency = f[frequency_block*(x_points*y_points)]
lambda0 = nf2ff.calc_freespace_wavelength(frequency)
wavenumber = nf2ff.calc_freespace_wavenumber(frequency)
print("Frequency:  "+str(frequency/1e6)+" MHz")
print("Wavelength: "+str(np.round(lambda0, 3))+" m")
print("Wavenumber: "+str(np.round(wavenumber, 3))+" rad/m\n")

print("Transforming data into meshgrid")
x_grid = rff.transform_data_coord_to_grid(x_points, y_points, x)
y_grid = rff.transform_data_coord_to_grid(x_points, y_points, y)
z_grid = rff.transform_data_coord_to_grid(x_points, y_points, z)
ex_grid = rff.transform_data_coord_to_grid(x_points, y_points, ex)
ey_grid = rff.transform_data_coord_to_grid(x_points, y_points, ey)
ez_grid = rff.transform_data_coord_to_grid(x_points, y_points, ez)

print("Increasing angular spectrum resolution with a zero padding factor of: "+str(pad_factor))
x_grid, y_grid, ex_grid, ey_grid = nf2ff.pad_nearfield_grid(x_grid, y_grid, ex_grid, ey_grid, pad_factor)

print("Generating k-space")
kx_grid, ky_grid, kz_grid = nf2ff.generate_kspace(x_grid, y_grid, wavenumber)

print("Generating theta phi spherical grid\n")
theta_grid, phi_grid = nf2ff.generate_spherical_theta_phi_grid(theta_steps, phi_steps, theta_lim, phi_lim)

print("Calculating angular spectrum of nearfield..."),
fex_grid = nf2ff.calc_angular_spectrum(ex_grid)
fey_grid = nf2ff.calc_angular_spectrum(ey_grid)
fez_grid = nf2ff.calc_angular_spectrum(ez_grid)
print("[DONE]")

# TODO Interpolate the angular spectrum data onto the spherical coordinate grid (fe_spherical = transform_cartesian to_spherical(kx_grid,ky_grid,fe_grid,wavenumber,theta,phi))

# TODO Define a distance in the definite farfield r = 10000

# TODO Calculate the propagation coeficient (get the right name) C = calc_propagation_coef(wavenumber, distance)

# TODO Calculate Etheta and Ephi (Etheta, Ephi = calc_spherical_components(fex_spherical, fey_spherical, theta, phi))

# TODO Calculate E (E = calc_total_theta_phi_field(Etheta, Ephi))

# TODO Probably need to do some calculations here to get scaling right
