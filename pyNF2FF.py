"""
NF2FF transformation testing script

Dr Hardie Pienaar
16 April 2016
"""

import FileReading.readFromFile as rff
import NF2FF.NF2FF as nf2ff
import numpy as np
import NF2FF.NFFFandFPlotting as plotting
import matplotlib.pyplot as plt

"""General settings"""
filename = "WaveguideHorn80deg.efe"
separation = 0.0899377374000528
frequency_block = 0  # Select the first frequency block in the file
pad_factor = 4


# Spherical farfield pattern settings
scan_angle = np.deg2rad(10)
theta_lim = (-np.pi/2+scan_angle, np.pi/2-scan_angle)
phi_lim = (0, np.pi)
theta_steps = 41
phi_steps = 41

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
print("Delta:       "+str(delta)+" m")
print("Separation:  "+str(round(separation, 3))+" m\n")

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
ex_grid = rff.transform_data_coord_to_grid(x_points, y_points, ex)
ey_grid = rff.transform_data_coord_to_grid(x_points, y_points, ey)
ez_grid = rff.transform_data_coord_to_grid(x_points, y_points, ez)

print("Increasing angular spectrum resolution with a zero padding factor of: "+str(pad_factor))
x_grid_pad, y_grid_pad, ex_grid_pad, ey_grid_pad, ez_grid_pad = nf2ff.pad_nearfield_grid(x_grid, y_grid,
                                                                                         ex_grid, ey_grid, ez_grid,
                                                                                         pad_factor)

print("Generating k-space")
kx_grid, ky_grid, kz_grid = nf2ff.generate_kspace(x_grid_pad, y_grid_pad, wavenumber)

print("Generating theta phi spherical grid\n")
theta_grid, phi_grid = nf2ff.generate_spherical_theta_phi_grid(theta_steps, phi_steps, theta_lim, phi_lim)

print("Calculating angular spectrum of nearfield..."),
fex_grid = nf2ff.calc_angular_spectrum(ex_grid_pad)
fey_grid = nf2ff.calc_angular_spectrum(ey_grid_pad)
fez_grid = -(kx_grid*fex_grid + ky_grid*fey_grid)/kz_grid
print("[DONE]")

print("Interpolating angular spectrum data onto spherical grid..."),
fex_spherical = nf2ff.interpolate_cartesian_to_spherical(kx_grid, ky_grid, fex_grid, wavenumber,
                                                         theta_grid[0], phi_grid[:, 0])
fey_spherical = nf2ff.interpolate_cartesian_to_spherical(kx_grid, ky_grid, fey_grid, wavenumber,
                                                         theta_grid[0], phi_grid[:, 0])
fez_spherical = nf2ff.interpolate_cartesian_to_spherical(kx_grid, ky_grid, fez_grid, wavenumber,
                                                         theta_grid[0], phi_grid[:, 0])
print("[DONE]")

print("Calculating theta and phi components..."),
r = 10000
C = nf2ff.calc_propagation_coef(frequency, r)
e_theta, e_phi = nf2ff.transform_cartesian_to_spherical(theta_grid, phi_grid, fex_spherical, fey_spherical)
e_theta = C*e_theta
e_phi = C*e_phi
e = nf2ff.calculate_total_e_field(e_theta, e_phi,0)
print("[DONE]")

mag_e = 20*np.log10(np.abs(e))
norm_mag_e = mag_e - np.max(mag_e)

z_upper = np.max([np.max(20*np.log10(np.abs(ex_grid))),
                 np.max(20*np.log10(np.abs(ey_grid))),
                 np.max(20*np.log10(np.abs(ez_grid)))])
range = 80
plotting.plot_nearfield_2d_all(x_grid, y_grid, ex_grid, ey_grid, ez_grid, "Nearfield")

plotting.plot_farfield_kspace_2d_all(kx_grid, ky_grid, fex_grid, fey_grid, fez_grid, "Farfield")

plotting.plot_farfield_3d_spherical(theta_grid, phi_grid, mag_e, "Farfield")
plotting.plot_farfield_3d_cartesian(theta_grid, phi_grid, mag_e, "Farfield")

plt.show()

# TODO: Check scaling with some principle cuts
# TODO: Program comparison vectors
# TODO: Import real farfield
# TODO: Compare farfields
# TODO: Decide on comparison methodology
# TODO: Decide on errors to focus on
# TODO: Write error injection code
# TODO: Write parameter sweep code
# TODO: Run comparisons for all error types
# TODO: Refactor NFFFandFPlotting
