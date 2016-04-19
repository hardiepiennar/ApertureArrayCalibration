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
filename_nearfield = "WaveguideHorn80degx2sep.efe"
filename_farfield = "WaveguideHorn.ffe"
separation = 0.0899377374000528
frequency_block = 0  # Select the first frequency block in the file
pad_factor = 4


# Spherical farfield pattern settings
scan_angle = np.deg2rad(0)
theta_lim = (-np.pi/2+scan_angle, np.pi/2-scan_angle)
phi_lim = (0, np.pi)
theta_steps = 101
phi_steps = 101

"""Start of script"""
print("\nStarting pyNF2FF\n")

print("Importing nearfield data from " + filename_nearfield)
f, x, y, z, ex, ey, ez, coord_structure = rff.read_fekonearfield_datafile(filename_nearfield)

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

print("Generating theta phi spherical grid\n")
theta_grid, phi_grid = nf2ff.generate_spherical_theta_phi_grid(theta_steps, phi_steps, theta_lim, phi_lim)

print("Starting NF2FF transform...")
r = 10000
e_theta, e_phi = nf2ff.calc_nf2ff(frequency, x_grid, y_grid, ex_grid, ey_grid, r, theta_grid, phi_grid, verbose=True)
U = nf2ff.calc_radiation_intensity(e_theta, e_phi)
#P_rad = nf2ff.calc_radiated_power(theta_grid, phi_grid, U)[0]  # TODO: Get a faster method to integrate
P_rad = 0.07965391801476739
D = 2*np.pi*U/P_rad/separation  # TODO: Not sure why this separation term is needed!
mag_e = 10*np.log10(np.abs(D))
print("[DONE]\n")

print("Importing farfield data from " + filename_nearfield)
f, theta_ff, phi_ff, e_theta_ff, e_phi_ff, coord_structure_ff = rff.read_fekofarfield_datafile(filename_farfield)

# Assuming an equally spaced 2D grid calculate grid properties (only 1 z-layer is currently permitted)
e_theta_points = coord_structure_ff[1]
e_phi_points = coord_structure_ff[2]
delta_theta = np.abs(theta_ff[1]-theta_ff[0])
delta_phi = np.abs(phi_ff[e_theta_points]-phi_ff[0])

print("File imported with following structure:")
print("Frequencies:   "+str(coord_structure[0]))
print("Theta-Points:  "+str(e_theta_points))
print("Phi-Points:    "+str(e_phi_points))
print("Theta-Step:    "+str(delta_theta)+" deg")
print("Phi-Step:      "+str(delta_phi)+" deg\n")

print("Extracting block: "+str(frequency_block)+" from imported dataset\n")
theta_ff, phi_ff, e_theta_ff, e_phi_ff = rff.read_frequency_block_from_farfield_dataset(frequency_block,
                                                                                        coord_structure_ff,
                                                                                        np.deg2rad(theta_ff),
                                                                                        np.deg2rad(phi_ff),
                                                                                        e_theta_ff, e_phi_ff)

print("Transforming data into meshgrid")
theta_grid_ff = rff.transform_data_coord_to_grid(e_theta_points, e_phi_points, theta_ff)
phi_grid_ff = rff.transform_data_coord_to_grid(e_theta_points, e_phi_points, phi_ff)
e_theta_grid_ff = rff.transform_data_coord_to_grid(e_theta_points, e_phi_points, e_theta_ff)
e_phi_grid_ff = rff.transform_data_coord_to_grid(e_theta_points, e_phi_points, e_phi_ff)
e_ff = nf2ff.calculate_total_gain(e_theta_grid_ff, e_phi_grid_ff)

print("Plotting data..."),
if(False):
    #plotting.plot_nearfield_2d_all(x_grid, y_grid, ex_grid, ey_grid, ez_grid, "Nearfield")
    #plotting.plot_farfield_kspace_2d_all(kx_grid, ky_grid, fex_grid, fey_grid, fez_grid, "Farfield")

    plotting.plot_farfield_3d_spherical(theta_grid, phi_grid, mag_e, "Transformed Farfield")
    plotting.plot_farfield_3d_cartesian(theta_grid, phi_grid, mag_e, "TransformedFarfield")

    plotting.plot_farfield_3d_cartesian(theta_grid_ff, phi_grid_ff, e_ff, "FEKO Farfield")
    plotting.plot_farfield_3d_spherical(theta_grid_ff, phi_grid_ff, e_ff, "FEKO Farfield")

if(True):
    e_cut_nf = rff.get_phi_cut_from_grid_data(50, mag_e)#-np.max(mag_e))
    e_cut_ff = rff.get_phi_cut_from_grid_data(50, e_ff)#-np.max(e_ff))
    plt.figure()
    plt.plot(np.rad2deg(theta_grid[50]), e_cut_nf)
    plt.plot(np.rad2deg(theta_grid_ff[0]), e_cut_ff)
    plt.xlim(-90, 90)
    plt.ylim(-40, 20)
    plt.grid(True)
    plt.axvline(-60)
    plt.axvline(60)
    plt.axvline(-80)
    plt.axvline(80)

print("[DONE]")

plt.show()

# TODO: Check scaling with some principle cuts
# TODO: Program comparison vectors
# TODO: Import real farfield
# TODO: Decide on comparison methodology
# TODO: Decide on errors to focus on
# TODO: Write error injection code
# TODO: Write parameter sweep code
# TODO: Run comparisons for all error types
# TODO: Refactor NFFFandFPlotting
# TODO: Add some 2D phase unwrapping?
# TODO: Sort out scaling factor