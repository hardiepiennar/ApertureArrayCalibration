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
import scipy.interpolate as interpolate


def import_ideal_farfield_data():
    print("Importing farfield data from " + filename_nearfield)
    f, theta_ff, phi_ff, e_theta_ff, e_phi_ff, coord_structure_ff = rff.read_fekofarfield_datafile(filename_farfield)

    # Assuming an equally spaced 2D grid calculate grid properties (only 1 z-layer is currently permitted)
    e_theta_points = coord_structure_ff[1]
    e_phi_points = coord_structure_ff[2]
    delta_theta = np.abs(theta_ff[1]-theta_ff[0])
    delta_phi = np.abs(phi_ff[e_theta_points]-phi_ff[0])

    print("Extracting block: "+str(frequency_block)+" from imported dataset\n")
    theta_ff, phi_ff, e_theta_ff, e_phi_ff = rff.read_frequency_block_from_farfield_dataset(frequency_block,
                                                                                            coord_structure_ff,
                                                                                            np.deg2rad(theta_ff),
                                                                                            np.deg2rad(phi_ff),
                                                                                            e_theta_ff, e_phi_ff)

    print("Transforming data into meshgrid")
    theta_grid_ff = rff.transform_data_coord_to_grid(e_theta_points, e_phi_points, theta_ff)
    phi_grid_ff = rff.transform_data_coord_to_grid(e_theta_points, e_phi_points, phi_ff)
    gain_theta_grid_ff = rff.transform_data_coord_to_grid(e_theta_points, e_phi_points, e_theta_ff)
    gain_phi_grid_ff = rff.transform_data_coord_to_grid(e_theta_points, e_phi_points, e_phi_ff)

    print("Calculating total farfield gain")
    gain_ff = nf2ff.calculate_total_gain(gain_theta_grid_ff, gain_phi_grid_ff)

    return theta_grid_ff, phi_grid_ff, gain_ff


def import_ideal_nearfield_data():
    print("Importing nearfield data from " + filename_nearfield)
    f, x, y, z, ex, ey, ez, coord_structure = rff.read_fekonearfield_datafile(filename_nearfield)
    print("Separation:  "+str(round(separation, 3))+" m\n")

    # Assuming an equally spaced 2D grid calculate grid properties (only 1 z-layer is currently permitted)
    delta = np.abs(x[1]-x[0])
    x_points = coord_structure[1]
    y_points = coord_structure[2]

    print("Extracting block: "+str(frequency_block)+" from imported dataset\n")
    x, y, z, ex, ey, ez = rff.read_frequency_block_from_nearfield_dataset(frequency_block, coord_structure, x, y, z,
                                                                          ex, ey, ez)
    frequency = f[frequency_block*x_points*y_points]


    print("Transforming NF2FF transform")
    theta_grid, phi_grid, e_theta, e_phi = nf2ff.calc_nf2ff_from_coord_data(frequency, x_points, y_points, x, y, ex, ey,
                                                      theta_steps, phi_steps, theta_lim, phi_lim,
                                                      verbose=True, pad_factor=pad_factor)
    U = nf2ff.calc_radiation_intensity(e_theta, e_phi)

    return theta_grid, phi_grid, 10*np.log10(np.abs(U)), frequency, x, y, ex, ey, x_points, y_points


"""General settings"""
#filename_nearfield = "WaveguideHorn80deg.efe"
#filename_farfield = "WaveguideHorn.ffe"
filename_nearfield = "Dipole88deg.efe"
filename_farfield = "Dipole.ffe"
separation = 0.0899377374000528
frequency_block = 0  # Select the first frequency block in the file
pad_factor = 4

# Spherical farfield pattern settings
scan_angle = np.deg2rad(0)
theta_lim = (-np.pi/2+scan_angle, np.pi/2-scan_angle)
phi_lim = (0, np.pi)
theta_steps = 101
phi_steps = 101

# Sweep settings
planar_loc_error_lim = (0*(3e8/1e9)/10, 1)
planar_loc_error_steps = 10

"""Start of script"""
print("\nStarting pyNF2FF\n")
theta_ff, phi_ff, gain_ff = import_ideal_farfield_data()
theta_nf, phi_nf, gain_nf, f, x, y, ex, ey, x_points, y_points = import_ideal_nearfield_data()

norm_gain_ff = gain_ff - np.max(gain_ff)
norm_factor = np.max(gain_nf)
norm_gain_nf = gain_nf - norm_factor

print("Inject error into nearfield data")
# Interpolate nearfield grid
x_grid = rff.transform_data_coord_to_grid(x_points, y_points, x)
y_grid = rff.transform_data_coord_to_grid(x_points, y_points, y)
ex_grid = rff.transform_data_coord_to_grid(x_points, y_points, ex)
ey_grid = rff.transform_data_coord_to_grid(x_points, y_points, ey)

# Calculate corrupted nearfield
x_n_amp = planar_loc_error_lim[0]
y_n_amp = planar_loc_error_lim[0]
ex_n_grid = nf2ff.add_position_noise(x_grid, y_grid, ex_grid, x_n_amp, y_n_amp)
ey_n_grid = nf2ff.add_position_noise(x_grid, y_grid, ey_grid, x_n_amp, y_n_amp)
ex_n = np.reshape(ex_n_grid, (1, len(ex_n_grid)*len(ex_n_grid[0])))[0]
ey_n = np.reshape(ey_n_grid, (1, len(ey_n_grid)*len(ey_n_grid[0])))[0]

# Calculate corrupted farfield
theta_n_grid, phi_n_grid, e_theta_nf_n, e_phi_nf_n = nf2ff.calc_nf2ff_from_coord_data(f, x_points, y_points, x, y, ex_n, ey_n,
                                                            theta_steps, phi_steps, theta_lim, phi_lim)
gain_nf_n = 10*np.log10(np.abs(nf2ff.calc_radiation_intensity(e_theta_nf_n, e_phi_nf_n)))
norm_gain_nf_n = gain_nf_n - norm_factor

print("Plotting data..."),
if(True):
    plotting.plot_farfield_3d_cartesian(theta_ff, phi_ff, norm_gain_ff, "FEKO Farfield", zlim=[-50, 0])
    plotting.plot_farfield_3d_spherical(theta_ff, phi_ff, norm_gain_ff, "FEKO Farfield")
    plotting.plot_farfield_3d_cartesian(theta_nf, phi_nf, norm_gain_nf, "Transformed Farfield", zlim=[-50, 0])
    plotting.plot_farfield_3d_spherical(theta_nf, phi_nf, norm_gain_nf, "Transformed Farfield")

if(True):
    h_cut_nf = rff.get_phi_cut_from_grid_data(np.floor(phi_steps/2), norm_gain_nf)
    h_cut_ff = rff.get_phi_cut_from_grid_data(np.floor(phi_steps/2), norm_gain_ff)
    h_cut_nf_n = rff.get_phi_cut_from_grid_data(np.floor(phi_steps/2), norm_gain_nf_n)
    e_cut_nf = rff.get_phi_cut_from_grid_data(0, norm_gain_nf)
    e_cut_ff = rff.get_phi_cut_from_grid_data(0, norm_gain_ff)
    e_cut_nf_n = rff.get_phi_cut_from_grid_data(0, norm_gain_nf_n)
    plt.figure()
    plt.title("E-Plane comparison (phi="+str(phi_ff[0][0])+")")
    plt.plot(np.rad2deg(theta_nf[0]), e_cut_nf)
    plt.plot(np.rad2deg(theta_ff[0]), e_cut_ff)
    #plt.plot(np.rad2deg(theta_ff[0]), e_cut_nf_n)
    plt.xlim(-90, 90)
    plt.ylim(-60, 0)
    plt.grid(True)
    plt.axvline(-60)
    plt.axvline(60)
    plt.axvline(-80)
    plt.axvline(80)
    plt.figure()
    plt.title("H-Plane comparison (phi="+str(phi_ff[np.floor(phi_steps/2)][0])+")")
    plt.plot(np.rad2deg(theta_nf[0]), h_cut_nf)
    plt.plot(np.rad2deg(theta_ff[0]), h_cut_ff)
    #:w
    # plt.plot(np.rad2deg(theta_ff[0]), h_cut_nf_n)
    plt.xlim(-90, 90)
    plt.ylim(-60, 0)
    plt.grid(True)
    plt.axvline(-60)
    plt.axvline(60)
    plt.axvline(-80)
    plt.axvline(80)
print("[DONE]")

plt.show()


# TODO: Program comparison vectors
# TODO: Decide on comparison methodology
# TODO: Decide on errors to focus on
# TODO: Write error injection code
# TODO: Write parameter sweep code
# TODO: Run comparisons for all error types
# TODO: Refactor NFFFandFPlotting
# TODO: Add some 2D phase unwrapping?

"""
Localization error characterisation
-----------------------------------
Sweep over frequency
    Read in ideal nearfield from feko -
    Read in ideal farfield from feko -
    Calculate farfield from ideal nearfield data to find normalisation values -
    Define error injection boundaries (meters from ideal x and y) -
    Sweep over error amplitude:
        Inject error into ideal nearfield coordinates
        Create interpolated nearfield function from corrupted coordinate data
        Calculate data onto equally spaced nearfield grid
        Use corrupted data to calculate farfield
        Calculate farfield error (principle cuts, max error, min error, average error)
        Store error data
"""

"""
#P_rad = nf2ff.calc_radiated_power(theta_grid, phi_grid, U)[0]  # TODO: Get a faster method to integrate
P_rad = 0.07965391801476739
D = 2*np.pi*U/P_rad/separation  # TODO: Not sure why this separation term is needed!
mag_D = 10*np.log10(np.abs(D))
norm_mag_d = mag_D - np.max(mag_D)

print("Calculating equivelent multi")
norm_nf = np.abs(D)/np.max(np.abs(D))
D_act = 10**(gain_ff/10)
norm_ff = np.abs(D_act)/np.max(np.abs(D_act))
empl = nf2ff.calc_empl(norm_nf, norm_ff)
"""

