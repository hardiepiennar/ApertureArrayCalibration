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

"""
x = np.linspace(0,2*np.pi,101)
y = np.cos(x)
delta = np.abs(x[1]-x[0])
x2 = x[::10] + delta/2
y2_func = interpolate.InterpolatedUnivariateSpline(x,y)
y2 = y2_func(x2)
plt.figure()
plt.plot(x,y)
plt.scatter(x2,y2)
plt.show()
exit()
"""

def import_ideal_farfield_data():

    return theta_grid_ff, phi_grid_ff, gain_ff


def import_ideal_nearfield_data():

    return theta_grid, phi_grid, 10*np.log10(np.abs(U)), frequency, x, y, ex, ey, x_points, y_points


"""General settings"""
#filename_nearfield = "WaveguideHorn80deg.efe"
#filename_farfield = "WaveguideHorn.ffe"
filename_nearfield = "Dipole80deg_lambda_10.efe"
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

# Flight path settings
sample_interval = 0.01

"""Start of script"""
print("\nStarting pyNF2FF\n")
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
x_grid = rff.transform_data_coord_to_grid(x_points, y_points, x)
y_grid = rff.transform_data_coord_to_grid(x_points, y_points, y)
ex_grid = rff.transform_data_coord_to_grid(x_points, y_points, ex)
ey_grid = rff.transform_data_coord_to_grid(x_points, y_points, ey)
theta_grid, phi_grid = nf2ff.generate_spherical_theta_phi_grid(theta_steps, phi_steps, theta_lim, phi_lim)

e_theta, e_phi = nf2ff.calc_nf2ff(frequency, x_grid, y_grid, ex_grid, ey_grid, 10000, theta_grid, phi_grid)

gain_nf = 10*np.log10(np.abs(nf2ff.calc_radiation_intensity(e_theta, e_phi)))

norm_gain_ff = gain_ff - np.max(gain_ff)
norm_factor = np.max(gain_nf)
norm_gain_nf = gain_nf - norm_factor

print("Inject error into nearfield data")
# Interpolate nearfield grid
x_grid = rff.transform_data_coord_to_grid(x_points, y_points, x)
y_grid = rff.transform_data_coord_to_grid(x_points, y_points, y)
ex_grid = rff.transform_data_coord_to_grid(x_points, y_points, ex)
ey_grid = rff.transform_data_coord_to_grid(x_points, y_points, ey)
"""
x = np.array([-2, -1, 0, 1, 2])
y = np.array([-2, -1, 0, 1, 2])
x_grid, y_grid = np.meshgrid(x, y)
ex_grid = np.array([[1,0,1,0,1],
                    [0,1,0,1,0],
                    [1,0,1,0,1],
                    [0,1,0,1,0],
                    [1,0,1,0,1]])
ey_grid = np.array([[1,0,1,0,1],
                    [0,1,0,1,0],
                    [1,0,1,0,1],
                    [0,1,0,1,0],
                    [1,0,1,0,1]])
                    """

# Generate noisy flight path (Vehicle thinks it is on the ideal path, however it is actually sampling on the noisy
# coordinates)
x_lim = (np.min(x_grid), np.max(x_grid))
y_lim = (np.min(y_grid), np.max(y_grid))
wavelength = nf2ff.calc_freespace_wavelength(frequency)
x_coords, y_coords = nf2ff.generate_planar_scanpath(x_lim, y_lim, wavelength/2, wavelength/4)
x_n_coords, y_n_coords = nf2ff.add_position_noise(x_coords, y_coords, 0,0)

# Sample noisy flight path points
ex_n_coords = nf2ff.probe_nearfield(x_grid, y_grid, ex_grid, x_n_coords, y_n_coords)
ey_n_coords = nf2ff.probe_nearfield(x_grid, y_grid, ey_grid, x_n_coords, y_n_coords)

# Use the noisy data to interpolate lamda/2 grid for nf2ff transformation
ex_n_grid = nf2ff.grid_flight_data(x_grid, y_grid, x_coords, y_coords, ex_n_coords)
ey_n_grid = nf2ff.grid_flight_data(x_grid, y_grid, x_coords, y_coords, ey_n_coords)

"""
plotting.plot_nearfield_2d(x_grid, y_grid, np.abs(ex_grid), "Flight path")
plt.scatter(x_n_coords, y_n_coords, c=np.abs(ex_n_coords), s=50)#, edgecolor='')
plotting.plot_nearfield_2d(x_grid, y_grid, np.abs(ex_n_grid), "Flight path")
plt.show()
exit()
"""

"""
# Calculate corrupted nearfield
x_n_amp = planar_loc_error_lim[0]
y_n_amp = planar_loc_error_lim[0]
ex_n_grid = nf2ff.add_position_noise(x_grid, y_grid, ex_grid, x_n_amp, y_n_amp)
ey_n_grid = nf2ff.add_position_noise(x_grid, y_grid, ey_grid, x_n_amp, y_n_amp)
ex_n = np.reshape(ex_n_grid, (1, len(ex_n_grid)*len(ex_n_grid[0])))[0]
ey_n = np.reshape(ey_n_grid, (1, len(ey_n_grid)*len(ey_n_grid[0])))[0]
"""
# Calculate corrupted farfield
e_theta_n_grid, e_phi_n_grid = nf2ff.calc_nf2ff(frequency, x_grid, y_grid, ex_n_grid, ey_n_grid, 10000, theta_grid, phi_grid)

gain_nf_n = 10*np.log10(np.abs(nf2ff.calc_radiation_intensity(e_theta_n_grid, e_phi_n_grid)))
norm_gain_nf_n = gain_nf_n - norm_factor

print("Plotting data..."),

if(True):
    h_cut_nf = rff.get_phi_cut_from_grid_data(np.floor(phi_steps/2), norm_gain_nf)
    h_cut_ff = rff.get_phi_cut_from_grid_data(np.floor(phi_steps/2), norm_gain_ff)
    h_cut_nf_n = rff.get_phi_cut_from_grid_data(np.floor(phi_steps/2), norm_gain_nf_n)
    e_cut_nf = rff.get_phi_cut_from_grid_data(0, norm_gain_nf)
    e_cut_ff = rff.get_phi_cut_from_grid_data(0, norm_gain_ff)
    e_cut_nf_n = rff.get_phi_cut_from_grid_data(0, norm_gain_nf_n)
    print(e_cut_nf_n)
    plt.figure()
    plt.title("E-Plane comparison (phi="+str(phi_grid_ff[0][0])+")")
    plt.plot(np.rad2deg(theta_grid[0]), e_cut_nf)
    plt.plot(np.rad2deg(theta_grid_ff[0]), e_cut_ff)
    plt.plot(np.rad2deg(theta_grid_ff[0]), e_cut_nf_n)
    plt.xlim(-90, 90)
    plt.ylim(-60, 0)
    plt.grid(True)
    plt.axvline(-60)
    plt.axvline(60)
    plt.axvline(-80)
    plt.axvline(80)
    plt.figure()
    plt.title("H-Plane comparison (phi="+str(phi_grid_ff[np.floor(phi_steps/2)][0])+")")
    plt.plot(np.rad2deg(theta_grid[0]), h_cut_nf)
    plt.plot(np.rad2deg(theta_grid_ff[0]), h_cut_ff)
    plt.plot(np.rad2deg(theta_grid_ff[0]), h_cut_nf_n)
    plt.xlim(-90, 90)
    plt.ylim(-60, 0)
    plt.grid(True)
    plt.axvline(-60)
    plt.axvline(60)
    plt.axvline(-80)
    plt.axvline(80)

if(False):
    plotting.plot_farfield_3d_cartesian(theta_ff, phi_ff, norm_gain_ff, "FEKO Farfield", zlim=[-50, 0])
    plotting.plot_farfield_3d_spherical(theta_ff, phi_ff, norm_gain_ff, "FEKO Farfield")
    plotting.plot_farfield_3d_cartesian(theta_nf, phi_nf, norm_gain_nf, "Transformed Farfield", zlim=[-50, 0])
    plotting.plot_farfield_3d_spherical(theta_nf, phi_nf, norm_gain_nf, "Transformed Farfield")
print("[DONE]")

plt.show()


# TODO: Program comparison vectors
# TODO: Decide on comparison methodology
# TODO: Decide on errors to focus on
# TODO: Write parameter sweep code
# TODO: Run comparisons for all error types
# TODO: Refactor NFFFandFPlotting
# TODO: Add some 2D phase unwrapping?

"""
Fix interpolation (something is very screwy)
Interpolate dataset and extract clean lamda/2 points for ideal nearfield (calc_clean_farfield)
Generate typical multicopter paths with noise (gen_multicopter_flight_path)
Interpolate multicopter measurements
Grid multicopter measurements into lambda/2 grid for transformation (grid_multicopter_scan)
"""

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

