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
import os


def calc_nf2ff_error(filename_nearfield, filename_farfield, separation, frequency_block, xy_error, z_error, plots=False, verbose=False):
    if verbose:
        print("Importing farfield data from " + filename_nearfield)
    f, theta_ff, phi_ff, e_theta_ff, e_phi_ff, coord_structure_ff = rff.read_fekofarfield_datafile(filename_farfield)

    # Assuming an equally spaced 2D grid calculate grid properties (only 1 z-layer is currently permitted)
    e_theta_points = coord_structure_ff[1]
    e_phi_points = coord_structure_ff[2]
    delta_theta = np.abs(theta_ff[1]-theta_ff[0])
    delta_phi = np.abs(phi_ff[e_theta_points]-phi_ff[0])

    if verbose:
        print("Extracting block: "+str(frequency_block)+" from imported dataset\n")
    theta_ff, phi_ff, e_theta_ff, e_phi_ff = rff.read_frequency_block_from_farfield_dataset(frequency_block,
                                                                                            coord_structure_ff,
                                                                                            np.deg2rad(theta_ff),
                                                                                            np.deg2rad(phi_ff),
                                                                                            e_theta_ff, e_phi_ff)

    if verbose:
        print("Transforming data into meshgrid")
    theta_grid_ff = rff.transform_data_coord_to_grid(e_theta_points, e_phi_points, theta_ff)
    phi_grid_ff = rff.transform_data_coord_to_grid(e_theta_points, e_phi_points, phi_ff)
    gain_theta_grid_ff = rff.transform_data_coord_to_grid(e_theta_points, e_phi_points, e_theta_ff)
    gain_phi_grid_ff = rff.transform_data_coord_to_grid(e_theta_points, e_phi_points, e_phi_ff)

    if verbose:
        print("Calculating total farfield gain")
    gain_ff = nf2ff.calculate_total_gain(gain_theta_grid_ff, gain_phi_grid_ff)

    if verbose:
        print("Importing nearfield data from " + filename_nearfield)
    f, x, y, z, ex, ey, ez, coord_structure = rff.read_fekonearfield_datafile(filename_nearfield)

    if verbose:
        print("Separation:  "+str(round(separation, 3))+" m\n")

    # Assuming an equally spaced 2D grid calculate grid properties (only 1 z-layer is currently permitted)
    delta = np.abs(x[1]-x[0])
    x_points = coord_structure[1]
    y_points = coord_structure[2]

    if verbose:
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

    if verbose:
        print("Inject error into nearfield data")
    # Interpolate nearfield grid
    x_grid = rff.transform_data_coord_to_grid(x_points, y_points, x)
    y_grid = rff.transform_data_coord_to_grid(x_points, y_points, y)
    ex_grid = rff.transform_data_coord_to_grid(x_points, y_points, ex)
    ey_grid = rff.transform_data_coord_to_grid(x_points, y_points, ey)

    # Generate noisy flight path (Vehicle thinks it is on the ideal path, however it is actually sampling on the noisy
    # coordinates)
    x_lim = (np.min(x_grid), np.max(x_grid))
    y_lim = (np.min(y_grid), np.max(y_grid))
    wavelength = nf2ff.calc_freespace_wavelength(frequency)
    x_coords, y_coords = nf2ff.generate_planar_scanpath(x_lim, y_lim, wavelength/4, wavelength/4)
    error = (3e8/frequency)/xy_error
    x_n_coords, y_n_coords = nf2ff.add_position_noise(x_coords, y_coords, error, error)

    # Sample noisy flight path points, this needs to be done with unwrapped phase
    ex_n_coords_abs = nf2ff.probe_nearfield(x_grid, y_grid, np.abs(ex_grid), x_n_coords, y_n_coords)
    ey_n_coords_abs = nf2ff.probe_nearfield(x_grid, y_grid, np.abs(ey_grid), x_n_coords, y_n_coords)
    # Use a simple 2D unwrapping method, not suitable for complex phase planes
    phase = np.angle(ex_grid)
    phase = np.unwrap(phase, axis=0)
    phase = np.unwrap(phase, axis=1)
    ex_n_coords_phase = nf2ff.probe_nearfield(x_grid, y_grid, phase, x_n_coords, y_n_coords)
    phase = np.angle(ey_grid)
    phase = np.unwrap(phase, axis=0)
    phase = np.unwrap(phase, axis=1)
    ey_n_coords_phase = nf2ff.probe_nearfield(x_grid, y_grid, phase, x_n_coords, y_n_coords)

    """
    Could be implemented in the future
    # Add phase error for z-position error
    #z_n = (np.random.rand(len(ex_n_coords_phase))*2 - 1)*z_error
    #import planes above and below
    #interpolate at z_n for z_error

    #wavenumber = nf2ff.calc_freespace_wavenumber(frequency)
    #theta = np.arctan(np.sqrt(x_n_coords**2 + y_n_coords**2)/separation)
    #kz = wavenumber*np.cos(theta)
    #z_n_phase_error = kz*z_n
    #ex_n_coords_phase += z_n_phase_error
    #ey_n_coords_phase += z_n_phase_error
    """

    # Use the noisy data to interpolate lamda/2 grid for nf2ff transformation
    ex_n_grid_abs = nf2ff.grid_flight_data(x_grid, y_grid, x_coords, y_coords, ex_n_coords_abs)
    ex_n_grid_phase = nf2ff.grid_flight_data(x_grid, y_grid, x_coords, y_coords, ex_n_coords_phase)
    ey_n_grid_abs = nf2ff.grid_flight_data(x_grid, y_grid, x_coords, y_coords, ey_n_coords_abs)
    ey_n_grid_phase = nf2ff.grid_flight_data(x_grid, y_grid, x_coords, y_coords, ey_n_coords_phase)
    ex_n_grid = ex_n_grid_abs*np.exp(1j*ex_n_grid_phase)
    ey_n_grid = ey_n_grid_abs*np.exp(1j*ey_n_grid_phase)

    if(plots):
        plotting.plot_nearfield_2d(x_grid, y_grid, np.abs(ex_grid), "orig abs ex")
        plotting.plot_nearfield_2d(x_grid, y_grid, np.angle(ex_grid), "orig phase ex")
        plotting.plot_nearfield_2d(x_grid, y_grid, np.abs(ey_grid), "orig abs ex")
        plotting.plot_nearfield_2d(x_grid, y_grid, np.angle(ey_grid), "orig phase ex")
        plotting.plot_nearfield_2d(x_grid, y_grid, np.abs(ex_n_grid), "noised abs ex")
        plotting.plot_nearfield_2d(x_grid, y_grid, np.angle(ex_n_grid), "noised phase ex")
        plotting.plot_nearfield_2d(x_grid, y_grid, np.abs(ex_n_grid_phase), "noised phase ex")
        plotting.plot_nearfield_2d(x_grid, y_grid, np.abs(ey_n_grid), "noised abs ey")
        plotting.plot_nearfield_2d(x_grid, y_grid, np.angle(ey_n_grid), "noised phase ey")
        plotting.plot_nearfield_2d(x_grid, y_grid, np.abs(ey_n_grid_phase), "noised phase ey")
        #plt.scatter(x_n_coords, y_n_coords, c=np.abs(ex_n_coords_abs), s=10)#, edgecolor='')
        #plt.show()
    #exit()

    # Calculate corrupted farfield
    e_theta_n_grid, e_phi_n_grid = nf2ff.calc_nf2ff(frequency, x_grid, y_grid, ex_n_grid, ey_n_grid, 10000, theta_grid, phi_grid)

    gain_nf_n = 10*np.log10(np.abs(nf2ff.calc_radiation_intensity(e_theta_n_grid, e_phi_n_grid)))
    norm_gain_nf_n = gain_nf_n - norm_factor

    # Calculate error map
    error = np.abs(10**(norm_gain_ff/10) - 10**(norm_gain_nf_n/10))/(10**(norm_gain_ff/10))

    if(plots):
        plotting.plot_farfield_3d_cartesian(theta_grid,phi_grid,norm_gain_nf_n, "Noised",zlim=[-40,0])
        plotting.plot_farfield_3d_cartesian(theta_grid,phi_grid,norm_gain_ff, "Original",zlim=[-40,0])

    if(plots):
        h_cut_nf = rff.get_phi_cut_from_grid_data(np.floor(phi_steps/2), norm_gain_nf)
        h_cut_ff = rff.get_phi_cut_from_grid_data(np.floor(phi_steps/2), norm_gain_ff)
        h_cut_nf_n = rff.get_phi_cut_from_grid_data(np.floor(phi_steps/2), norm_gain_nf_n)
        e_cut_nf = rff.get_phi_cut_from_grid_data(0, norm_gain_nf)
        e_cut_ff = rff.get_phi_cut_from_grid_data(0, norm_gain_ff)
        e_cut_nf_n = rff.get_phi_cut_from_grid_data(0, norm_gain_nf_n)
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
        plotting.plot_farfield_3d_cartesian(theta_grid, phi_grid, norm_gain_nf, "Transformed Farfield", zlim=[-50, 0])
        plotting.plot_farfield_3d_spherical(theta_grid, phi_grid, norm_gain_nf, "Transformed Farfield")

    return theta_grid, phi_grid, error

"""General settings"""
frequency_file = "Dipole_85deg_400MHz"
separation = 0.899377374000528
frequency_block = 0  # Select the first frequency block in the file
pad_factor = 4

# Spherical farfield pattern settings
scan_angle = np.deg2rad(0)
theta_lim = (-np.pi/2+scan_angle, np.pi/2-scan_angle)
phi_lim = (0, np.pi)
theta_steps = 101
phi_steps = 101

# Sweep settings
planar_loc_error_lim = (200, 4)  # Fraction of wavelength error
planar_loc_error_steps = 51
error_averages = 15  # 15 works well

# Flight path settings
#sample_interval = 1*(3e8/1e9)/4
#row_spacing = 1*(3e8/1e9)/4
#ptp_antenna_spacing = 0.5


"""Start of script"""
print("\nStarting pyNF2FF\n")
print("Simulating frequency file: "+str(frequency_file))
for error in np.linspace(planar_loc_error_lim[0], planar_loc_error_lim[1], planar_loc_error_steps):
    # Calculate the average and max error map a certain frequency and plane error amplitude
    print(str(error)+"m error, "),
    N = error_averages
    print("averaging "+str(N)+" times: "),
    for i in np.arange(N):
        print(str(i+1)+", "),
        theta_grid, phi_grid, error_map = calc_nf2ff_error("FEKOFields/"+frequency_file+".efe",
                                                           "FEKOFields/"+frequency_file+".ffe",
                                                           separation, # Not being used
                                                           0,
                                                           error, "Not used", plots=False, verbose=False)
        if i == 0:
            error_map_accum = error_map
            error_map_max = error_map
        else:
            error_map_accum += error_map
            error_map_max = np.maximum(error_map, error_map_max)

    error_map_avg = error_map_accum/N
    rff.write_farfield_gain_datafile("ErrorFields/error_map_avg_"+str(frequency_file)+"_"+str(error)+".dat",
                                     theta_grid, phi_grid, error_map_avg)
    rff.write_farfield_gain_datafile("ErrorFields/error_map_max_"+str(frequency_file)+"_"+str(error)+".dat",
                                     theta_grid, phi_grid, error_map_max)
    print(str("[DONE]"))

#Integrate the maps
import ErrorMapProcessingScript as eps
eps.run()





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

"""
# Testing the PTP transform
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

print("Importing nearfield data from " + filename_nearfield_ptp)
f, x, y, z, ex, ey, ez, coord_structure = rff.read_fekonearfield_datafile(filename_nearfield_ptp)
print("Separation:  "+str(round(separation, 3))+" m\n")

# Assuming an equally spaced 2D grid calculate grid properties (only 1 z-layer is currently permitted)
delta = np.abs(x[1]-x[0])
x_points = coord_structure[1]
y_points = coord_structure[2]

print("Extracting block: "+str(frequency_block)+" from imported dataset\n")
x, y, z, ex, ey, ez = rff.read_frequency_block_from_nearfield_dataset(frequency_block, coord_structure, x, y, z,
                                                                      ex, ey, ez)
ex_grid_2_p = rff.transform_data_coord_to_grid(x_points, y_points, ex)
ey_grid_2_p = rff.transform_data_coord_to_grid(x_points, y_points, ey)

plotting.plot_nearfield_2d(x_grid, y_grid, np.angle(ex_grid), title="")
# Use a simple unwrapping technique to unwrap the phase of the data
phase = np.angle(ex_grid)
phase = np.unwrap(phase, axis=0)
phase = np.unwrap(phase, axis=1)
ex_grid = np.abs(ex_grid)*np.exp(1j*phase)
plotting.plot_nearfield_2d(x_grid, y_grid, np.angle(ex_grid), title="")
plt.show()
exit()

# Below is a PTP transform thats broken. It might help us again one day.
# This is what we have... lets get that phase
ex_grid_1 = np.abs(ex_grid)
ey_grid_1 = np.abs(ey_grid)
ex_grid_2 = np.abs(ex_grid_2_p)
ey_grid_2 = np.abs(ey_grid_2_p)

wavenumber = nf2ff.calc_freespace_wavenumber(frequency)
theta = np.arctan(np.sqrt(x_grid**2 + y_grid**2)/(separation))
kz = wavenumber*np.cos(theta)

plt.figure()
plt.title("Error")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.grid(True)
N = 10000
random_phase = np.reshape(2*np.pi*np.random.rand(len(ex_grid)*len(ex_grid[0])) - np.pi, np.shape(ex_grid))
plane = ex_grid_1*np.exp(1j*np.angle(ex_grid)*0.1)
alpha = 0.0
for i in np.arange(N):
    plane = np.fft.ifft2(np.fft.fft2(plane)*np.exp(-1j*kz*(ptp_antenna_spacing)))
    phase_pert = (alpha*(N-i)/N)*np.reshape(2*np.pi*np.random.rand(len(ex_grid)*len(ex_grid[0])) - np.pi, np.shape(ex_grid))
    plane = ex_grid_2*np.exp(1j*(np.angle(plane)+phase_pert))
    plane = np.fft.ifft2(np.fft.fft2(plane)*np.exp(1j*kz*(ptp_antenna_spacing)))

    error = np.sum((np.angle(plane) - np.angle(ex_grid))**2)#/np.sum(np.angle(ex_grid)**2)
    #error = np.sum((np.abs(delta))**2)/np.sum(ex_grid_1**2)
    if i%10 == 0:
        print(str(i)+"  "+str(error))
        if i%2000 == 0:
            plotting.plot_nearfield_2d(x_grid, y_grid, np.angle(plane), title="PTP")
            #plt.scatter(i, error)
            plt.draw()
            plt.pause(0.01)
    if i < N-1:
        phase_pert = (alpha*(N-i)/N)*np.reshape(2*np.pi*np.random.rand(len(ex_grid)*len(ex_grid[0])) - np.pi, np.shape(ex_grid))
        plane = ex_grid_1*np.exp(1j*(np.angle(plane)+phase_pert))


plotting.plot_nearfield_2d(x_grid, y_grid, np.angle(plane), title="PTP")
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.show()
exit()
"""
