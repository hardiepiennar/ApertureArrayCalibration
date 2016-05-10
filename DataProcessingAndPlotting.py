"""
Process and plot the final data
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

import NF2FF.NF2FF as nf2ff
import NF2FF.NFFFandFPlotting as plotting
import FileReading.readFromFile as rff
import pyNF2FF as ecs
from scipy import stats

fig_size = 4.5

def load_error_data(filename):
    # import lambda error data
    file_handle = open(filename)
    reader = csv.reader(file_handle, delimiter=' ')

    error_data = []
    avg_data = []
    max_data = []

    # Read header and body of file
    line_no = 0
    for row in reader:
        string = '    '.join(row)
        elements = string.split()
        error_data.append(float(elements[0]))
        avg_data.append(float(elements[1]))
        max_data.append(float(elements[2]))
        line_no += 1

    # Close file after reading
    file_handle.close()

    error_data = np.array(error_data)
    avg_data = np.array(avg_data)
    max_data = np.array(max_data)

    return error_data, avg_data, max_data


def plot_planar_pos_error_sensitivity():
    error, avg, max = load_error_data("errorFinal.dat")
    error_frac = float(1)/error

    fig = plt.figure()
    fig.set_size_inches(fig_size, (float(6)/8)*fig_size)
    plt.grid(True)
    #plt.title("Farfield error due to planar uncertainty")
    plt.xlabel("Planar position error [$\lambda$]")
    plt.ylabel("Farfield error")
    plt.xlim(np.min(error_frac), np.max(error_frac))
    plt.plot(error_frac, avg)
    plt.plot(error_frac, max)
    plt.legend(["avg", "max"], loc='upper left')

    # Calculate error sensitivity equations
    slope, intercept, r_value, p_value, std_error = stats.linregress(error_frac, avg)
    print("Avg")
    print(slope)
    print(intercept)
    slope, intercept, r_value, p_value, std_error = stats.linregress(error_frac, max)
    print("Max")
    print(slope)
    print(intercept)
    x = np.linspace(0, 0.25, 50)
    #plt.plot(x, x*2.021+0.015)
    #plt.plot(x, x*6.207+0.025)


def plot_feko_farfield_3d():
    # Load data
    frequency, theta, phi, theta_gain, phi_gain, no_samples = rff.read_fekofarfield_datafile("FEKOFields/Dipole_51points.ffe")
    total_gain = nf2ff.calculate_total_gain(theta_gain, phi_gain)

    # Grid data
    theta_points = no_samples[1]
    phi_points = no_samples[2]

    new_shape = (theta_points, phi_points)
    theta_grid = np.reshape(np.deg2rad(theta), new_shape)
    phi_grid = np.reshape(np.deg2rad(phi), new_shape)
    gain_grid = np.reshape(total_gain, new_shape)

    ax, fig = plotting.plot_farfield_3d_cartesian(theta_grid, phi_grid, gain_grid, "", zlim=[-30,10])
    fig.set_size_inches(fig_size, fig_size*(float(6)/8))
    ax.set_xticks([-90, 90])
    ax.set_yticks([0, 180])
    ax.set_zticks([-30, -20, -10, 0, 10])


def plot_feko_nearfield():
    # Load data
    frequency, x, y, z, ex, ey, ez, no_samples = rff.read_fekonearfield_datafile("FEKOFields/Dipole_85deg_400MHz.efe")

    wavelength = nf2ff.calc_freespace_wavelength(frequency[0])
    x /= wavelength
    y /= wavelength
    z /= wavelength

    # Grid data
    x_points = no_samples[1]
    y_points = no_samples[2]

    new_shape = (x_points, y_points)
    ey_grid = np.reshape(ey, new_shape)
    x_grid = np.reshape(x, new_shape)
    y_grid = np.reshape(y, new_shape)

    plt.figure()
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)


    ax[0].set_title("Magnitude [dB]")
    ax[0].set_xlabel("x [$\lambda$]")
    ax[0].set_ylabel("y [$\lambda$]")
    ax[0].set_ylim(np.min(y_grid), np.max(y_grid))
    ax[0].set_xlim(np.min(x_grid), np.max(x_grid))
    extents = (np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid))
    data = 10*np.log10(np.abs(ey_grid))
    levels = np.arange(-40, 0, 8)

    cax = ax[0].imshow(data, extent=extents)
    ax[0].contour(data, extent=extents, colors='k', origin='upper', levels=levels)
    cb = fig.colorbar(cax, orientation='horizontal', ax=ax[0])
    cb.set_ticks(levels)

    ax[1].set_title("Unwrapped phase [rad]")
    ax[1].set_xlabel("x [$\lambda$]")
    ax[1].set_ylim(np.min(y_grid), np.max(y_grid))
    ax[1].set_xlim(np.min(x_grid), np.max(x_grid))
    extents = (np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid))
    data = np.angle(ey_grid)
    data = np.unwrap(data, axis=0)
    data = np.unwrap(data, axis=1)
    levels = np.arange(0, 300, 60)

    cax = ax[1].imshow(data, extent=extents)
    ax[1].contour(data, extent=extents, colors='k', origin='upper')
    cb = fig.colorbar(cax, orientation='horizontal', ax=ax[1])
    cb.set_ticks(levels)

    #ax[0].set_aspect("equal")
    #ax[1].set_aspect("equal")
    #fig.set_tight_layout(True)

    fig.set_size_inches(fig_size, fig_size*(float(6)/8))


def plot_principle_plane_errors():
    frequency_file = "Dipole_85deg_400MHz"
    separation = 0.899377374000528

    # Spherical farfield pattern settings
    scan_angle = np.deg2rad(0)
    theta_lim = (-np.pi/2+scan_angle, np.pi/2-scan_angle)
    phi_lim = (0, np.pi)
    theta_steps = 101
    phi_steps = 101

    # Sweep settings
    planar_loc_error_lim = (200, 4)  # Fraction of wavelength error
    planar_loc_error_steps = 4
    fig = plt.figure()
    plot_ff = True
    errors = [200, 50, 10, 5]
    linestyle = ['-', '--','-.','.']
    i = 0
    for error in errors:
        print(error)
        # Calculate the average and max error map a certain frequency and plane error amplitude
        theta_grid, phi_grid, error_map = ecs.calc_nf2ff_error("FEKOFields/"+frequency_file+".efe",
                                                               "FEKOFields/"+frequency_file+".ffe",
                                                               theta_lim, theta_steps, phi_lim, phi_steps,
                                                               separation, # Not being used
                                                               0,
                                                               error, "Not used",
                                                               e_ideal_plots=False,
                                                               eplots=True,
                                                               verbose=False,
                                                               linestyle=linestyle[i])
        i += 1
        plot_ff = False
    #plt.title("E-Plane Farfield Errors")
    plt.ylim(-15, 5)
    plt.legend(["${\lambda}/200$", "${\lambda}/50$", "${\lambda}/10$", "${\lambda}/5$"], loc='lower right')
    plt.xlabel("$\Theta$ [deg]")
    plt.ylabel("Normalized farfield [dB]")

    fig.set_size_inches(fig_size, fig_size*(float(6)/8))


def plot_max_error_dgps():
    f = np.linspace(0, 2e9, 201)
    wavelength = nf2ff.calc_freespace_wavelength(f)

    error_upper = 0.05
    error_lower = 0.01

    frac_wavelength_upper = wavelength/error_upper
    frac_wavelength_lower = wavelength/error_lower

    error_upper = (1/frac_wavelength_upper)*6.207 + 0.025
    error_lower = (1/frac_wavelength_lower)*6.207 + 0.025

    fig = plt.figure()
    plt.grid(True)
    plt.semilogx(f*1e-6, error_upper, color='black')
    plt.semilogx(f*1e-6, error_lower, color='black')
    plt.fill_between(f*1e-6, error_upper, error_lower, color='lightgrey')
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Farfield error")
    plt.xlim(np.min(f*1e-6),np.max(f*1e-6))
    plt.ylim(0, 1)
    #plt.title("DGPS Error Sensitivity")
    fig.set_size_inches(fig_size, (float(6)/8)*fig_size)


#Direct input
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

plot_max_error_dgps()
plt.savefig('Figures/DGPS_error_sensitivity.pdf', dpi=1000, bbox_inches='tight')

plot_principle_plane_errors()
plt.savefig('Figures/Farfield_error.pdf', dpi=1000, bbox_inches='tight')

plot_planar_pos_error_sensitivity()
plt.savefig('Figures/Error_sensitivity.pdf', dpi=1000, bbox_inches='tight')

plot_feko_farfield_3d()
plt.savefig('Figures/Dipole_farfield.pdf', dpi=1000, bbox_inches='tight')

plot_feko_nearfield()
plt.savefig('Figures/Dipole_nearfield.pdf', dpi=1000, bbox_inches='tight')


plt.show()