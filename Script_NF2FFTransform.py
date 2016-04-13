from FileReading import readFromFile as rff
import numpy as np
import matplotlib.pyplot as plt
from NF2FF.NF2FF import *
import NF2FF.NFFFandFPlotting as nfff_plot

"""
Read in the farfield of an array dipole antenna
"""

print("Reading in farfield data..."),
#filename = "DDA_DualPol_FF_noFB.ffe"
filename = "Horizontal_Dipole_Above_Plane.ffe"
f, theta, phi, gain_theta, gain_phi, no_samples = rff.read_fekofarfield_datafile(filename)
f_points = no_samples[0]
theta_points = no_samples[1]
phi_points = no_samples[2]
block_size = theta_points*phi_points
print("(F:"+str(f_points)+" T:"+str(theta_points)+" P:"+str(phi_points)+")..."),
print("[DONE]")

print("Reading in nearfield data..."),
filename = "Horizontal_Dipole_Above_Plane.efe"
f_nf, x, y, z, ex, ey, ez, no_samples_nf = rff.read_fekonearfield_datafile(filename)
f_points = no_samples_nf[0]
ex_points = no_samples_nf[1]
ey_points = no_samples_nf[2]
ez_points = no_samples_nf[2]
block_size_nf = ex_points*ey_points*ez_points
print("[DONE]")

print("Extracting first frequency point: "),
block = 0
print(str(f[block*block_size])+"Hz ..."),
theta, phi, gain_theta, gain_phi = rff.read_frequency_block_from_dataset(block, no_samples, theta, phi, gain_theta,
                                                                         gain_phi)
print("[DONE]")

print("Calculating total gain..."),
gain = calculate_total_gain(gain_theta, gain_phi)
print("[DONE]")

print("Calculate total e-field..."),
e = calculate_total_e_field(ex, ey, ez)
print("[DONE]")

print("Extracting phi cut: "),
trace_no = 0
print(str(phi[trace_no*theta_points])+" degrees ..."),
phi_cut = rff.get_phi_cut(trace_no, no_samples, gain)
print("[DONE]")

print("Arrange data into grid...")
print("Farfield")
resolution = [theta_points, phi_points]
grid_x, grid_y, theta_gain_grid = transform_data_coord_to_grid([theta, phi], gain_theta, resolution)
grid_x, grid_y, phi_gain_grid = transform_data_coord_to_grid([theta, phi], gain_phi, resolution)
grid_x, grid_y, gain_grid = transform_data_coord_to_grid([theta, phi], gain, resolution)
print("Nearfield")
resolution = [ex_points, ey_points]
grid_x_nf, grid_y_nf, ex_grid = transform_data_coord_to_grid([x, y], ex, resolution)
grid_x_nf, grid_y_nf, ey_grid = transform_data_coord_to_grid([x, y], ey, resolution)
grid_x_nf, grid_y_nf, ez_grid = transform_data_coord_to_grid([x, y], ez, resolution)
grid_x_nf, grid_y_nf, e_grid = transform_data_coord_to_grid([x, y], e, resolution)
grid_z_nf = np.ones((len(grid_x_nf), len(grid_x_nf[0])))*z[0]
print("[DONE]")

ex_grid = np.abs(ex_grid)*np.exp(1j*np.angle(ex_grid))

print("Transforming nearfield data to farfield data... "),
padding = 400
ex_grid = np.pad(ex_grid, (0, padding), 'constant')
ey_grid = np.pad(ey_grid, (0, padding), 'constant')
trans_farfield_x, trans_farfield_y = calc_nf2ff(ex_grid, ey_grid)
grid_x_ff, grid_y_ff = calc_kgrid(grid_x_nf, grid_y_nf, padding)
f = 1e9
c = 3e8
lambd = c/f
grid_z_ff = (z[0]**2 - grid_x_ff**2 -grid_y_ff**2)**-2
trans_farfield_z = -(trans_farfield_x*grid_x_ff + trans_farfield_y*grid_y_ff)/(grid_z_ff)
trans_farfield_xyz = calculate_total_e_field(trans_farfield_x, trans_farfield_y, trans_farfield_z)

print("calculating spherical data"),
trans_theta, trans_phi, trans_farfield_theta, trans_farfield_phi = transform_cartesian_to_spherical(grid_x_ff,
                                                                                                    grid_y_ff,
                                                                                                    grid_z_ff,
                                                                                                    trans_farfield_x,
                                                                                                    trans_farfield_y,
                                                                                                    trans_farfield_x)
theta_trans_coords = np.rad2deg(trans_theta.reshape((1, len(trans_theta)*len(trans_theta[0]))))[0]
phi_trans_coords = np.rad2deg(trans_phi.reshape((1, len(trans_phi)*len(trans_phi[0]))))[0]
trans_farfield_gain = np.log10(np.abs(np.sqrt(trans_farfield_theta**2 + trans_farfield_phi**2)))
print("[DONE]")

print("Displaying data...")
if False:
    nfff_plot.plot_phi_cut(theta[0:theta_points], phi_cut, "Phi cut")
if True:
    #nfff_plot.plot_farfield_2d(theta, phi, gain_grid, "Farfield pattern", [-20, 10], only_top_hemisphere=False)
    #nfff_plot.plot_farfield_2d(grid_x, grid_y, np.angle(theta_gain_grid), "Transformed Farfield pattern",
    #                           only_top_hemisphere=False)
    nfff_plot.plot_farfield_2d(theta_trans_coords, phi_trans_coords, np.abs(trans_farfield_gain), "Transformed Farfield pattern",
                               only_top_hemisphere=False)
    pass
if True:
    #nfff_plot.plot_nearfield_2d(x, y, np.abs(ex_grid), "Nearfield x pattern", [0.01, 1.24])
    #nfff_plot.plot_nearfield_2d(x, y, np.abs(ey_grid), "Nearfield y pattern", [0.01, 1.24])
    #nfff_plot.plot_nearfield_2d(x, y, np.abs(ez_grid), "Nearfield z pattern", [0.01, 1.24])
    #nfff_plot.plot_nearfield_2d(x, y, np.abs(e_grid), "Nearfield pattern", [0.01, 1.24])
    #nfff_plot.plot_nearfield_2d(x, y, (180/np.pi)*np.angle(e_grid), "Nearfield pattern")
    #nfff_plot.plot_nearfield_2d(x, y, 20*np.log10(np.abs(np.sqrt(trans_farfield_x**2 + trans_farfield_y**2 + trans_farfield_z**2))), "Farfield pattern x")
    #nfff_plot.plot_nearfield_2d(x, y, (180/np.pi)*np.angle(trans_farfield_x), "Farfield pattern x")
    #nfff_plot.plot_nearfield_2d(x, y, np.abs(trans_farfield_x), "Farfield pattern x")
    #nfff_plot.plot_nearfield_2d(x, y, np.abs(trans_farfield_y), "Farfield pattern y")
    #nfff_plot.plot_nearfield_2d(x, y, np.abs(trans_farfield_z), "Farfield pattern z")
    nfff_plot.plot_nearfield_2d_all(grid_x_ff, grid_y_ff, trans_farfield_x, trans_farfield_y, trans_farfield_z,
                                    "farfield (z = 1m)",
                                    xlim=[np.min(x), np.max(x)],
                                    ylim=[np.min(y), np.max(y)])
    nfff_plot.plot_nearfield_2d_all(x, y, ex_grid, ey_grid, ez_grid,
                                    "nearfield (z = 1m)",
                                    xlim=[np.min(x), np.max(x)],
                                    ylim=[np.min(y), np.max(y)])
    nfff_plot.plot_nearfield_2d(grid_x_ff, grid_y_ff, np.abs(trans_farfield_xyz), "Transformed Farfield pattern")

plt.show()
exit()

#Write function to calculate farfield from nearfield from ex, ey and ez and return fx, fy.
#Write function to transform fx,fy into fTheta and fPhi
#Add label to color bar
#Add e-field subplots for 3 orientations
#Do nf2ff transform and compare with real farfield

