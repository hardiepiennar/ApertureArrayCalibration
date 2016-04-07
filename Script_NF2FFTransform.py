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
f_points = no_samples[0]
ex_points = no_samples[1]
ey_points = no_samples[2]
ez_points = no_samples[2]
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
print("[DONE]")

print("Transforming nearfield data to farfield data... "),
grid_z_nf = np.ones((len(grid_x_nf), len(grid_x_nf[0])))
trans_farfield_x, trans_farfield_y, trans_farfield_z = calc_nf2ff(grid_x_nf, grid_y_nf, grid_z_nf, ex_grid, ey_grid)
print("[DONE]")

print("Displaying data...")
if False:
    nfff_plot.plot_phi_cut(theta[0:theta_points], phi_cut, "Phi cut")
if False:
    nfff_plot.plot_farfield_2d(theta, phi, gain_grid, "Farfield pattern", [-20, 10], only_top_hemisphere=True)
    #nfff_plot.plot_farfield_2d(trans_theta, trans_phi, 20*np.log10(np.abs(trans_farfield)), "Transformed Farfield pattern",
    #                           only_top_hemisphere=True)
if False:
    #nfff_plot.plot_nearfield_2d(x, y, np.abs(ex_grid), "Nearfield x pattern", [0.01, 1.24])
    #nfff_plot.plot_nearfield_2d(x, y, np.abs(ey_grid), "Nearfield y pattern", [0.01, 1.24])
    #nfff_plot.plot_nearfield_2d(x, y, np.abs(ez_grid), "Nearfield z pattern", [0.01, 1.24])
    #nfff_plot.plot_nearfield_2d(x, y, np.abs(e_grid), "Nearfield pattern", [0.01, 1.24])
    #nfff_plot.plot_nearfield_2d(x, y, (180/np.pi)*np.angle(e_grid), "Nearfield pattern")
    nfff_plot.plot_nearfield_2d(x, y, np.abs(trans_farfield_x), "Farfield pattern x")
    nfff_plot.plot_nearfield_2d(x, y, (180/np.pi)*np.angle(trans_farfield_x), "Farfield pattern x")


plt.show()
exit()

#Write function to calculate farfield from nearfield from ex, ey and ez and return fx, fy.
#Write function to transform fx,fy into fTheta and fPhi
#Add label to color bar
#Add e-field subplots for 3 orientations
#Do nf2ff transform and compare with real farfield

