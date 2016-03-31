from FileReading import readFromFile as rff
import numpy as np
import matplotlib.pyplot as plt
from NF2FF.NF2FF import *

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

print("Extracting first frequency point: "),
block = 0
print(str(f[block*block_size])+"Hz ..."),
theta, phi, gain_theta, gain_phi = rff.read_frequency_block_from_dataset(block, no_samples, theta, phi, gain_theta,
                                                                         gain_phi)
print("[DONE]")

print("Calculating total gain..."),
gain = calculate_total_gain(gain_theta, gain_phi)
print("[DONE]")

print("Extracting phi cut: "),
trace_no = 3
print(str(phi[trace_no*theta_points])+" degrees ..."),
phi_cut = []
for i in np.arange(trace_no*theta_points, (trace_no+1)*theta_points):
    phi_cut.append(gain[i])
phi_cut = np.array(phi_cut)
print("[DONE]")

print("Arrange data into grid..."),
resolution = [theta_points, phi_points]
grid_x, grid_y, theta_gain_grid = transform_data_coord_to_grid([theta, phi], gain_theta, resolution)
grid_x, grid_y, phi_gain_grid = transform_data_coord_to_grid([theta, phi], gain_phi, resolution)
grid_x, grid_y, gain_grid = transform_data_coord_to_grid([theta, phi], gain, resolution)
print("[DONE]")

print("Displaying data...")
if(True):
    plt.figure()
    plt.title("Gain")
    plt.xlabel("theta [degrees]")
    plt.ylabel("Gain [dB]")
    plt.plot(theta[0:theta_points], phi_cut)
    plt.xlim(0, 180)
    plt.ylim(-80, 10)
    plt.grid(True)
    plt.show()
    exit()

plt.figure()
plt.title("Gain")
plt.xlabel("theta [degrees]")
plt.ylabel("phi [degrees]")
plt.xlim(0, 180)
plt.ylim(0, 360)
plt.imshow(gain_grid, extent=(np.min(theta), np.max(theta), np.min(phi), np.max(phi)))
plt.show()
