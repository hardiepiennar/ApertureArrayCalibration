from FileReading import readFromFile as rff
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
from NF2FF.NF2FF import transform_data_coord_to_grid

"""
Read in the farfield of an array dipole antenna
"""

print("Reading in farfield data..."),
filename = "DDA_DualPol_FF_noFB.ffe"
f, theta, phi, gain_theta, gain_phi, no_samples = rff.read_fekofarfield_datafile(filename)
f_points = no_samples[0]
theta_points = no_samples[1]
phi_points = no_samples[2]
block_size = theta_points*phi_points
print("(F:"+str(f_points)+" T:"+str(theta_points)+" P:"+str(phi_points)+")..."),
print("[DONE]")

print("Extracting first frequency point..."),
block = 100
theta = theta[block_size*block:block_size*(block+1)-1]
phi = phi[block_size*block:block_size*(block+1)-1]
gain_theta = gain_theta[block_size*block:block_size*(block+1)-1]
gain_phi = gain_phi[block_size*block:block_size*(block+1)-1]
print("[DONE]")

print("Calculating total gain..."),
gain = np.sqrt(np.square(gain_theta)+np.square(gain_phi))
print("[DONE]")

print("Arrange data into grid..."),
resolution = [theta_points, phi_points]
grid_x, grid_y, theta_gain_grid = transform_data_coord_to_grid([theta, phi], gain_theta, resolution)
grid_x, grid_y, phi_gain_grid = transform_data_coord_to_grid([theta, phi], gain_phi, resolution)
grid_x, grid_y, gain_grid = transform_data_coord_to_grid([theta, phi], gain, resolution)
print("[DONE]")

if False:
    plt.figure()
    plt.title("Gain Theta")
    plt.xlabel("theta [degrees]")
    plt.ylabel("phi [degrees]")
    plt.imshow(theta_gain_grid, extent=(np.min(theta), np.max(theta), np.min(phi), np.max(phi)))
    plt.figure()
    plt.title("Gain Phi")
    plt.xlabel("theta [degrees]")
    plt.ylabel("phi [degrees]")
    plt.imshow(phi_gain_grid, extent=(np.min(theta), np.max(theta), np.min(phi), np.max(phi)))

plt.figure()
plt.title("Gain")
plt.xlabel("theta [degrees]")
plt.ylabel("phi [degrees]")
plt.xlim(0, 90)
plt.ylim(0, 360)
plt.imshow(gain_grid, extent=(np.min(theta), np.max(theta), np.min(phi), np.max(phi)))
plt.show()


