from FileReading import readFromFile as rff

"""
Read in the farfield of an array dipole antenna
"""

filename = "DDA_DualPol_FF_noFB.ffe"
f, theta, phi, gain_theta, gain_phi, no_samples = rff.read_fekofarfield_datafile(filename)

