import FileReading.readFromFile as rff
import numpy as np
import matplotlib.pyplot as plt

"""
Read in the nearfield of a yagi antenna
"""

x, y, z, Ex, Ey, Ez, frequency = rff.readFEKONearfieldDataFile("NFSimulations/YagiNearfield2m.efe")

#Put data into 2D matrix
#Show data with imshow
#Transform data using 2D fourier transform
#Compare with feko farfield export
#Evaluate difference when truncating
