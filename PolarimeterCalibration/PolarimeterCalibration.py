"""
Experimentation into the calibration of polarimeters
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from PolarimeterCalibration import Stokes


def addNoise(vn):
    return ((np.random.ranf()*vn-vn/2) + (np.random.ranf()*vn-vn/2)*1j)


'''Noise magnitude'''
systemTemp = 273 + 20  # K
bandwidth = 12e3  # Hz
resistance = 50  # Ohm
kb = 1.38064852e-23
vn = np.sqrt(4*kb*systemTemp*resistance*bandwidth)
print("Noise voltage: " + str(vn))

'''Standards in terms of Stokes parameters'''
# NS Plane Wave
ex = 0
ey = -1
sNSStandard = Stokes.volts_to_stokes(ex, ey)

# EW Plane Wave
ex = 1
ey = 0
sEWStandard = Stokes.volts_to_stokes(ex, ey)

# NE Plane Wave
ex = 7.07107e-1
ey = -7.07107e-1
sNEStandard = Stokes.volts_to_stokes(ex, ey)

# RHCP Plane Wave
ex = 1
ey = -1j
sRHCPStandard = Stokes.volts_to_stokes(ex, ey)

# LHCP Plane Wave
ex = 1
ey = 1j
sLHCPStandard = Stokes.volts_to_stokes(ex, ey)

# 10Deg Plane Wave
ex = 1.74e-1
ey = -9.85e-1
s10DegStandard = Stokes.volts_to_stokes(ex, ey)

'''Measurements in terms of Stokes parameters'''
# NS Plane Wave
ex = 0.1866e-8 + 1j*0.6527e-9 + addNoise(vn)
ey = 0.2288e-2 + 1j*0.1159e-1 + addNoise(vn)
sNSMeasured = Stokes.volts_to_stokes(ex, ey)

# EW Plane Wave
ex = 0.6688e-2 - 1j*0.8094e-2 + addNoise(vn)
ey = 0.1925e-8 + 1j*0.1645e-8 + addNoise(vn)
sEWMeasured = Stokes.volts_to_stokes(ex, ey)

# NE Plane Wave
ex = 0.4729e-2 - 1j*0.5723e-2 + addNoise(vn)
ey = 0.1618e-2 + 1j*0.8192e-2 + addNoise(vn)
sNEMeasured = Stokes.volts_to_stokes(ex, ey)

# RHCP Plane Wave
ex = 0.6688e-2 - 1j*0.8094e-2 + addNoise(vn)
ey = -0.1159e-1 + 1j*0.2288e-2 + addNoise(vn)
sRHCPMeasured = Stokes.volts_to_stokes(ex, ey)

# LHCP Plane Wave
ex = 0.6688e-2 - 1j*0.8094e-2 + addNoise(vn)
ey = 0.1159e-1 - 1j*0.2288e-2 + addNoise(vn)
sLHCPMeasured = Stokes.volts_to_stokes(ex, ey)

# 10Deg Plane Wave
ex = 0.1161e-2 - 1j*0.1406e-2 + addNoise(vn)
ey = 0.2253e-2 + 1j*0.1141e-1 + addNoise(vn)
s10DegMeasured = Stokes.volts_to_stokes(ex, ey)

"""Calculate the inversion matrix from calibration measurements"""
#  Build the transmission matrix from the measurements S' = TS
Ss = np.transpose([sNSStandard, sNEStandard, sRHCPStandard, sLHCPStandard])
Sm = np.transpose([sNSMeasured, sNEMeasured, sRHCPMeasured, sLHCPMeasured])

#  Correction matrix
SsInv = np.linalg.inv(Ss)
T = np.dot(Sm, SsInv)
TInv = np.linalg.inv(T)


"""Apply corrections to measurements"""
#  S = TInv*S'
s10DegExtracted = np.dot(TInv, s10DegMeasured)

print(Stokes.polarised_angle(s10DegStandard) * 180 / np.pi)
print(Stokes.polarised_angle(s10DegExtracted) * 180 / np.pi)


'''Calculate jones matrices and from that calculate the IXR of the antenna'''
#condNumber = np.linalg.norm(T)*np.linalg.norm(TInv)
#IXR = ((condNumber + 1)/(condNumber - 1))**2
#print(IXR)

'''Visualise how close the array is to an identity matrix'''
plt.imshow(TInv/np.max(TInv), interpolation="nearest", cmap=cm.Greys)
plt.show()