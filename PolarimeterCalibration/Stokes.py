"""
Conversion functions for measured voltage to stokes parameters
Radio Astronomy - Kraus

Hardie Pienaar
19 Feb 2016
"""

import numpy as np


def volts_to_stokes(ex, ey):
    """
    Returns the stokes parameters for given E-field components in cartesian coordinates
    :param ex: East-West E-field component
    :param ey: North-South E-field component
    :return: stokes parameters I, Q, U, V
    """
    s = list()
    s.append(np.abs(ex)**2 + np.abs(ey)**2)  # I
    s.append(np.abs(ex)**2 - np.abs(ey)**2)  # Q
    s.append(2*np.real(ex*np.conjugate(ey)))  # U
    s.append(-2*np.imag(ex*np.conjugate(ey)))  # V
    return s


def linear_polarised_intensity(s):
    """
    Returns the polarisation intensity of the given stokes parameters
    :param s: stokes parameters I, Q, U, V
    :return: polarisation intensity
    """
    p = np.sqrt(s[1]**2 + s[2]**2)
    return p


def circular_polarised_intensity(s):
    """
    Returns the polarisation intensity of the given stokes parameters
    :param s: stokes parameters I, Q, U, V
    :return: polarisation intensity
    """
    p = np.sqrt(s[3]**2)
    return p


def total_polarised_intensity(s):
    """
    Returns the polarisation intensity of the given stokes parameters
    :param s: stokes parameters I, Q, U, V
    :return: polarisation intensity
    """
    p = np.sqrt(s[1]**2 + s[2]**2 + s[3]**2)
    return p


def polarised_angle(s):
    """
    Returns the polarisation angle of the given stokes parameters.
    Returns 0 for purely circularly polarised waves
    :param s: stokes parameters I, Q, U, V
    :return: polarisation intensity
    """

    if s[1] == 0:
        if s[2] > 0:
            return 0.5*np.pi/2
        elif s[2] == 0:
            return 0
        else:
            return -0.5*np.pi/2
    pa = 0.5*np.arctan(s[2]/s[1])

    return pa

def todo():
    #  Add calibration functions (T calculation for given standards, extraction method given T)
    return 0