"""
Read data from file. This class has different methods for different file types.

Hardie Pienaar
1/2/2016
"""

import csv
import numpy as np


def readFarfieldScanDataFile(filename):
    """Open data file and read in relevant parameters into their objects,
    also detect if the file is corrupted or not in the right format"""

    # Constants relevant to the file structure
    HEADER_SIZE = 7
    THETA_RANGE_LINE = 2
    PHI_RANGE_LINE = 3
    TX_POWER_LINE = 4
    DISTANCE_LINE = 5

    # Open csv reader
    file = open(filename)
    reader = csv.reader(file, delimiter=' ')

    # Create array data structures
    theta = []
    phi = []
    data = []

    # Read header and body of file
    lineNo = 0
    for row in reader:
        string = '    '.join(row)
        if lineNo == THETA_RANGE_LINE:
            thetaStart, thetaStop, thetaSteps = parseAngleRangeLine(string)
        elif lineNo == PHI_RANGE_LINE:
            phiStart, phiStop, phiSteps = parseAngleRangeLine(string)
        elif lineNo == TX_POWER_LINE:
            txPower = float(string.split()[1])
        elif lineNo == DISTANCE_LINE:
            distance = float(string.split()[1])
        elif lineNo == HEADER_SIZE:
            frequency = int(string.split()[0])
        elif lineNo > HEADER_SIZE:
            elements = string.split()
            theta.append(float(elements[0]))
            phi.append(float(elements[1]))
            data.append(float(elements[2]))
        lineNo += 1

    # Close file after reading
    file.close()

    # Convert arrays to numpy array types
    theta = np.array(theta)
    phi = np.array(phi)
    data = np.array(data)

    return theta, phi, data, frequency, txPower, distance


def readFEKOFarfieldDataFile(filename):
    """Read FEKO antenna farfield pattern from file"""

    # Header structure constants
    FREQUENCY_LINE = 1

    # Open farfield CSV file
    file = open(filename)
    reader = csv.reader(file, delimiter=' ')

    frequency = 0
    theta = []
    phi = []
    gainTheta = []
    gainPhi = []

    # Read through header and body
    lineNo = 0
    foundHeaderStart = False
    foundBodyStart = False
    for row in reader:
        string = '    '.join(row)

        # Only start reading when header start has been found
        if string.__contains__("#Request"):
            foundHeaderStart = True

        if foundHeaderStart and not string.__contains__('#'):
            foundBodyStart = True

        # Read frequency from header
        if lineNo == FREQUENCY_LINE:
            frequency = float(string.split()[1])

        # Read body into arrays
        if foundBodyStart and len(string) > 0:
            elements = string.split()
            theta.append(float(elements[0]))
            phi.append(float(elements[1]))
            gainTheta.append(float(elements[6]))
            gainPhi.append(float(elements[7]))

        if foundHeaderStart:
            lineNo += 1

    theta = np.array(theta)
    phi = np.array(phi)
    gainTheta = np.array(gainTheta)
    gainPhi = np.array(gainPhi)

    return theta, phi, gainTheta, gainPhi, frequency

def readFEKONearfieldDataFile(filename):
    # Header structure constants
    FREQUENCY_LINE = 1

    # Open farfield CSV file
    file = open(filename)
    reader = csv.reader(file, delimiter=' ')

    frequency = 0
    x = []
    y = []
    z = []
    Ex = []
    Ey = []
    Ez = []

    # Read through header and body
    lineNo = 0
    foundHeaderStart = False
    foundBodyStart = False
    for row in reader:
        string = '    '.join(row)

        # Only start reading when header start has been found
        if string.__contains__("#Request"):
            foundHeaderStart = True

        if foundHeaderStart and not string.__contains__('#'):
            foundBodyStart = True

        # Read frequency from header
        if lineNo == FREQUENCY_LINE:
            frequency = float(string.split()[1])

        # Read body into arrays
        if foundBodyStart and len(string) > 0:
            elements = string.split()
            x.append(float(elements[0]))
            y.append(float(elements[1]))
            z.append(float(elements[2]))
            Ex.append(float(elements[3]) + float(elements[4])*1j)
            Ey.append(float(elements[5]) + float(elements[6])*1j)
            Ez.append(float(elements[7]) + float(elements[8])*1j)

        if foundHeaderStart:
            lineNo += 1

    file.close()

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    Ex = np.array(Ex)
    Ey = np.array(Ey)
    Ez = np.array(Ez)

    return x, y, z, Ex, Ey, Ez, frequency


def parseAngleRangeLine(line):
    """Parse header line containing start, stop and number of steps"""
    elements = str(line).split()
    return float(elements[1]), float(elements[2]), float(elements[3])

