"""
Read data from file. This class has different methods for different file types.

Hardie Pienaar
1/2/2016
"""

import csv
import numpy as np


def read_farfieldscan_datafile(filename):
    """
    Open data file and read in relevant parameters into their objects,
    also detect if the file is corrupted or not in the right format
    :param filename - name of the datafile, needs to be in the farfield scan format
    """

    # Constants relevant to the file structure
    header_size = 7
    theta_range_line = 2
    phi_range_line = 3
    tx_power_line = 4
    distance_line = 5

    # Open csv reader
    file_handle = open(filename)
    reader = csv.reader(file_handle, delimiter=' ')

    # Create array data structures
    theta = []
    phi = []
    data = []

    # Read header and body of file
    line_no = 0
    for row in reader:
        string = '    '.join(row)
        if line_no == theta_range_line:
            theta_start, theta_stop, theta_steps = parseAngleRangeLine(string)
        elif line_no == phi_range_line:
            phi_start, phi_stop, phi_steps = parseAngleRangeLine(string)
        elif line_no == tx_power_line:
            tx_power = float(string.split()[1])
        elif line_no == distance_line:
            distance = float(string.split()[1])
        elif line_no == header_size:
            frequency = int(string.split()[0])
        elif line_no > header_size:
            elements = string.split()
            theta.append(float(elements[0]))
            phi.append(float(elements[1]))
            data.append(float(elements[2]))
        line_no += 1

    # Close file after reading
    file_handle.close()

    # Convert arrays to numpy array types
    theta = np.array(theta)
    phi = np.array(phi)
    data = np.array(data)

    return theta, phi, data, frequency, tx_power, distance


def read_fekofarfield_datafile(filename):
    """
    Read FEKO antenna farfield pattern from file
    :param filename - name of datafile, needs to be a .ffe file from feko
    """

    # Header structure constants
    frequency_line = 1

    # Open farfield CSV file
    file_handle = open(filename)
    reader = csv.reader(file_handle, delimiter=' ')

    frequency = 0
    theta = []
    phi = []
    gain_theta = []
    gain_phi = []

    # Read through header and body
    line_no = 0
    found_header_start = False
    found_body_start = False
    for row in reader:
        string = '    '.join(row)

        # Only start reading when header start has been found
        if string.__contains__("#Request"):
            found_header_start = True

        if found_header_start and not string.__contains__('#'):
            found_body_start = True

        # Read frequency from header
        if line_no == frequency_line:
            frequency = float(string.split()[1])

        # Read body into arrays
        if found_body_start and len(string) > 0:
            elements = string.split()
            theta.append(float(elements[0]))
            phi.append(float(elements[1]))
            gain_theta.append(float(elements[6]))
            gain_phi.append(float(elements[7]))

        if found_header_start:
            line_no += 1

    theta = np.array(theta)
    phi = np.array(phi)
    gain_theta = np.array(gain_theta)
    gain_phi = np.array(gain_phi)

    return theta, phi, gain_theta, gain_phi, frequency

def readFEKONearfieldDataFile(filename):
    # Header structure constants
    FREQUENCY_LINE = 1
    NO_EX_LINE = 3
    NO_EY_LINE = 4
    NO_EZ_LINE = 5

    # Open farfield CSV file
    file = open(filename)
    reader = csv.reader(file, delimiter=' ')

    frequency = 0
    no_f_samples = 0
    f = []
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

        # Read header contents
        if lineNo == FREQUENCY_LINE:
            frequency = float(string.split()[1])
            no_f_samples += 1
        if lineNo == NO_EX_LINE:
            no_ex_samples = float(string.split()[4])
        if lineNo == NO_EY_LINE:
            no_ey_samples = float(string.split()[4])
        if lineNo == NO_EZ_LINE:
            no_ez_samples = float(string.split()[4])

        # Read body into arrays
        if foundBodyStart and len(string) > 0:
            elements = string.split()
            f.append(frequency)
            x.append(float(elements[0]))
            y.append(float(elements[1]))
            z.append(float(elements[2]))
            Ex.append(float(elements[3]) + float(elements[4])*1j)
            Ey.append(float(elements[5]) + float(elements[6])*1j)
            Ez.append(float(elements[7]) + float(elements[8])*1j)

            # Check if this is the last row in the frequency
            if len(x) == no_ex_samples * no_ey_samples * no_ez_samples:
                foundBodyStart = False
                foundHeaderStart = False
                lineNo = 0

        if foundHeaderStart:
            lineNo += 1

    file.close()

    f = np.array(f)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    Ex = np.array(Ex)
    Ey = np.array(Ey)
    Ez = np.array(Ez)

    return f, x, y, z, Ex, Ey, Ez, [no_f_samples, no_ex_samples, no_ey_samples, no_ez_samples]

def parseAngleRangeLine(line):
    """Parse header line containing start, stop and number of steps"""
    elements = str(line).split()
    return float(elements[1]), float(elements[2]), float(elements[3])

