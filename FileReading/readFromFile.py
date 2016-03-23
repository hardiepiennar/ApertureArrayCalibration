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
    frequency = 0
    distance = 0
    tx_power = 0
    theta = []
    phi = []
    data = []

    # Read header and body of file
    line_no = 0
    for row in reader:
        string = '    '.join(row)
        if line_no == theta_range_line:
            # theta_start, theta_stop, theta_steps = parse_angle_range_line(string)
            pass
        elif line_no == phi_range_line:
            # phi_start, phi_stop, phi_steps = parse_angle_range_line(string)
            pass
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

    return [theta, phi, data, frequency, tx_power, distance]


def read_fekofarfield_datafile(filename):
    """
    Read FEKO antenna farfield pattern from file
    :param filename - name of datafile, needs to be a .ffe file from feko
    """

    # Header structure constants
    frequency_line = 1
    no_theta_line = 3
    no_phi_line = 4

    # Open farfield CSV file
    file_handle = open(filename)
    reader = csv.reader(file_handle, delimiter=' ')

    frequency = 0
    no_f_samples = 0
    no_theta_samples = 0
    no_phi_samples = 0
    f = []
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
            no_f_samples += 1
            print(frequency)
        if line_no == no_theta_line:
            no_theta_samples = float(string.split()[4])
        if line_no == no_phi_line:
            no_phi_samples = float(string.split()[4])

        # Read body into arrays
        if found_body_start and len(string) > 0:
            elements = string.split()
            f.append(frequency)
            theta.append(float(elements[0]))
            phi.append(float(elements[1]))
            gain_theta.append(float(elements[6]))
            gain_phi.append(float(elements[7]))

            # Check if this is the last row in the frequency
            if len(theta) % (no_theta_samples * no_phi_samples) == 0:
                found_body_start = False
                found_header_start = False
                line_no = 0

        if found_header_start:
            line_no += 1

    f = np.array(f)
    theta = np.array(theta)
    phi = np.array(phi)
    gain_theta = np.array(gain_theta)
    gain_phi = np.array(gain_phi)

    return f, theta, phi, gain_theta, gain_phi, [no_f_samples, no_theta_samples, no_phi_samples]


def read_fekonearfield_datafile(filename):
    # Header structure constants
    frequency_line = 1
    no_ex_line = 3
    no_ey_line = 4
    no_ez_line = 5

    # Open farfield CSV file
    file_handle = open(filename)
    reader = csv.reader(file_handle, delimiter=' ')

    frequency = 0
    no_f_samples = 0
    no_ex_samples = 0
    no_ey_samples = 0
    no_ez_samples = 0
    f = []
    x = []
    y = []
    z = []
    ex = []
    ey = []
    ez = []

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

        # Read header contents
        if line_no == frequency_line:
            frequency = float(string.split()[1])
            no_f_samples += 1
        if line_no == no_ex_line:
            no_ex_samples = float(string.split()[4])
        if line_no == no_ey_line:
            no_ey_samples = float(string.split()[4])
        if line_no == no_ez_line:
            no_ez_samples = float(string.split()[4])

        # Read body into arrays
        if found_body_start and len(string) > 0:
            elements = string.split()
            f.append(frequency)
            x.append(float(elements[0]))
            y.append(float(elements[1]))
            z.append(float(elements[2]))
            ex.append(float(elements[3]) + float(elements[4])*1j)
            ey.append(float(elements[5]) + float(elements[6])*1j)
            ez.append(float(elements[7]) + float(elements[8])*1j)

            # Check if this is the last row in the frequency
            if len(x) % (no_ex_samples * no_ey_samples * no_ez_samples) == 0:
                found_body_start = False
                found_header_start = False
                line_no = 0

        if found_header_start:
            line_no += 1

    file_handle.close()

    f = np.array(f)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    ex = np.array(ex)
    ey = np.array(ey)
    ez = np.array(ez)

    return f, x, y, z, ex, ey, ez, [no_f_samples, no_ex_samples, no_ey_samples, no_ez_samples]


def parse_angle_range_line(line):
    """
    Parse header line containing start, stop and number of steps
    :param line - angle range text line from farfield file that needs to be parsed
    """
    elements = str(line).split()
    return float(elements[1]), float(elements[2]), float(elements[3])
