import unittest

import numpy as np

import readFromFile as rFF


class TestReadFarfieldFile(unittest.TestCase):
    def test_readThetaAndPhiRangeParser(self):
        line = "Theta_Range: 0 1.57 11"
        start, stop, steps = rFF.parse_angle_range_line(line)
        self.assertEquals(start, 0)
        self.assertEquals(stop, 1.57)
        self.assertEquals(steps, 11)

    def test_readFarfieldScanDataFileData(self):
        # Open file and get variables
        filename = "testFarfieldScanFile.dat"
        theta, phi, data, f, tx_power, distance = rFF.read_farfieldscan_datafile(filename)

        # Check that header data is correct
        self.assertEquals(f, 9182500)
        self.assertEquals(tx_power, 15)
        self.assertEquals(distance, 100)

        # Check that body data is correct
        self.assertEquals(theta[0], 0.14272727272727)
        self.assertEquals(theta[-1], 1.57)
        self.assertEquals(len(theta), 41*11)
        self.assertEquals(phi[0], 0.15317073170732)
        self.assertEquals(phi[-1], 6.28)
        self.assertEquals(len(phi), 41*11)
        self.assertEquals(data[0], -39.230606296504)
        self.assertEquals(data[-1], -26.957962712898)
        self.assertEquals(len(data), 41*11)

    def test_checkFarfieldScanDataFileReturnTypes(self):
        # Open file and get variables
        filename = "testFarfieldScanFile.dat"
        theta, phi, data, f, tx_power, distance = rFF.read_farfieldscan_datafile(filename)

        # Check if arrays are of type numpy
        self.assertEquals(np.array([0]).__class__, theta.__class__)
        self.assertEquals(np.array([0]).__class__, phi.__class__)
        self.assertEquals(np.array([0]).__class__, data.__class__)

        # Check if f, txPower and distance are of correct types
        self.assertEquals(int(0).__class__, f.__class__)
        self.assertEquals(float(0.0).__class__, tx_power.__class__)
        self.assertEquals(float(0.0).__class__, distance.__class__)


class TestReadFEKOFarfieldFile(unittest.TestCase):
    def testReadFEKOFarfieldData(self):
        # Read FEKO antenna patter from file
        filename = "testAntennaPatternFile.ffe"
        f, theta, phi, gain_theta, gain_phi, no_samples = rFF.read_fekofarfield_datafile(filename)

        # Check that the no_samples vector is correct
        self.assertEquals(no_samples[0], 3)
        self.assertEquals(no_samples[1], 3)
        self.assertEquals(no_samples[2], 3)
        len_of_array = no_samples[0]*no_samples[1]*no_samples[2]
        len_of_block = no_samples[1]*no_samples[2]

        # Check that arrays are correct
        self.assertEquals(f[0], 9.18250000E+06)
        self.assertEquals(f[-1], 11182500.0 )
        self.assertEquals(len(f), len_of_array)
        self.assertEquals(theta[0], 0.00000000E+00)
        self.assertEquals(theta[len_of_block-1], 4.50000000E+01)
        self.assertEquals(theta[len_of_block], 1.00000000E+00)
        self.assertEquals(theta[-1], 4.0000000E+01)
        self.assertEquals(len(theta), len_of_array)
        self.assertEquals(phi[0], 0.00000000E+00)
        self.assertEquals(phi[-1], 1.0000000E+00)
        self.assertEquals(len(phi), len_of_array)
        self.assertEquals(gain_theta[0], -1.00000000E+03)
        self.assertEquals(gain_theta[-1], -11.0000000E+03)
        self.assertEquals(len(gain_theta), len_of_array)
        self.assertEquals(gain_phi[0], 1.42922660E+00)
        self.assertEquals(gain_phi[-1], 1.42922917E+00)
        self.assertEquals(len(gain_phi), len_of_array)

    def testReadFEKOFarfieldDataReturnTypes(self):
        # Read FEKO antenna patter from file
        filename = "testAntennaPatternFile.ffe"
        f, theta, phi, gain_theta, gain_phi, no_samples = rFF.read_fekofarfield_datafile(filename)

        # Check if arrays are of type numpy
        self.assertEquals(np.array([0]).__class__, f.__class__)
        self.assertEquals(np.array([0]).__class__, theta.__class__)
        self.assertEquals(np.array([0]).__class__, phi.__class__)
        self.assertEquals(np.array([0]).__class__, gain_theta.__class__)
        self.assertEquals(np.array([0]).__class__, gain_phi.__class__)


class TestReadFEKONearfieldFile(unittest.TestCase):

    def testReadFEKONearfieldData(self):
        # Read FEKO nearfield from file
        filename = "testNearfieldPatternFile.efe"
        f, x, y, z, ex, ey, ez, no_samples = rFF.read_fekonearfield_datafile(filename)

        # Check that the no_samples vector is correct
        self.assertEquals(no_samples[0], 2)
        self.assertEquals(no_samples[1], 3)
        self.assertEquals(no_samples[2], 3)
        self.assertEquals(no_samples[3], 1)
        len_of_array = no_samples[0]*no_samples[1]*no_samples[2]*no_samples[3]
        len_of_block = no_samples[1]*no_samples[2]*no_samples[3]

        # Check that arrays are correct
        self.assertEquals(f[0], 1E9)
        self.assertEquals(f[-1], 2E9)
        self.assertEquals(len(f), len_of_array)

        self.assertEquals(x[0], -11.5)
        self.assertEquals(x[-1], 7.66666667E-01)
        self.assertEquals(len(x), len_of_array)

        self.assertEquals(y[0], -11.5)
        self.assertEquals(y[-1], -11.6)
        self.assertEquals(len(y), len_of_array)

        self.assertAlmostEquals(z[0], 2.69063189)
        self.assertAlmostEquals(z[-1], 2.69063189)
        self.assertEquals(len(z), len_of_array)

        self.assertEquals(ex[0], -1.54041147E-02 + 1j*4.01204394E-03)
        self.assertEquals(ex[len_of_block-1], 2.73158692E-02 + 1j*8.59837337E-03)
        self.assertEquals(ex[len_of_block], -1.54041147E-02 + 1j*4.01204394E-03)
        self.assertEquals(ex[-1], 2.73158692E-02 + 1j*8.59837337E-03)
        self.assertEquals(len(ex), len_of_array)

        self.assertEquals(ey[0], 1.47913264E-02 - 1j*3.92015931E-03)
        self.assertEquals(ey[-1], 1.74954795E-03 + 1j*5.76822063E-04)
        self.assertEquals(len(ey), len_of_array)

        self.assertEquals(ez[0], -2.83216437E-03 + 1j*1.08340531E-03)
        self.assertEquals(ez[-1], -3.57347990E-04 + 1j*-2.41266729E-06)
        self.assertEquals(len(ez), len_of_array)

    def testReadFEKONearfieldDataReturnTypes(self):
        # Read FEKO antenna patter from file
        filename = "testNearfieldPatternFile.efe"
        x, y, z, ex, ey, ez, f, no_samples = rFF.read_fekonearfield_datafile(filename)

        # Check if arrays are of type numpy
        self.assertEquals(np.array([0]).__class__, f.__class__)
        self.assertEquals(np.array([0]).__class__, x.__class__)
        self.assertEquals(np.array([0]).__class__, y.__class__)
        self.assertEquals(np.array([0]).__class__, z.__class__)
        self.assertEquals(np.array([0]).__class__, ex.__class__)
        self.assertEquals(np.array([0]).__class__, ey.__class__)
        self.assertEquals(np.array([0]).__class__, ez.__class__)

        i = int(0)
        self.assertEquals(i.__class__, no_samples[0].__class__)
        self.assertEquals(i.__class__, no_samples[1].__class__)
        self.assertEquals(i.__class__, no_samples[2].__class__)
        self.assertEquals(i.__class__, no_samples[3].__class__)


class TestFEKOFileUtilities(unittest.TestCase):

    def testReadFrequencyBlockFromDataset(self):
        theta = [0, 1, 2, 0, 1, 2, 0, 1, 2,
                 0, 1, 2, 0, 1, 2, 0, 1, 2,
                 0, 1, 2, 0, 1, 2, 0, 1, 2]
        phi = [0, 0, 0, 1, 1, 1, 2, 2, 2,
               0, 0, 0, 1, 1, 1, 2, 2, 2,
               0, 0, 0, 1, 1, 1, 2, 2, 2]
        gain_theta = [0, 1, 2, 3, 4, 5, 6, 7, 8,
                      9, 10, 11, 12, 13, 14, 15, 16, 17,
                      18, 19, 20, 21, 22, 23, 24, 25, 26]
        gain_phi = [8, 7, 6, 5, 4, 3, 2, 1, 0,
                    17, 16, 15, 14, 13, 12, 11, 10, 9,
                    26, 25, 24, 23, 22, 21, 20, 19, 18]
        no_samples = [3, 3, 3]
        block_no = 0
        theta_block, phi_block, gain_theta_block, gain_phi_block = rFF.read_frequency_block_from_farfield_dataset(
            block_no,
            no_samples,
            theta,
            phi,
            gain_theta,
            gain_phi)

        self.assertEqual(theta_block, [0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.assertEqual(phi_block, [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual(gain_theta_block, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(gain_phi_block, [8, 7, 6, 5, 4, 3, 2, 1, 0])

        x = [0, 1, 2, 0, 1, 2, 0, 1, 2,
             0, 1, 2, 0, 1, 2, 0, 1, 2,
             0, 1, 2, 0, 1, 2, 0, 1, 2]
        y = [0, 0, 0, 1, 1, 1, 2, 2, 2,
             0, 0, 0, 1, 1, 1, 2, 2, 2,
             0, 0, 0, 1, 1, 1, 2, 2, 2]
        z = [0, 1, 2, 3, 4, 5, 6, 7, 8,
             9, 10, 11, 12, 13, 14, 15, 16, 17,
             18, 19, 20, 21, 22, 23, 24, 25, 26]
        data_x = [0, 7, 6, 5, 4, 3, 2, 1, 0,
                  17, 16, 15, 14, 13, 12, 11, 10, 9,
                  26, 25, 24, 23, 22, 21, 20, 19, 18]
        data_y = [1, 7, 6, 5, 4, 3, 2, 1, 0,
                  17, 16, 15, 14, 13, 12, 11, 10, 9,
                  26, 25, 24, 23, 22, 21, 20, 19, 18]
        data_z = [2, 7, 6, 5, 4, 3, 2, 1, 0,
                  17, 16, 15, 14, 13, 12, 11, 10, 9,
                  26, 25, 24, 23, 22, 21, 20, 19, 18]
        no_samples = [3, 3, 3, 3]
        block_no = 0
        x_block, y_block, z_block, x_data_block, y_data_block, z_data_block =\
            rFF.read_frequency_block_from_nearfield_dataset(block_no,
                                                            no_samples,
                                                            x,
                                                            y,
                                                            z,
                                                            data_x,
                                                            data_y,
                                                            data_z)

        self.assertEqual(x_block, [0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.assertEqual(y_block, [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual(z_block, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(x_data_block, [0, 7, 6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(y_data_block, [1, 7, 6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(z_data_block, [2, 7, 6, 5, 4, 3, 2, 1, 0])

    def testExtractPhiCutFromCoordData(self):
        theta = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        phi = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        gain = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        no_samples = np.array([1, 3, 3])

        phi_block_no = 0
        cut = rFF.get_phi_cut_from_coord_data(phi_block_no, no_samples, gain)
        self.assertEqual(cut[0], 0)
        self.assertEqual(cut[2], 2)

        phi_block_no = 2
        cut = rFF.get_phi_cut_from_coord_data(phi_block_no, no_samples, gain)
        self.assertEqual(cut[0], 6)
        self.assertEqual(cut[2], 8)

    def testExtractPhiCutFromGridData(self):
        theta = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        phi = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        gain = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

        phi_block_no = 0
        cut = rFF.get_phi_cut_from_grid_data(phi_block_no, gain)
        self.assertEqual(cut[0], 0)
        self.assertEqual(cut[2], 2)

        phi_block_no = 2
        cut = rFF.get_phi_cut_from_grid_data(phi_block_no, gain)
        self.assertEqual(cut[0], 6)
        self.assertEqual(cut[2], 8)

    def test_transform_data_from_coord_to_grid_form(self):
        """Test the transformation of data from coordinate form to grid form"""
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        x_length = 3
        y_length = 4
        grid_data = rFF.transform_data_coord_to_grid(x_length, y_length, x)

        grid_data_test = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        self.assertEqual(grid_data_test.__class__, grid_data.__class__)
        self.assertEqual(len(grid_data), len(grid_data_test))
        self.assertEqual(len(grid_data[0]), len(grid_data_test[0]))
        self.assertEqual(np.sum(grid_data-grid_data_test), 0)


if __name__ == '__main__':
    unittest.main()
