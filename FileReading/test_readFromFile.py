import unittest

import numpy as np

import FileReading.readFromFile as rff


class TestReadFarfieldFile(unittest.TestCase):
    def test_readThetaAndPhiRangeParser(self):
        line = "Theta_Range: 0 1.57 11"
        start, stop, steps = rff.parseAngleRangeLine(line)
        self.assertEquals(start, 0)
        self.assertEquals(stop, 1.57)
        self.assertEquals(steps, 11)

    def test_readFarfieldScanDataFileData(self):
        # Open file and get variables
        filename = "FileReading/testFarfieldScanFile.dat"
        theta, phi, data, frequency, txPower, distance = rff.readFarfieldScanDataFile(filename)

        # Check that header data is correct
        self.assertEquals(frequency, 9182500)
        self.assertEquals(txPower, 15)
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
        filename = "FileReading/testFarfieldScanFile.dat"
        theta, phi, data, frequency, txPower, distance = rff.readFarfieldScanDataFile(filename)

        # Check if arrays are of type numpy
        self.assertEquals(np.array([0]).__class__, theta.__class__)
        self.assertEquals(np.array([0]).__class__, phi.__class__)
        self.assertEquals(np.array([0]).__class__, data.__class__)

        # Check if frequency, txPower and distance are of correct types
        self.assertEquals(int(0).__class__, frequency.__class__)
        self.assertEquals(float(0.0).__class__, txPower.__class__)
        self.assertEquals(float(0.0).__class__, distance.__class__)


class TestReadFEKOFarfieldFile(unittest.TestCase):
    def testReadFEKOFarfieldData(self):
        # Read FEKO antenna patter from file
        filename = "FileReading/testAntennaPatternFile.ffe"
        theta, phi, gainTheta, gainPhi, frequency= rff.readFEKOFarfieldDataFile(filename)

        # Check that frequency data is correct in Hz
        self.assertEquals(frequency, 9182500)

        # Check that arrays are correct
        self.assertEquals(theta[0], 0.00000000E+00)
        self.assertEquals(theta[-1], 1.80000000E+02)
        self.assertEquals(len(theta), 37*73)
        self.assertEquals(phi[0], 0.00000000E+00)
        self.assertEquals(phi[-1], 3.60000000E+02)
        self.assertEquals(len(phi), 37*73)
        self.assertEquals(gainTheta[0], -1.00000000E+03)
        self.assertEquals(gainTheta[-1], -3.10789956E+02)
        self.assertEquals(len(gainTheta), 37*73)
        self.assertEquals(gainPhi[0], 1.42922660E+00)
        self.assertEquals(gainPhi[-1], 1.42922655E+00)
        self.assertEquals(len(gainPhi), 37*73)

    def testReadFEKOFarfieldDataReturnTypes(self):
        # Read FEKO antenna patter from file
        filename = "FileReading/testAntennaPatternFile.ffe"
        theta, phi, gainTheta, gainPhi, frequency = rff.readFEKOFarfieldDataFile(filename)

        # Check if arrays are of type numpy
        self.assertEquals(np.array([0]).__class__, theta.__class__)
        self.assertEquals(np.array([0]).__class__, phi.__class__)
        self.assertEquals(np.array([0]).__class__, gainTheta.__class__)
        self.assertEquals(np.array([0]).__class__, gainPhi.__class__)

        # Check if frequency, txPower and distance are of correct types
        self.assertEquals(float(0).__class__, frequency.__class__)


class TestReadFEKONearfieldFile(unittest.TestCase):

    def testReadFEKONearfieldData(self):
        # Read FEKO nearfield from file
        filename = "FileReading/testNearfieldPatternFile.efe"
        x, y, z, Ex, Ey, Ez, frequency = rff.readFEKONearfieldDataFile(filename)

        # Check that frequency data is correct in Hz
        self.assertEquals(frequency, 1.00000000e9)

        # Check that arrays are correct
        self.assertEquals(x[0], -11.5)
        self.assertEquals(x[-1], 11.5)
        self.assertEquals(len(x), 16*16*1)
        self.assertEquals(y[0], -11.5)
        self.assertEquals(y[-1], 11.5)
        self.assertEquals(len(y), 16*16*1)
        self.assertAlmostEquals(z[0], 2.69063189)
        self.assertAlmostEquals(z[-1], 2.69063189)
        self.assertEquals(len(z), 16*16*1)
        self.assertEquals(Ex[0], -1.54041147E-02 + 1j*4.01204394E-03)
        self.assertEquals(Ex[-1], -1.53141019E-02 + 1j*3.95306420E-03)
        self.assertEquals(len(Ex), 16*16*1)
        self.assertEquals(Ey[0], 1.47913264E-02 - 1j*3.92015931E-03)
        self.assertEquals(Ey[-1], 1.47058147E-02 - 1j*3.86364258E-03)
        self.assertEquals(len(Ey), 16*16*1)
        self.assertEquals(Ez[0], -2.83216437E-03 + 1j*1.08340531E-03)
        self.assertEquals(Ez[-1], 2.81352121E-03 - 1j*1.07024549E-03)
        self.assertEquals(len(Ez), 16*16*1)

    def testReadFEKONearfieldDataReturnTypes(self):
        # Read FEKO antenna patter from file
        filename = "FileReading/testNearfieldPatternFile.efe"
        x, y, z, Ex, Ey, Ez, frequency = rff.readFEKONearfieldDataFile(filename)

        # Check if arrays are of type numpy
        self.assertEquals(np.array([0]).__class__, x.__class__)
        self.assertEquals(np.array([0]).__class__, y.__class__)
        self.assertEquals(np.array([0]).__class__, z.__class__)
        self.assertEquals(np.array([0]).__class__, Ex.__class__)
        self.assertEquals(np.array([0]).__class__, Ey.__class__)
        self.assertEquals(np.array([0]).__class__, Ez.__class__)

        # Check if frequency, txPower and distance are of correct types
        self.assertEquals(float(0).__class__, frequency.__class__)
if __name__ == '__main__':
    unittest.main()