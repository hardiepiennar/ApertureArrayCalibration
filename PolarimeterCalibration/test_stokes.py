import unittest

from PolarimeterCalibration import Stokes


class TestStokesFunctions(unittest.TestCase):
    def test_convertVoltagesToStokes(self):
        """
        Test the conversion of E-field values to stokes parameters
        """

        # Test for a left circularly polarised wave
        ex = 1
        ey = 1j
        s = Stokes.volts_to_stokes(ex, ey)

        self.assertAlmostEqual(s[0], 2)
        self.assertAlmostEqual(s[1], 0)
        self.assertAlmostEqual(s[2], 0)
        self.assertAlmostEqual(s[3], 2)

        # Test for a right circularly polarised wave
        ex = 1
        ey = -1j
        s = Stokes.volts_to_stokes(ex, ey)

        self.assertAlmostEqual(s[0], 2)
        self.assertAlmostEqual(s[1], 0)
        self.assertAlmostEqual(s[2], 0)
        self.assertAlmostEqual(s[3], -2)

        # Test for a linear polarised wave
        ex = 1
        ey = 0
        s = Stokes.volts_to_stokes(ex, ey)

        self.assertAlmostEqual(s[0], 1)
        self.assertAlmostEqual(s[1], 1)
        self.assertAlmostEqual(s[2], 0)
        self.assertAlmostEqual(s[3], 0)

    def test_linear_polarised_intensity(self):
        """
        Test the calculation of polarised intensity
        """

        #  Test a linear polarised wave
        s = [1, 1, 0, 0]
        p = Stokes.linear_polarised_intensity(s)
        self.assertAlmostEqual(p, 1)

        #  Test a circularly polarised wave
        s = [1, 0, 0, 1]
        p = Stokes.linear_polarised_intensity(s)
        self.assertAlmostEqual(p, 0)

    def test_circular_polarised_intensity(self):
        """
        Test the calculation of polarised intensity
        """

        #  Test a linear polarised wave
        s = [1, 1, 0, 0]
        p = Stokes.circular_polarised_intensity(s)
        self.assertAlmostEqual(p, 0)

        #  Test a circularly polarised wave
        s = [1, 0, 0, 1]
        p = Stokes.circular_polarised_intensity(s)
        self.assertAlmostEqual(p, 1)

    def test_total_polarised_intensity(self):
        """
        Test the calculation of polarised intensity
        """

        #  Test a linear polarised wave
        s = [1, 1, 0, 0]
        p = Stokes.total_polarised_intensity(s)
        self.assertAlmostEqual(p, 1)

        #  Test a circularly polarised wave
        s = [1, 0, 0, 0]
        p = Stokes.total_polarised_intensity(s)
        self.assertAlmostEqual(p, 0)

    def test_polarised_angle(self):
        """
        Test the calculation of polarised angle
        """

        #  Test a linear polarised wave
        s = [1, 1, 0, 0]  # EW - 0 angle
        pa = Stokes.polarised_angle(s)
        self.assertAlmostEqual(pa, 0)
        s = [1, -1, 0, 0]
        pa = Stokes.polarised_angle(s)
        self.assertAlmostEqual(pa, 0)

        #  Test a circularly polarised wave
        s = [1, 0, 0, 1]
        pa = Stokes.polarised_angle(s)
        self.assertAlmostEqual(pa, 0)

if __name__ == '__main__':
    unittest.main()
