import unittest
import numpy as np
import NF2FF as nf2ff


class NF2FFTestCases(unittest.TestCase):
    def test_calc_recommended_sampling_param(self):
        """Test the recommended sampling parameters function"""
        d = 0.42
        m = 85*(np.pi/180)
        p = 0.15
        z = 1
        lambd = 0.3

        probe_gain, scan_width, no_samples_per_ray,sample_delta = nf2ff.calc_recommended_sampling_param(d, m, p, z, lambd)

        self.assertEqual(probe_gain, 0.5*(np.pi/np.tan(m/1.03))**2)
        self.assertEqual(scan_width, d + p + 2*z*np.tan(m))
        self.assertEqual(no_samples_per_ray, 0.5*(lambd)/(np.sin(m)*scan_width))
        self.assertEqual(sample_delta, 0.5*(no_samples_per_ray/(no_samples_per_ray + 1))*lambd/np.sin(m))

    def test_calc_nearfield_bounds(self):
        """Test the calculation of nearfield bounds function"""
        f = 1e9
        c = 3e8
        lambd = c/f
        d = 0.42

        start, stop = nf2ff.calc_nearfield_bounds(f, d)
        self.assertEquals(start, 3*lambd)
        self.assertEquals(stop, (2*d**2)/lambd)

    def test_transform_data_from_coord_to_grid_form(self):
        """Test the transformation of data from coordinate form to grid form using the scipy library"""
        x = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        v = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        coordinates = np.array([x, y])
        values = v
        res_x = 4
        res_y = 3
        grid_x, grid_y, grid_data = nf2ff.transform_data_coord_to_grid(coordinates, values, [res_x, res_y])

        grid_data_test = np.array([[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]])
        self.assertEqual(grid_data_test.__class__, grid_data.__class__)
        self.assertEqual(len(grid_data), len(grid_data_test))
        self.assertEqual(len(grid_data[0]), len(grid_data_test[0]))
        self.assertEqual(np.sum(grid_data-grid_data_test), 0)

        grid_x_test = np.array([[0, 1, 2, 3],[0, 1, 2, 3],[0, 1, 2, 3]])
        self.assertEqual(grid_x_test.__class__, grid_x.__class__)
        self.assertEqual(len(grid_x), len(grid_x_test))
        self.assertEqual(np.sum(grid_x-grid_x_test), 0)

        grid_y_test = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])
        self.assertEqual(grid_y_test.__class__, grid_y.__class__)
        self.assertEqual(len(grid_y), len(grid_y_test))
        self.assertEqual(np.sum(grid_y-grid_y_test), 0)

    def test_calculate_total_gain(self):
        gain_theta = 7.46389928E+00
        gain_phi = -1.36970651E+01
        gain = nf2ff.calculate_total_gain(gain_theta, gain_phi)
        self.assertAlmostEqual(gain, 7.49701476E+00)

    def test_calculate_total_e_field(self):
        ex = 1 + 1j
        ey = 3 + 2j
        ez = 2 + 2j
        e = nf2ff.calculate_total_e_field(ex, ey, ez)
        self.assertEqual(e, np.sqrt(ex**2 + ey**2 + ez**2))

    def test_nearfield_to_farfield(self):
        nearfield_x = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
        nearfield_y = np.array([[0, 0, 0], [0, 0.1, 0], [0, 0, 0]])
        x = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        y = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        z = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        theta = np.array([-45, 0, 45, -45, 0, 45, -45, 0, 45])
        phi = np.array([-45, -45, -45, 0, 0, 0, 45, 45, 45])

        farfield_test_theta = 2*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        farfield_test_phi = 2*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        theta_test = np.array([125.2643897, 135, 125.2643897, 135, 180, 135, 125.2643897, 135, 125.2643897])
        phi_test = np.array([225, 180, 135, 270, 0, 90, 315, 0, 45])

        trans_theta, trans_phi, trans_farfield_theta, trans_farfield_phi = nf2ff.calc_nf2ff(x, y, z, nearfield_x,
                                                                                            nearfield_y)
        for x in np.arange(len(trans_farfield_theta)):
            for y in np.arange(len(trans_farfield_theta[0])):
                self.assertAlmostEqual(trans_farfield_theta[x][y], farfield_test_theta[x][y])
        for x in np.arange(len(trans_farfield_phi)):
            for y in np.arange(len(trans_farfield_phi[0])):
                self.assertAlmostEqual(trans_farfield_phi[x][y], farfield_test_phi[x][y])
        for x in np.arange(len(trans_theta)):
                self.assertAlmostEqual(trans_theta[x], theta_test[x])
        for x in np.arange(len(trans_phi)):
                self.assertAlmostEqual(trans_phi[x], phi_test[x])

    def test_2d_discrete_fourier_transform(self):
        data = np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
        x = np.array([-1, 0, 1, 2, -1, 0, 1, 2, -1, 0, 1, 2])
        y = np.array([-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1])
        z = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        kx = np.array([-45, 0, 45, -45, 0, 45, -45, 0, 45])
        ky = np.array([-45, -45, -45, 0, 0, 0, 45, 45, 45])

        trans_data = nf2ff.calc_dft2(x, y, z, data, kx, ky)

        """Test if the transform has been calculated correctly"""
        test_trans_data = np.array([[2*np.exp(-1j*np.pi/2), 2*np.exp(-1j*np.pi/4), 2],
                                   [2*np.exp(-1j*np.pi/4), 2, 2*np.exp(1j*np.pi/4)],
                                   [2, 2*np.exp(1j*np.pi/4), 2*np.exp(1j*np.pi/2)]])

        for x in np.arange(len(test_trans_data)):
            for y in np.arange(len(test_trans_data[0])):
                self.assertEqual(trans_data[x][y], test_trans_data[x][y])

if __name__ == '__main__':
    unittest.main()

