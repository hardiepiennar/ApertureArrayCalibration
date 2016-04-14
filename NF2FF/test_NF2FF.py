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
        x = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        v = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        coordinates = np.array([x, y])
        values = v
        res_x = 3
        res_y = 4
        grid_x, grid_y, grid_data = nf2ff.transform_data_coord_to_grid(coordinates, values, [res_x, res_y])

        grid_data_test = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        self.assertEqual(grid_data_test.__class__, grid_data.__class__)
        self.assertEqual(len(grid_data), len(grid_data_test))
        self.assertEqual(len(grid_data[0]), len(grid_data_test[0]))
        self.assertEqual(np.sum(grid_data-grid_data_test), 0)

        grid_x_test = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
        self.assertEqual(grid_x_test.__class__, grid_x.__class__)
        self.assertEqual(len(grid_x), len(grid_x_test))
        self.assertEqual(np.sum(grid_x-grid_x_test), 0)

        grid_y_test = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
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

    def test_2d_discrete_fourier_transform(self):
        data = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        scale = 1
        x = scale*np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        y = scale*np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        z = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        trans_data = nf2ff.calc_dft2(x, y, z, data)

        """Test if the transform has been calculated correctly"""
        test_trans_data = np.array([[np.exp(-2j*np.pi*-1*1/3 - 2j*np.pi*-1*1/3),
                                    np.exp(-2j*np.pi*0*1/3 - 2j*np.pi*-1*1/3),
                                    np.exp(-2j*np.pi*1*1/3 - 2j*np.pi*-1*1/3)],
                                   [np.exp(-2j*np.pi*-1*1/3 - 2j*np.pi*0*1/3),
                                    np.exp(-2j*np.pi*0*1/3 - 2j*np.pi*0*1/3),
                                    np.exp(-2j*np.pi*1*1/3 - 2j*np.pi*0*1/3)],
                                   [np.exp(-2j*np.pi*-1*1/3 - 2j*np.pi*1*1/3),
                                    np.exp(-2j*np.pi*0*1/3 - 2j*np.pi*1*1/3),
                                    np.exp(-2j*np.pi*1*1/3 - 2j*np.pi*1*1/3)]])

        for x in np.arange(len(test_trans_data)):
            for y in np.arange(len(test_trans_data[0])):
                self.assertEqual(trans_data[x][y], test_trans_data[x][y])

    def test_nearfield_to_farfield_xyz(self):
        nearfield_x = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
        nearfield_y = np.array([[0, 0, 0], [0, 0.1, 0], [0, 0, 0]])
        scale = (float(1)/10)

        N = 3
        d = 2

        farfield_test_x = (float(1)/N**2)*2*np.array([[np.exp(-2j*np.pi*-1*d/N - 2j*np.pi*-1*d/N),
                                                        np.exp(-2j*np.pi*0*d/N - 2j*np.pi*-1*d/N),
                                                        np.exp(-2j*np.pi*1*d/N - 2j*np.pi*-1*d/N)],
                                                        [np.exp(-2j*np.pi*-1*d/N - 2j*np.pi*0*d/N),
                                                        np.exp(-2j*np.pi*0*d/N - 2j*np.pi*0*d/N),
                                                        np.exp(-2j*np.pi*1*d/N - 2j*np.pi*0*d/N)],
                                                        [np.exp(-2j*np.pi*-1*d/N - 2j*np.pi*1*d/N),
                                                        np.exp(-2j*np.pi*0*d/N - 2j*np.pi*1*d/N),
                                                        np.exp(-2j*np.pi*1*d/N - 2j*np.pi*1*d/N)]])


        farfield_test_y = 0.1*farfield_test_x/2

        trans_farfield_x, trans_farfield_y = nf2ff.calc_nf2ff(nearfield_x, nearfield_y)

        """Test if the transforms were done correctly for the x and y farfields"""
        for y in np.arange(len(farfield_test_x)):
            for x in np.arange(len(farfield_test_x[0])):
                self.assertAlmostEqual(trans_farfield_x[y][x], farfield_test_x[y][x])
        for y in np.arange(len(trans_farfield_y)):
            for x in np.arange(len(trans_farfield_y[0])):
                self.assertAlmostEqual(trans_farfield_y[y][x], farfield_test_y[y][x])

    def test_generate_kspace(self):
        x_grid = 0.1*np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        y_grid = 0.1*np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        wavenumber = 30

        kx_grid, ky_grid, kz_grid = nf2ff.generate_kspace(x_grid, y_grid, wavenumber)

        scaling = 2*np.pi/(3*0.1)
        kx_grid_test = scaling*np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        ky_grid_test = scaling*np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        kz_grid_test = np.sqrt(wavenumber**2 - kx_grid_test**2 - ky_grid_test**2)

        self.assertEqual(len(kx_grid), len(kx_grid_test))
        self.assertEqual(len(kx_grid[0]), len(kx_grid_test[0]))
        self.assertEqual(len(ky_grid), len(ky_grid_test))
        self.assertEqual(len(ky_grid[0]), len(ky_grid_test[0]))
        self.assertEqual(len(kz_grid), len(kz_grid_test))
        self.assertEqual(len(kz_grid[0]), len(kz_grid_test[0]))

        for y in np.arange(len(kx_grid_test)):
            for x in np.arange(len(kx_grid_test[0])):
                self.assertAlmostEqual(kx_grid_test[y][x], kx_grid[y][x])
        for y in np.arange(len(ky_grid_test)):
            for x in np.arange(len(ky_grid_test[0])):
                self.assertAlmostEqual(ky_grid_test[y][x], ky_grid[y][x])
        for y in np.arange(len(kz_grid_test)):
            for x in np.arange(len(kz_grid_test[0])):
                self.assertAlmostEqual(kz_grid_test[y][x], kz_grid[y][x])

    def test_farfield_cartesian_to_spherical_conversion(self):
        x_grid = np.array([[-2, -1, 0, 1, 2],
                           [-2, -1, 0, 1, 2],
                           [-2, -1, 0, 1, 2],
                           [-2, -1, 0, 1, 2],
                           [-2, -1, 0, 1, 2]])
        y_grid = np.array([[-2, -2, -2, -2, -2],
                           [-1, -1, -1, -1, -1],
                           [0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1],
                           [2, 2, 2, 2, 2]])
        z_grid = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1]])

        farfield_x = 2*np.array([[1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1]])
        farfield_y = 0.1*farfield_x/2
        farfield_z = (farfield_x*x_grid + farfield_y*y_grid)/z_grid

        theta, phi, farfield_theta, farfield_phi = nf2ff.transform_cartesian_to_spherical(x_grid, y_grid, z_grid,
                                                                              farfield_x, farfield_y, farfield_z)

        """Check that theta and phi coords are correct"""
        self.assertAlmostEqual(theta[0][0], np.arctan2(np.sqrt(8), 1))
        self.assertAlmostEqual(theta[0][-1], np.arctan2(np.sqrt(8), 1))
        self.assertAlmostEqual(theta[1][2], np.pi/4)

        self.assertAlmostEqual(phi[0][0], -np.pi/4-np.pi/2)
        self.assertAlmostEqual(phi[0][-1], -np.pi/4)
        self.assertAlmostEqual(phi[-1][-1], np.pi/4)

        """Check that farfield theta and phi values are correct"""
        self.assertAlmostEqual(farfield_theta[2][2], 2)
        self.assertAlmostEqual(farfield_phi[2][2], 0.1)
        self.assertAlmostEqual(farfield_theta[2][1], -2*np.cos(np.pi/4))
        self.assertAlmostEqual(farfield_phi[2][1], -0.1)

    def test_get_fundamental_constants(self):
        c0, e0, u0 = nf2ff.get_fundamental_constants()
        self.assertEqual(c0, 299792458)
        self.assertEqual(e0, 8.8541878176e-12)
        self.assertEqual(u0, 4*np.pi*1e-7)

    def test_calc_freespace_wavelength(self):
        frequency = 1e9
        wavelength = nf2ff.calc_freespace_wavelength(frequency)
        self.assertEqual(wavelength, 299792458/frequency)

    def test_calc_freespace_wavenumber(self):
        frequency = 1e9
        wavenumber = nf2ff.calc_freespace_wavenumber(frequency)
        self.assertEqual(wavenumber, 2*np.pi/(299792458/frequency))

    def test_pad_nearfield_grid(self):
        grid_x = np.array([[0, 1], [0, 1]])
        grid_y = np.array([[0, 0], [1, 1]])
        nearfield_x = np.array([[0, 0], [0, 2]])
        nearfield_y = np.array([[0, 0], [0, 0.1]])

        pad_factor = 2
        grid_x, grid_y, nearfield_x, nearfield_y = nf2ff.pad_nearfield_grid(grid_x, grid_y,
                                                                            nearfield_x, nearfield_y,
                                                                            pad_factor)

        grid_x_test = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        grid_y_test = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        nearfield_x_test = np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        nearfield_y_test = np.array([[0, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        self.assertEqual(len(grid_x), len(grid_x_test))
        self.assertEqual(len(grid_x[0]), len(grid_x_test[0]))
        self.assertEqual(len(grid_y), len(grid_y_test))
        self.assertEqual(len(grid_y[0]), len(grid_y_test[0]))
        self.assertEqual(len(nearfield_x), len(nearfield_x_test))
        self.assertEqual(len(nearfield_x_test[0]), len(nearfield_x_test[0]))
        self.assertEqual(len(nearfield_y), len(nearfield_y_test))
        self.assertEqual(len(nearfield_y_test[0]), len(nearfield_y_test[0]))

        for y in np.arange(len(grid_x_test)):
            for x in np.arange(len(grid_x_test[0])):
                self.assertAlmostEqual(grid_x[y][x], grid_x_test[y][x])
        for y in np.arange(len(grid_y_test)):
            for x in np.arange(len(grid_y_test[0])):
                self.assertAlmostEqual(grid_y[y][x], grid_y_test[y][x])

        for y in np.arange(len(nearfield_x_test)):
            for x in np.arange(len(nearfield_x_test[0])):
                self.assertAlmostEqual(nearfield_x[y][x], nearfield_x_test[y][x])
        for y in np.arange(len(nearfield_y_test)):
            for x in np.arange(len(nearfield_y_test[0])):
                self.assertAlmostEqual(nearfield_y[y][x], nearfield_y_test[y][x])

    def test_generate_spherical_theta_phi_grid(self):
        theta_steps = 3
        phi_steps = 5
        theta_lim = (-np.pi/2, np.pi/2)
        phi_lim = (0, 2*np.pi)
        theta_grid, phi_grid = nf2ff.generate_spherical_theta_phi_grid(theta_steps, phi_steps, theta_lim, phi_lim)

        test_theta_grid = np.array([[-np.pi/2, 0, np.pi/2],
                                    [-np.pi/2, 0, np.pi/2],
                                    [-np.pi/2, 0, np.pi/2],
                                    [-np.pi/2, 0, np.pi/2],
                                    [-np.pi/2, 0, np.pi/2]])

        test_phi_grid = np.array([[0, 0, 0],
                                  [np.pi/2, np.pi/2, np.pi/2],
                                  [np.pi, np.pi, np.pi],
                                  [3*np.pi/2, 3*np.pi/2, 3*np.pi/2],
                                  [2*np.pi, 2*np.pi, 2*np.pi]])

        self.assertEqual(len(test_theta_grid), len(theta_grid))
        self.assertEqual(len(test_theta_grid[0]), len(theta_grid[0]))
        self.assertEqual(len(test_phi_grid), len(phi_grid))
        self.assertEqual(len(test_phi_grid[0]), len(phi_grid[0]))

        for y in np.arange(len(test_theta_grid)):
            for x in np.arange(len(test_theta_grid[0])):
                self.assertAlmostEqual(test_theta_grid[y][x], theta_grid[y][x])
        for y in np.arange(len(test_phi_grid)):
            for x in np.arange(len(test_phi_grid[0])):
                self.assertAlmostEqual(test_phi_grid[y][x], phi_grid[y][x])

    def end(self):
        pass


if __name__ == '__main__':
    unittest.main()

