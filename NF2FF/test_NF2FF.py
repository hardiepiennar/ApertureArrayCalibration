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
        x = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        y = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        v = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        coordinates = [x, y]
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


if __name__ == '__main__':
    unittest.main()

