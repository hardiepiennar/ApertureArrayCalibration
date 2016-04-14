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
        nearfield_z = np.array([[0, 0], [0, 0.2]])

        pad_factor = 2
        grid_x, grid_y, nearfield_x, nearfield_y, nearfield_z = nf2ff.pad_nearfield_grid(grid_x, grid_y,
                                                                                         nearfield_x,
                                                                                         nearfield_y,
                                                                                         nearfield_z,
                                                                                         pad_factor)

        grid_x_test = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        grid_y_test = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        nearfield_x_test = np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        nearfield_y_test = np.array([[0, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        nearfield_z_test = np.array([[0, 0, 0, 0], [0, 0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        self.assertEqual(len(grid_x), len(grid_x_test))
        self.assertEqual(len(grid_x[0]), len(grid_x_test[0]))
        self.assertEqual(len(grid_y), len(grid_y_test))
        self.assertEqual(len(grid_y[0]), len(grid_y_test[0]))
        self.assertEqual(len(nearfield_x), len(nearfield_x_test))
        self.assertEqual(len(nearfield_x_test[0]), len(nearfield_x_test[0]))
        self.assertEqual(len(nearfield_y), len(nearfield_y_test))
        self.assertEqual(len(nearfield_y_test[0]), len(nearfield_y_test[0]))
        self.assertEqual(len(nearfield_z), len(nearfield_z_test))
        self.assertEqual(len(nearfield_z_test[0]), len(nearfield_z_test[0]))

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
        for y in np.arange(len(nearfield_z_test)):
            for x in np.arange(len(nearfield_z_test[0])):
                self.assertAlmostEqual(nearfield_z[y][x], nearfield_z_test[y][x])

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

    def test_calculate_angular_spectrum(self):
        nearfield = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        farfield = nf2ff.calc_angular_spectrum(nearfield)

        test_farfield = (float(1)/9)*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        for y in np.arange(len(test_farfield)):
            for x in np.arange(len(test_farfield[0])):
                self.assertAlmostEqual(np.abs(test_farfield[y][x]), np.abs(farfield[y][x]))

        # TODO: test phase

    def test_interpolate_cartesian_to_spherical(self):
        kx = np.array([-171.618901628412, -122.584929734580, -73.5509578407478,	-24.5169859469160, 24.5169859469160,
                       73.5509578407478, 122.584929734580, 171.618901628412])
        ky = kx
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        wavenumber = 2.095845021951682e+02
        theta = np.linspace(-np.pi/2, np.pi/2, 5)
        phi = np.linspace(0, np.pi, 5)
        fe_grid = np.array([[0.00000000000000 + 0.00000000000000j, 0.00000000000000 + 0.00000000000000j,
                             0.00000000000000 + 0.00000000000000j, 0.00000000000000 + 0.00000000000000j,
                             0.00000000000000 + 0.00000000000000j, 0.00000000000000 + 0.00000000000000j,
                             0.00000000000000 + 0.00000000000000j, 0.00000000000000 + 0.00000000000000j],
                            [0.00000000000000 + 0.00000000000000j, 0.0108331882103510 + 0.0218911208585877j,
                             0.0343562363589819 + 0.0456543198251686j, 0.241962291788929 + 0.409449000488838j,
                             -1.37236284739377 - 0.0224365397246469j, 0.460617342111102 + 0.118430987480321j,
                             0.0456543198251686 - 0.0343562363589819j, 0.00781913916147228 - 0.0231395808522910j],
                            [0.00000000000000 + 0.00000000000000j, -0.0137398239816984 + 0.00986141014910014j,
                             -0.0175607265624998 + 0.0415145678437502j,	-0.228017295229509 + 0.378197645497473j,
                             -0.645659918843750 - 1.20143131071875j, 0.106193544075448 + 0.428658695444650j,
                             0.0415145678437502 + 0.0175607265624998j, 0.0166885926982591 + 0.00274245272127793j],
                            [0.00000000000000 + 0.00000000000000j, 0.315794397413931 - 0.00305241744866214j,
                             0.583854287285398 + 0.0275241336255694j, 4.28239830770234 + 0.196123298008400j,
                             -6.15668734300495 + 9.35578093053623j,	3.16679299708853 - 2.88943276914771j,
                             0.0275241336255694 - 0.583854287285398j, -0.225458744949071 - 0.221141974795149j],
                            [0.00000000000000 + 0.00000000000000j, -0.522264100582747 + 0.709849364023251j,
                             -1.00170456240625 + 1.22101412946875j,	-7.24872334920165 + 9.26108517382945j,
                             -11.3405563681875 - 28.6603950781250j,	1.42295469239525 + 11.6741975627267j,
                             1.22101412946875 + 1.00170456240625j, 0.871235786014152 - 0.132642811829446j],
                            [0.00000000000000 + 0.00000000000000j, 0.221141974795148 - 0.225458744949071j,
                             0.432309827297269 - 0.393384824231419j, 3.16679299708853 - 2.88943276914771j,
                             2.26210076939377 + 10.9689715091621j, 0.196123298008400 - 4.28239830770234j,
                             -0.393384824231419 - 0.432309827297269j, -0.315794397413930 + 0.00305241744866236j],
                            [0.00000000000000 + 0.00000000000000j, 0.00986141014910014 + 0.0137398239816984j,
                             0.0415145678437502 + 0.0175607265624998j, 0.378197645497473 + 0.228017295229509j,
                             -1.20143131071875 + 0.645659918843750j, 0.428658695444650 - 0.106193544075448j,
                             0.0175607265624998 - 0.0415145678437502j, 0.00274245272127793 - 0.0166885926982591j],
                            [0.00000000000000 + 0.00000000000000j, 0.00781913916147232 - 0.0231395808522905j,
                             0.00798895143335229 - 0.0565760068443202j,	0.118430987480321 - 0.460617342111102j,
                             0.954542046254953 + 0.986272105026270j, -0.241962291788930 - 0.409449000488838j,
                             -0.0565760068443202 - 0.00798895143335229j, -0.0218911208585874 + 0.0108331882103507j]])
        
        fe_spherical = nf2ff.interpolate_cartesian_to_spherical(kx_grid,ky_grid,fe_grid,wavenumber,theta,phi)

        fe_spherical_test = np.array([[-1.19798150562089 + 19.3797511507843j,
                                      -0.227964875625187 - 2.07083872882152j,
                                      -7.98042564710115 + 4.81284813513545j,
                                      -3.47463478357622 + 4.12300261398233j,
                                      34.5652524272203 - 43.4613894696073j],
                                     [0.316884874378104 + 0.192356769257137j,
                                      -0.0115539438318393 + 0.148913211027039j,
                                      -7.98042564710115 + 4.81284813513545j,
                                      -0.173370941001693 + 3.70191446043908j
                                      -0.166156570604876 + 2.86011902866362j],
                                     [-1.94415044619004 + 15.1691583332975j,
                                      0.134466179122087 - 1.69826826556644j,
                                      -7.98042564710115 + 4.81284813513545j,
                                      -1.94169113866461 + 4.60548677858829j,
                                      24.7469933111242 - 43.1344041703480j],
                                     [-0.265038684720100 - 0.953366740538459j,
                                      -0.0153679320156506 - 0.440670149314655j,
                                      -7.98042564710115 + 4.81284813513545j,
                                      0.126694991242735 - 1.08728625015510j,
                                      -0.742216143509634 - 0.841813991415576j],
                                     [34.5652524272203 - 43.4613894696072j,
                                      -3.47463478357622 + 4.12300261398232j,
                                      -7.98042564710115 + 4.81284813513545j,
                                      -0.227964875625189 - 2.07083872882153j,
                                      -1.19798150562087 + 19.3797511507843j]])

        self.assertEqual(len(fe_spherical), len(fe_spherical_test))
        self.assertEqual(len(fe_spherical[0]), len(fe_spherical_test[0]))


    def end(self):
        pass


if __name__ == '__main__':
    unittest.main()

