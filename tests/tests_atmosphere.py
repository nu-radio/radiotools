from __future__ import print_function

from radiotools.atmosphere import models as atmos

import unittest
import numpy as np


zen = np.deg2rad(70)
xmax = 800.  # g/cm2


class AtmosphereTests(unittest.TestCase):

    # this is a rather weak test checking if the observation_level is used in the calculation at all
    def test_geometric_distance_to_xmax_for_different_obs_lvl(self):
        atm = atmos.Atmosphere()
        dxmax_0 = atm.get_distance_xmax_geometric(zen, xmax, observation_level=0.)
        dxmax_1564 = atm.get_distance_xmax_geometric(zen, xmax, observation_level=1564.)
        dxmax_3000 = atm.get_distance_xmax_geometric(zen, xmax, observation_level=3000.)
        self.assertFalse(np.allclose(dxmax_0, dxmax_1564, rtol=0.1, atol=100))
        self.assertFalse(np.allclose(dxmax_0, dxmax_3000, rtol=0.1, atol=100))
        self.assertFalse(np.allclose(dxmax_1564, dxmax_3000, rtol=0.1, atol=100))


    def test_distance_xmax_round_trip(self):
        atm = atmos.Atmosphere()
        zeniths = np.deg2rad(np.linspace(0, 85, 30))
        dX = np.array([750 - atm.get_xmax_from_distance(atm.get_distance_xmax_geometric(z, 750, 1400), z, 1400) for z in zeniths])
        self.assertFalse(np.allclose(dX, np.zeros_like(dX)))

    def test_height_above_ground_to_distance_transformation(self):
        zeniths = np.deg2rad(np.linspace(0, 90, 10))
        for zenith in zeniths:
            heights = np.linspace(0, 1e5, 20)
            for h in heights:
                obs_levels = np.linspace(0, 2e3, 4)
                for obs in obs_levels:
                    d = atmos.get_distance_for_height_above_ground(h, zenith, observation_level=obs)
                    h2 = atmos.get_height_above_ground(d, zenith, observation_level=obs)
                    self.assertAlmostEqual(h, h2)

    def test_flat_atmosphere(self):
        atm = atmos.Atmosphere(curved=False)
        zeniths = np.deg2rad(np.linspace(0, 89, 10))
        heights = np.linspace(0, 1e4, 10)
        atm1 = atm.get_atmosphere(zeniths, heights)
        atm2 = atm.get_atmosphere(np.zeros(10), heights) / np.cos(zeniths)
        for i in range(len(atm1)):
            self.assertAlmostEqual(atm1[i], atm2[i])

        heights2 = np.linspace(1e4, 1e5, 10)
        atm1 = atm.get_atmosphere(zeniths, heights, heights2)
        atm2 = atm.get_atmosphere(np.zeros(10), heights, heights2) / np.cos(zeniths)
        for i in range(len(atm1)):
            self.assertAlmostEqual(atm1[i], atm2[i])

        z = np.deg2rad(50)
        atm1 = atm.get_atmosphere(z, 0)
        atm2 = atm.get_atmosphere(0, 0) / np.cos(z)
        self.assertAlmostEqual(atm1, atm2, delta=1e-3)

        atm1 = atm.get_atmosphere(z, 10, 1e4)
        atm2 = atm.get_atmosphere(0, 10, 1e4) / np.cos(z)
        self.assertAlmostEqual(atm1, atm2, delta=1e-3)

    def test_numeric_atmosphere(self):
        atm_flat = atmos.Atmosphere(curved=False)
        atm_num = atmos.Atmosphere(curved=True, zenith_numeric=0)
        zeniths = np.deg2rad(np.linspace(0, 20, 3))
        atm1 = atm_flat.get_atmosphere(zeniths, 0)
        atm2 = atm_num.get_atmosphere(zeniths, 0)
        for i in range(len(atm1)):
            delta = 1e-3 + np.rad2deg(zeniths[i]) * 1e-2
            self.assertAlmostEqual(atm1[i], atm2[i], delta=delta)

        atm1 = atm_flat.get_atmosphere(zeniths, 1e3)
        atm2 = atm_num.get_atmosphere(zeniths, 1e3)
        for i in range(len(atm1)):
            delta = 1e-3 + np.rad2deg(zeniths[i]) * 1e-2
            self.assertAlmostEqual(atm1[i], atm2[i], delta=delta)

        atm1 = atm_flat.get_atmosphere(zeniths, 1e3, 1e4)
        atm2 = atm_num.get_atmosphere(zeniths, 1e3, 1e4)
        for i in range(len(atm1)):
            delta = 1e-3 + np.rad2deg(zeniths[i]) * 1e-2
            self.assertAlmostEqual(atm1[i], atm2[i], delta=delta)

        z = np.deg2rad(0)
        atm1 = atm_flat.get_atmosphere(z, 0)
        atm2 = atm_num.get_atmosphere(z, 0)
        self.assertAlmostEqual(atm1, atm2, delta=1e-3)

        atm1 = atm_flat.get_atmosphere(z, 10, 1e4)
        atm2 = atm_num.get_atmosphere(z, 10, 1e4)
        self.assertAlmostEqual(atm1, atm2, delta=1e-2)

    def test_taylor_atmosphere(self):
        atm_taylor = atmos.Atmosphere(curved=True)
        atm_num = atmos.Atmosphere(curved=True, zenith_numeric=0)

        for h in np.linspace(0, 1e4, 10):
            atm1 = atm_taylor.get_atmosphere(0, h_low=h)
            atm2 = atm_num.get_atmosphere(0, h_low=h)
            self.assertAlmostEqual(atm1, atm2, delta=1e-3)

        zeniths = np.deg2rad([0, 11.478341, 30.683417])
        for i in range(len(zeniths)):
            delta = 1e-6
            atm1 = atm_taylor.get_atmosphere(zeniths[i], 0)
            atm2 = atm_num.get_atmosphere(zeniths[i], 0)
            self.assertAlmostEqual(atm1, atm2, delta=delta)

        atm1 = atm_taylor.get_atmosphere(zeniths, 1e3)
        atm2 = atm_num.get_atmosphere(zeniths, 1e3)
        for i in range(len(atm1)):
            delta = 1e-5
            self.assertAlmostEqual(atm1[i], atm2[i], delta=delta)

        atm1 = atm_taylor.get_atmosphere(zeniths, 1e3, 1e4)
        atm2 = atm_num.get_atmosphere(zeniths, 1e3, 1e4)
        for i in range(len(atm1)):
            delta = 1e-5
            self.assertAlmostEqual(atm1[i], atm2[i], delta=delta)

        z = np.deg2rad(0)
        atm1 = atm_taylor.get_atmosphere(z, 0)
        atm2 = atm_num.get_atmosphere(z, 0)
        self.assertAlmostEqual(atm1, atm2, delta=1e-3)

        atm1 = atm_taylor.get_atmosphere(z, 10, 1e4)
        atm2 = atm_num.get_atmosphere(z, 10, 1e4)
        self.assertAlmostEqual(atm1, atm2, delta=1e-2)

    def test_taylor_atmosphere2(self):
        atm_taylor = atmos.Atmosphere(curved=True)
        atm_num = atmos.Atmosphere(curved=True, zenith_numeric=0)

        zeniths = np.deg2rad(np.linspace(0, 83, 20))
        for i in range(len(zeniths)):
            delta = 1e-3
            # print "checking z = %.1f" % np.rad2deg(zeniths[i])
            atm1 = atm_taylor.get_atmosphere(zeniths[i], 0)
            atm2 = atm_num.get_atmosphere(zeniths[i], 0)
            delta = max(delta, 1.e-5 * atm1)
            self.assertAlmostEqual(atm1, atm2, delta=delta)

        zeniths = np.deg2rad(np.linspace(0, 83, 20))
        for i in range(len(zeniths)):
            delta = 1e-2
            # print "checking z = %.1f" % np.rad2deg(zeniths[i])
            atm1 = atm_taylor.get_atmosphere(zeniths[i], 1e3)
            atm2 = atm_num.get_atmosphere(zeniths[i], 1e3)
            self.assertAlmostEqual(atm1, atm2, delta=delta)

        zeniths = np.deg2rad(np.linspace(0, 83, 20))
        for i in range(len(zeniths)):
            delta = 1e-2
            # print "checking z = %.1f" % np.rad2deg(zeniths[i])
            atm1 = atm_taylor.get_atmosphere(zeniths[i], 0, 1e4)
            atm2 = atm_num.get_atmosphere(zeniths[i], 0, 1e4)
            self.assertAlmostEqual(atm1, atm2, delta=delta)

    def test_vertical_height_flat_numeric(self):
        atm_flat = atmos.Atmosphere(curved=False)
        atm_num = atmos.Atmosphere(curved=True, zenith_numeric=0)
        zenith = 0
        xmax = np.linspace(300, 900, 20)
        atm1 = atm_flat.get_vertical_height(zenith * np.ones_like(xmax), xmax)
        atm2 = atm_num.get_vertical_height(zenith * np.ones_like(xmax), xmax)
        for i in range(len(xmax)):
            self.assertAlmostEqual(atm1[i], atm2[i], delta=1e-2)

        zeniths = np.deg2rad(np.linspace(0, 30, 4))
        xmax = 600
        atm1 = atm_flat.get_vertical_height(zeniths, xmax)
        atm2 = atm_num.get_vertical_height(zeniths, xmax)
        for i in range(len(zeniths)):
            self.assertAlmostEqual(atm1[i], atm2[i], delta=1e-3 * atm1[i])

    def test_vertical_height_taylor_numeric(self):
        atm_taylor = atmos.Atmosphere(curved=True)
        atm_num = atmos.Atmosphere(curved=True, zenith_numeric=0)

        zeniths = np.deg2rad(np.linspace(0, np.rad2deg(atm_taylor._Atmosphere__zenith_numeric) - 1e-3, 30))
        xmax = 600

        atm1 = atm_taylor.get_vertical_height(zeniths, xmax)
        atm2 = atm_num.get_vertical_height(zeniths, xmax)

        for i in range(len(zeniths)):
            self.assertAlmostEqual(atm1[i], atm2[i], delta=2e-5 * atm1[i])

#
#     def test_atmosphere_above_height_for_flat_atm(self):
#         curved = False
#         zeniths = np.deg2rad(np.linspace(0, 70, 8))
#         for zenith in zeniths:
#             catm = atmos.Atmosphere(zenith, curved=curved)
#             heights = np.linspace(0, 1e5, 20)
#             for h in heights:
#                 atm = get_atmosphere(h) / np.cos(zenith)
#                 atm2 = catm.get_atmosphere2(h_low=h)
#                 self.assertAlmostEqual(atm, atm2)
#
#     def test_density_for_flat_atm(self):
#         curved = False
#         zeniths = np.deg2rad(np.linspace(0, 70, 8))
#         for zenith in zeniths:
#             catm = atmos.Atmosphere(zenith, curved=curved)
#             heights = np.linspace(0, 1e5, 20)
#             for h in heights:
#                 rho = get_density(h)
#                 xmax = catm.get_atmosphere2(h_low=h)
#                 rho2 = catm.get_density2(xmax)
#                 self.assertAlmostEqual(rho, rho2)
#
#     def test_numerical_density_integration(self):
#
#         def allowed_discrepancy(zenith):
#             z = np.rad2deg(zenith)
#             return z ** 2 / 2500 + z / 90. + 1e-2
#
#         zeniths = np.deg2rad(np.linspace(0, 40, 5))
#         for zenith in zeniths:
#             catm = atmos.Atmosphere(zenith)
#             heights = np.linspace(0, 1e4, 2)
#             for h in heights:
#                 atm1 = get_atmosphere(h) / np.cos(zenith)
#                 atm2 = catm.get_atmosphere2(h_low=h)
#                 self.assertAlmostEqual(atm1, atm2, delta=allowed_discrepancy(zenith))
#
#     def test_get_distance_to_xmax_flat_vs_curved(self):
#
#         def allowed_discrepancy(zenith):
#             z = np.rad2deg(zenith)
#             return z ** 2 / 2500 + z / 90. + 1e-2
#
#         zeniths = np.deg2rad(np.linspace(0, 40, 5))
#         for zenith in zeniths:
#             catm = atmos.Atmosphere(zenith)
#             catm_flat = atmos.Atmosphere(zenith, curved=False)
#             xmaxs = np.linspace(0, 1e3, 4)
#             for xmax in xmaxs:
#                 dxmax1 = catm_flat.get_distance_xmax(xmax, observation_level=0)
#                 dxmax2 = catm.get_distance_xmax(xmax, observation_level=0)
#                 # print "zenith %.0f xmax = %.2g, %.5g, %.5g" % (np.rad2deg(zenith), xmax, dxmax1, dxmax2)
#                 self.assertAlmostEqual(dxmax1, dxmax2, delta=allowed_discrepancy(zenith))
#
#     def test_get_distance_to_xmax_geometric_flat_self_consitency(self):
# #         print
# #         print
# #         print "test_get_distance_to_xmax_geometric_flat_self_consitency"
#         zeniths = np.deg2rad(np.linspace(0, 80, 9))
#         dxmaxs = np.linspace(0, 4e3, 5)
#         obs_levels = np.linspace(0, 2e3, 4)
#         for dxmax1 in dxmaxs:
#             for zenith in zeniths:
#                 catm = atmos.Atmosphere(zenith, curved=False)
#                 for obs in obs_levels:
#                     # print "\tdxmax1 = %.4f, z=%.1f observation level = %.2f" % (dxmax1, np.rad2deg(zenith), obs)
#                     h1 = dxmax1 * np.cos(zenith) + obs
#                     xmax = get_atmosphere(h1) / np.cos(zenith)
#                     dxmax2 = catm.get_distance_xmax_geometric(xmax, observation_level=obs)
#                     self.assertAlmostEqual(dxmax1, dxmax2, delta=1e-5)
#
#     def test_get_distance_to_xmax_geometric_curved_self_consitency(self):
# #         print
# #         print
# #         print "test_get_distance_to_xmax_geometric_curved_self_consitency"
#         zeniths = np.deg2rad(np.linspace(0, 89, 10))
#         dxmaxs = np.linspace(0, 4e3, 5)
#         obs_levels = np.linspace(0, 2e3, 5)
#         for dxmax1 in dxmaxs:
#             # print "checking dxmax = %.2f" % dxmax1
#             for zenith in zeniths:
#                 # print "checking zenith angle of %.1f" % (np.rad2deg(zenith))
#                 catm = atmos.Atmosphere(zenith)
#                 delta = 1e-4
#                 if zenith > np.deg2rad(85):
#                     delta = 1.e-2
#                 for obs in obs_levels:
#                     # print "\tdxmax1 = %.4f, z=%.1f observation level = %.2f" % (dxmax1, np.rad2deg(zenith), obs)
#                     # print "testing"
#                     h1 = get_height_above_ground(dxmax1, zenith, observation_level=obs) + obs
#                     xmax = catm.get_atmosphere2(h_low=h1)
#                     # print "zenith %.0f dmax = %.2g, obslevel = %.3g -> h1 = %.3g, xmax = %.3g" % (np.rad2deg(zenith), dxmax1, obs, h1, xmax)
#                     dxmax2 = catm.get_distance_xmax_geometric(xmax, observation_level=obs)
#                     self.assertAlmostEqual(dxmax1, dxmax2, delta=delta)
#
#     def test_get_distance_to_xmax_geometric_flat_vs_curved(self):
# #         print
# #         print
# #         print "test_get_distance_to_xmax_geometric_flat_vs_curved"
#
#         def allowed_discrepancy(zenith):
#             z = np.rad2deg(zenith)
#             return z ** 2 / 20.**2 * 1e-3 + 1e-3
#
#         zeniths = np.deg2rad(np.linspace(0, 60, 7))
#         xmaxs = np.linspace(100, 900, 4)
#         obs_levels = np.linspace(0, 1.5e3, 4)
#         for zenith in zeniths:
#             catm = atmos.Atmosphere(zenith)
#             catm_flat = atmos.Atmosphere(zenith, curved=False)
#             for xmax in xmaxs:
#                 for obs in obs_levels:
#                     dxmax1 = catm_flat.get_distance_xmax_geometric(xmax, observation_level=obs)
#                     dxmax2 = catm.get_distance_xmax_geometric(xmax, observation_level=obs)
#                     # print "zenith %.0f xmax = %.2g, obslevel = %.3g, %.5g, %.5g %.2g" % (np.rad2deg(zenith), xmax, obs, dxmax1, dxmax2, 3 + np.abs(dxmax1 * allowed_discrepancy(zenith)))
#                     self.assertAlmostEqual(dxmax1, dxmax2, delta=3. + np.abs(dxmax1 * allowed_discrepancy(zenith)))

#     def test_get_distance_to_xmax_geometric_flat_vs_curved2(self):
#
#         def allowed_discrepancy(zenith):
#             z = np.rad2deg(zenith)
#             return z ** 2 / 20.**2 * 1e-3 + 1e-3
#
#         zeniths = np.deg2rad(np.linspace(0, 60, 7))
#         xmaxs = np.linspace(0, 900, 4)
#         obs_levels = np.linspace(0, 1.5e3, 4)
#         for zenith in zeniths:
#             for xmax in xmaxs:
#                 for obs in obs_levels:
#                     print
#                     print "testing "
#                     dxmax1 = get_distance_xmax_geometric2(xmax, zenith, observation_level=obs, curved=False)
#                     if dxmax1 < 0:
#                         print "\t skipping negetive distances"
#                         continue
#                     print "zenith %.0f xmax = %.2g, obslevel = %.3g, %.5g" % (np.rad2deg(zenith), xmax, obs, dxmax1)
#                     dxmax2 = get_distance_xmax_geometric2(xmax, zenith, observation_level=obs, curved=True)
#                     print "zenith %.0f xmax = %.2g, obslevel = %.3g, %.5g, %.5g %.2g" % (np.rad2deg(zenith), xmax, obs, dxmax1, dxmax2, 1e-1 + np.abs(dxmax1 * allowed_discrepancy(zenith)))
#                     self.assertAlmostEqual(dxmax1, dxmax2, delta=3. + np.abs(dxmax1 * allowed_discrepancy(zenith)))


if __name__ == "__main__":
    try:
        # initilize Atmosphere (tables)
        atmos.Atmosphere()
    except SystemExit:
        print('Initilized Atmosphere')

    unittest.main()
