from __future__ import print_function

from radiotools.atmosphere import refractivity
from radiotools import helper
import unittest
import numpy as np

axis = helper.spherical_to_cartesian(np.deg2rad(80), 0)
tab_flat = refractivity.RefractivityTable(27)
tab_curved = refractivity.RefractivityTable(27, curved=True, number_of_zenith_bins=100)

class RefractivityTests(unittest.TestCase):

    def test_compare_tabs_for_flat_table(self):
        heights = np.arange(0, 5e4, 1e3)
        for height in heights:
            self.assertTrue(np.allclose(tab_flat.get_integrated_refractivity_for_height_tabulated(height),
                                        tab_curved.get_integrated_refractivity_for_height_tabulated(height)))

    def test_curved_table(self):
        refractivities = []
        for dist in np.arange(1e3, 1e5, 1e4):
            point = axis * dist
            r = tab_curved.get_refractivity_between_two_points_tabulated(point, np.array([0, 0, 0]))
            refractivities.append(r)

        self.assertTrue(np.allclose(refractivities, np.array([0.00030889595670082067, 0.000282411093234063, 0.000258794023252093,
            0.00023771311586077572, 0.0002188753498864094, 0.0002020187326970188, 0.0001865660513275855, 0.0001719832852117586,
            0.00015860798388484651, 0.00014650532450181743])))

    def test_num_cal(self):
        refractivities = []
        for dist in np.arange(1e3, 1e5, 1e4):
            point = axis * dist
            r = tab_curved.get_refractivity_between_two_points_numerical(point, np.array([0, 0, 0]))
            refractivities.append(r)
        self.assertTrue(np.allclose(refractivities, np.array([0.00030917724719699887, 0.000282667441765032, 0.0002590260410419691,
            0.00023792193761338818, 0.00021906244225125651, 0.00020218896167550733, 0.0001867273146300562, 0.00017212383567359824,
            0.0001587301696693485, 0.00014661389359611995])))


if __name__ == "__main__":
    unittest.main()
