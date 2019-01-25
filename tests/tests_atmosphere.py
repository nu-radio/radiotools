from radiotools.atmosphere import models as atm

import unittest
import numpy as np

# initilize atmosphere
at = atm.Atmosphere(model=27)

zen = np.deg2rad(70)
xmax = 800.  # g/cm2


class AtmosphereTests(unittest.TestCase):

    # this is a rather weak test checking if the observation_level is used in the calculation at all 
    def test_geometric_distance_to_xmax_for_different_obs_lvl(self):
        dxmax_0 = at.get_distance_xmax_geometric(zen, xmax, observation_level=0.)
        dxmax_1564 = at.get_distance_xmax_geometric(zen, xmax, observation_level=1564.)
        dxmax_3000 = at.get_distance_xmax_geometric(zen, xmax, observation_level=3000.)
        self.assertFalse(np.allclose(dxmax_0, dxmax_1564, rtol=0.1, atol=100))
        self.assertFalse(np.allclose(dxmax_0, dxmax_3000, rtol=0.1, atol=100))
        self.assertFalse(np.allclose(dxmax_1564, dxmax_3000, rtol=0.1, atol=100))


if __name__ == "__main__":
    unittest.main()
