from radiotools import coordinatesystems

import unittest
import numpy as np


zenith = np.deg2rad(60.)
azimuth = np.deg2rad(225.)
cs = coordinatesystems.cstrafo(zenith, azimuth)
v_vB_vvB = np.array([0, 0, 1])
v_xyz = np.array([-1 * np.sin(zenith) * np.cos(azimuth),
                  -1 * np.sin(zenith) * np.sin(azimuth),
                  -1 * np.cos(zenith)])

v_vB_vvB2 = np.array([v_vB_vvB, 2 * v_vB_vvB])


class CoordinateTests(unittest.TestCase):

    def test_transform_to_xyz(self):
        v_xyz_test = cs.transform_from_vxB_vxvxB(v_vB_vvB)
        self.assertTrue(np.allclose(v_xyz, v_xyz_test))

    def test_transform_to_vB_vvB(self):
        v_vB_vvB_test = cs.transform_to_vxB_vxvxB(v_xyz)
        self.assertTrue(np.allclose(v_vB_vvB, v_vB_vvB_test))

    def test_transform_single_postion(self):
        v_vB_vvB_test = cs.transform_from_vxB_vxvxB(cs.transform_to_vxB_vxvxB(v_vB_vvB))
        self.assertTrue(np.allclose(v_vB_vvB, v_vB_vvB_test))

    def test_transform_mutiple_postions(self):
        v_vB_vvB2_test = cs.transform_from_vxB_vxvxB(cs.transform_to_vxB_vxvxB(v_vB_vvB2))
        self.assertTrue(np.allclose(v_vB_vvB2, v_vB_vvB2_test))


if __name__ == "__main__":
    unittest.main()
