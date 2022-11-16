from radiotools import coordinatesystems
from radiotools import helper

import unittest
import copy
import numpy as np


# initilize coordinatesystem
# shower geometry matches stored traces, azimuth transformation from corsika to auger coordinates
zenith = np.deg2rad(65.4267987)
azimuth = helper.get_normalized_angle(3 * np.pi / 2. + np.deg2rad(-94.31200338))
cs = coordinatesystems.cstrafo(zenith, azimuth)
shower_axis = helper.spherical_to_cartesian(zenith, azimuth)

# test vectors
v_vB_vvB = np.array([0, 0, 1])
v_xyz = -1 * np.array([np.sin(zenith) * np.cos(azimuth),
                       np.sin(zenith) * np.sin(azimuth),
                       np.cos(zenith)])

v_vB_vvB2 = np.array([v_vB_vvB, 2 * v_vB_vvB])

# test traces
test_data = np.load("tests/test_data/efield_traces.npz", "r")
t_xyz = test_data["t_xyz"]
t_vB_vvB = test_data["t_vBvvB"]


class CoordinateTests(unittest.TestCase):

    def test_constant_input(self):
        v_xyz_test = copy.deepcopy(v_xyz)
        cs.transform_to_vxB_vxvxB(v_xyz, core=np.array([1, 1, 1]))
        self.assertTrue(np.allclose(v_xyz, v_xyz_test))

    def test_transform_station_to_xyz(self):
        v_xyz_test = cs.transform_from_vxB_vxvxB(v_vB_vvB)
        self.assertTrue(np.allclose(v_xyz, v_xyz_test))

    def test_transform_station_to_vB_vvB(self):
        v_vB_vvB_test = cs.transform_to_vxB_vxvxB(v_xyz)
        self.assertTrue(np.allclose(v_vB_vvB, v_vB_vvB_test))

    def test_transform_station(self):
        v_vB_vvB_test = cs.transform_from_vxB_vxvxB(cs.transform_to_vxB_vxvxB(v_vB_vvB))
        self.assertTrue(np.allclose(v_vB_vvB, v_vB_vvB_test))

    def test_transform_station_list(self):
        v_vB_vvB2_test = cs.transform_from_vxB_vxvxB(cs.transform_to_vxB_vxvxB(v_vB_vvB2))
        self.assertTrue(np.allclose(v_vB_vvB2, v_vB_vvB2_test))

    def test_transform_traces_to_xyz(self):
        t_xyz_test = cs.transform_from_vxB_vxvxB(t_vB_vvB)
        self.assertTrue(np.allclose(t_xyz, t_xyz_test, rtol=1.e-5, atol=1.e-5))

    def test_transform_traces_to_vB_vvB(self):
        t_vB_vvB_test = cs.transform_to_vxB_vxvxB(t_xyz)
        self.assertTrue(np.allclose(t_vB_vvB, t_vB_vvB_test, rtol=1.e-5, atol=1.e-5))

    def test_transform_to_early_late(self):
        shower_axis_tramsform = cs.transform_to_early_late(shower_axis)
        self.assertTrue(np.allclose(
            np.array([0, 0, 1]), shower_axis_tramsform, rtol=1.e-5, atol=1.e-5))

    def test_transform_traces(self):
        t_vB_vvB_test = cs.transform_from_vxB_vxvxB(cs.transform_to_vxB_vxvxB(t_vB_vvB))
        self.assertTrue(np.allclose(t_vB_vvB, t_vB_vvB_test))


if __name__ == "__main__":
    unittest.main()
