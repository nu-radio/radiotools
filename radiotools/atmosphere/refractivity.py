from radiotools.atmosphere import models as atm
from radiotools import helper

import numpy as np
import sys
import copy
import os

def n_param_ZHAireS(h):
    a = 325e-6
    b = 0.1218e-3
    return 1 + a * np.exp(-b * h)


class RefractivityTable(object):

    def __init__(self, atm_model=atm.default_model, param=False, refractivity_at_sea_level=312e-6, curved=False,
                 interpolate_zenith=True, number_of_zenith_bins=1000, distance_increment=100):

        self._distance_increment = distance_increment
        self._number_of_zenith_bins = number_of_zenith_bins
        self._interpolate_zenith = interpolate_zenith
        self._refractivity_at_sea_level = refractivity_at_sea_level
        self._use_param = param
        self._atm_model = atm_model
        self._rho0 = atm.get_density(0, allow_negative_heights=False, model=self._atm_model)

        self._height_increment = 10
        self._heights = np.arange(0, 2e5, self._height_increment)
        self._refractivity_table = np.zeros(len(self._heights))
        for hdx, height in enumerate(self._heights):
            if self._use_param:
                refractivity = n_param_ZHAireS(height) - 1
            else:
                refractivity = self._refractivity_at_sea_level * \
                    atm.get_density(height, allow_negative_heights=True, model=self._atm_model) / self._rho0

            self._refractivity_table[hdx] = refractivity

        self._curved = curved
        if self._curved:
            self.get_cached_table_curved_atmosphere()

        else:
            self._refractivity_integrated_table = np.zeros(len(self._heights))

            self._refractivity_integrated_table[0] = self._refractivity_table[0] * self._height_increment
            self._refractivity_integrated_table[1:] = self._refractivity_integrated_table[:-1] \
                                                    + self._refractivity_table[1:] * self._height_increment

    def get_cached_table_curved_atmosphere(self):
        basedir = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.join(basedir, "refractivity_%02d_%.0f_%d.npz" % (self._atm_model,
                self._refractivity_at_sea_level * 1e6, self._number_of_zenith_bins))

        if os.path.exists(fname):
            print("Read in {} ...".format(fname))
            data = np.load(fname, "r")
            self._refractivity_integrated_table = data["refractivity_integrated_table"]
            self._distance_increment = data["distance_increment"]
            self._distances = data["distances"]
            self._zeniths = data["zeniths"]
        else:
            print("Write {} ...".format(fname))

            self._distances = np.arange(-2e4, 3e5, self._distance_increment)
            self._zeniths = np.hstack([np.arccos(1 / np.linspace(1.5557238268604123, 57.2986884985499, self._number_of_zenith_bins))])  # from 50 to 89 deg
            self._refractivity_integrated_table = np.zeros((len(self._zeniths), len(self._distances)))

            for zdx, zen in enumerate(self._zeniths):
                for ddx, dist in enumerate(self._distances):
                    height = atm.get_height_above_ground(dist, zen, observation_level=0)
                    if height > np.amax(self._heights):
                        sys.exit("Max height reached")
                    refractivity = self.get_refractivity_for_height_tabulated(height)

                    if ddx == 0:
                        self._refractivity_integrated_table[zdx, 0] = refractivity * self._distance_increment
                    else:
                        self._refractivity_integrated_table[zdx, ddx] = \
                                self._refractivity_integrated_table[zdx, ddx-1] \
                                + refractivity * self._distance_increment

            data = {"refractivity_integrated_table": self._refractivity_integrated_table,
                    "distance_increment": self._distance_increment,
                    "distances": self._distances,
                    "zeniths": self._zeniths}

            np.savez(fname, **data)


    def get_refractivity_for_height_tabulated(self, h):
        if h == 0:
            return self._refractivity_at_sea_level

        fidx = (h - self._heights[0]) / self._height_increment
        idx = int(fidx)
        f = fidx - idx

        return (1-f) * self._refractivity_table[idx] + f * self._refractivity_table[idx+1]


    def get_integrated_refractivity_for_height_tabulated(self, h):
        # if self._curved:
        #     sys.exit("Config is wrong: curved")

        if h == 0:
            return self._refractivity_at_sea_level

        fidx = (h - self._heights[0]) / self._height_increment
        idx = int(fidx)
        f = fidx - idx

        return (1-f) * self._refractivity_integrated_table[idx] + f * self._refractivity_integrated_table[idx+1]


    def _get_integrated_refractivity_for_distance(self, d, zenith):
        if not self._curved:
            sys.exit("Config is wrong: not curved")

        # next zenith bin. checks if zenith is in or out of range
        zenith_idx = self.get_zenith_bin(zenith)

        if d < np.amin(self._distances):
            print(d, np.amin(self._distances))
            sys.exit('_get_integrated_refractivity_for_distance: distance out of range')

        distance_idx = (d - self._distances[0]) / self._distance_increment
        idx = int(distance_idx)

        # if distance is out of table
        if not idx < len(self._distances) - 1:
            slope10 = (self._refractivity_integrated_table[zenith_idx, -1] - \
                       self._refractivity_integrated_table[zenith_idx, -10]) / 10
            return self._refractivity_integrated_table[zenith_idx, -1] + slope10 * (distance_idx - len(self._distances))

        f = distance_idx - idx
        return (1 - f) * self._refractivity_integrated_table[zenith_idx, idx] \
                + f * self._refractivity_integrated_table[zenith_idx, idx+1]


    def get_zenith_bin(self, zenith):
        if zenith < np.amin(self._zeniths) or zenith > np.amax(self._zeniths):
            print("get_zenith_bin zenith out of range: {} not in [{}, {}]".format(*np.rad2deg([zenith,
                                                                                     np.amin(self._zeniths),
                                                                                     np.amax(self._zeniths)])))
            raise ValueError
        return np.argmin(np.abs(self._zeniths - zenith))


    def get_integrated_refractivity_for_distance(self, d, zenith):
        if not self._curved:
            sys.exit("Config is wrong: not curved")

        if not self._interpolate_zenith:
            return self._get_integrated_refractivity_for_distance(d, zenith)

        zenith_idx = self.get_zenith_bin(zenith)

        if self._zeniths[zenith_idx] - zenith < 0:
            bin_low, bin_up = zenith_idx, zenith_idx + 1
        else:
            bin_low, bin_up = zenith_idx - 1, zenith_idx

        rlow = self._get_integrated_refractivity_for_distance(d, self._zeniths[bin_low])
        rup = self._get_integrated_refractivity_for_distance(d, self._zeniths[bin_up])

        if rlow < rup:
            rinterp = rlow + (rup - rlow) / (self._zeniths[bin_up] - self._zeniths[bin_low]) * (zenith - self._zeniths[bin_low])
        elif rup > rlow:
            rinterp = rup + (rlow - rup) / (self._zeniths[bin_up] - self._zeniths[bin_low]) * (zenith - self._zeniths[bin_low])
        else:
            rinterp = rlow

        return rinterp


    def get_refractivity_between_two_points_from_distance(self, zenith, d1, d2):
        r1 = self.get_integrated_refractivity_for_distance(d1, zenith=zenith)
        r2 = self.get_integrated_refractivity_for_distance(d2, zenith=zenith)

        return (r2 - r1) / (d2 - d1)


    def get_refractivity_between_two_altitudes(self, h1, h2):
        r1 = self.get_integrated_refractivity_for_height_tabulated(h1)
        r2 = self.get_integrated_refractivity_for_height_tabulated(h2)

        return (r2 - r1) / (h2 - h1)


    def get_refractivity_between_two_points_tabulated(self, p1, p2):
        dist = np.linalg.norm(p1 - p2)
        zenith_local = helper.get_local_zenith_angle(p1, p2)
        obs_level_local = helper.get_local_altitude(p2)

        if zenith_local < np.amin(self._zeniths):
            return self.get_refractivity_between_two_altitudes(obs_level_local, helper.get_local_altitude(p1))

        zenith_at_earth, distance_to_earth = helper.get_zenith_angle_at_earth(zenith_local, obs_level_local)
        d2 = dist + distance_to_earth

        if zenith_at_earth < np.amin(self._zeniths):
            return self.get_refractivity_between_two_altitudes(obs_level_local, helper.get_local_altitude(p1))

        return self.get_refractivity_between_two_points_from_distance(zenith_at_earth, distance_to_earth, d2)


    def get_refractivity_between_to_points_numerical(self, p1, p2, debug=False):
        return get_refractivity_between_to_points_numerical(p1, p2, atmModel=self._atm_model,
            n_asl=1+self._refractivity_at_sea_level, debug=debug)


def get_refractivity_between_to_points_numerical(p1, p2, atmModel, n_asl, debug=False):

    line = p1 - p2
    max_dist = np.linalg.norm(line)
    zenith = helper.get_local_zenith_angle(p1, p2)

    nsteps = max(1000, min(int(max_dist / 100), 2001))
    distances = np.linspace(0, max_dist, nsteps, endpoint=False)
    obs_level = helper.get_local_altitude(p2)

    dstep = distances[1] - distances[0]
    refractivity = 0
    for dist in distances:
        height_asl = atm.get_height_above_ground(dist + dstep / 2, zenith, observation_level=obs_level) + obs_level
        refractivity += (atm.get_n(height_asl, n0=n_asl, allow_negative_heights=False, model=atmModel) - 1) * dstep

    if debug:
        height_max = atm.get_height_above_ground(max_dist, zenith, observation_level=obs_level) + obs_level
        print("calculate num for %.3f deg, %.1f m distance, %.1f m min height, %.1f m max height: N = %.3e" %
             (np.rad2deg(zenith), max_dist, obs_level, height_max, refractivity / max_dist))

    return refractivity / max_dist


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from scipy import constants
    from collections import defaultdict

    tab = RefractivityTable(atm_model=27, interpolate_zenith=True, curved=True, number_of_zenith_bins=1000)
    at = atm.Atmosphere(27)

    data = defaultdict(list)

    zeniths = np.deg2rad(np.arange(65, 86, 5))
    depths = [400]
    core = np.array([0, 0, 1400])

    positions = np.array([np.linspace(-1000, 1000, 20), np.zeros(20), np.zeros(20)]).T + core
    print(positions)
    print(positions.shape)


    for zenith in zeniths:
        print('zenith', np.rad2deg(zenith))
        shower_axis = helper.spherical_to_cartesian(zenith, 0)

        for depth in depths:
            dist = at.get_distance_xmax_geometric(zenith, depth, observation_level=core[-1])
            if dist < 0:
                continue

            # sea level
            point_on_axis = shower_axis * dist + core
            point_on_axis_height = atm.get_height_above_ground(dist, zenith, observation_level=core[-1]) + core[-1]
            print("Height of point in inital sys:", point_on_axis_height)

            for pos in positions:
                r_num = tab.get_refractivity_between_to_points_numerical(point_on_axis, pos, debug=False)
                r_tab = tab.get_refractivity_between_two_points_tabulated(point_on_axis, pos)

                zenith_eff = helper.get_local_zenith_angle(point_on_axis, pos)
                obs_level_local = helper.get_local_altitude(pos)
                pos_dist = np.linalg.norm(point_on_axis - pos)

                data["r_num"].append(r_num)
                data["r_tab"].append(r_tab)
                data["distances"].append(pos_dist)
                data["zenith_station"].append(zenith_eff)


        data["zeniths"] += [zenith] * len(positions) * len(depths)

    for key in data:
        data[key] = np.array(data[key])

    fig, axs = plt.subplots(2)

    t_num = (data["distances"] / constants.c * (data["r_num"] + 1))
    t_tab = (data["distances"] / constants.c * (data["r_tab"] + 1))

    axs[0].plot(np.rad2deg(data["zenith_station"]), t_num * 1e6, "o", markersize=10, label="numerical")
    axs[0].plot(np.rad2deg(data["zenith_station"]), t_tab * 1e6, "o", label="tabulated")
    axs[0].set_ylabel(r"$t$ / $\mu$s")

    axs[1].plot(np.rad2deg(data["zenith_station"]), (t_tab - t_num) * 1e9, "o")
    axs[1].set_ylabel(r"$t_\mathrm{tab}$ - $t_\mathrm{num}$ / ns")


    axs[0].legend()
    axs[1].set_xlabel(r"zenith angle / deg")
    plt.tight_layout()
    plt.show()
