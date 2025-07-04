from radiotools.atmosphere import models as atm
from radiotools import helper

import numpy as np
import sys
import os

import logging

logger = logging.getLogger('radiotools.atmosphere.refractivity')


"""
For the integrated refractivity from the table as function of the zenith angle and distance (curved is True) a minor,
almost stable, bias of ~ 0.05 ns (propagated to propagation time) w.r.t. the numerical calculation is found. This is
not found for the integrated refractivity from the table as function of height only. However this table assumes a flat
atmosphere and thus loses accuracy for zenith angles beyond 60 - 70 deg.
"""


def n_param_ZHAireS(h):
    """
    Parametrisation for the refractivity N = n - 1, as funtion of height above sea level N(h) as used in ZHAireS.

    Parameters
    ----------
    h : float
        Height over sea level in meter

    Returns
    -------

    refractivity : float
        Refractive index for given height
    """

    a = 325e-6
    b = 0.1218e-3
    return 1 + a * np.exp(-b * h)


def get_refractivity_between_two_points_numerical(p1, p2, atm_model=None, refractivity_at_sea_level=None, table=None, debug=None):
    """
    Numerical calculation of the integrated refractivity between two positions along a straight line in the atmosphere.
    Takes curvature of a spherical earth into account.
    p1 and p2 need to be in the same coordiate system with the origin at sea level.

    Parameters
    ----------

    p1 : array (3,)
        coordinates in meter.
    p2 : array (3,)
        coordinates in meter.
    atm_model : int
        Number of the atmospheric model (from radiotools.atmosphere.models) which provides the desity profile rho(h)
        to calculate the refractivity via Gladstone-Dale relation: N(h) = N(0) * rho(h) / rho(0).
        Is only used if "table" is not None (default: None)
    refractivity_at_sea_level : float
        Refractivity at earth surface, i.e., N(0) (default: None). Necessary if refractivity is calculated
        via Gladstone-Dale relation. Not necessary/used if a RefractivityTable is given.
    table : RefractivityTable
        If given, used to determine N(h). Instead of using Gladstone-Dale relation. (default: None)

    Returns
    -------

    integrated_refractivity : float
        Refractive index integrated along a straight line between two given points
    """
    if debug is not None:
        logger.warning(
            "The debug parameter of get_refractivity_between_two_points_numerical has been deprecated. "
            "Please use the logger functionality instead. "
            "For your convenience, I will now set the logging level of radiotools to DEBUG."
        )
        logger.setLevel(logging.DEBUG)

    if table is None and (atm_model is None and refractivity_at_sea_level is None):
        sys.exit("Invalid arguments. You have to specify table or atm_model and refractivity_at_sea_level.")

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
        if table is None:
            refractivity += (atm.get_n(height_asl, n0=refractivity_at_sea_level+1, allow_negative_heights=False,
                model=atm_model) - 1) * dstep
        else:
            refractivity += table.get_refractivity_for_height_tabulated(height_asl) * dstep

    height_max = atm.get_height_above_ground(max_dist, zenith, observation_level=obs_level) + obs_level
    logger.debug("calculate num for %.3f deg, %.1f m distance, %.1f m min height, %.1f m max height: N = %.3e" %
            (np.rad2deg(zenith), max_dist, obs_level, height_max, refractivity / max_dist))

    return refractivity / max_dist


class RefractivityTable(object):
    """
    Class to calculate the integrated refractivity between two positions in the atmosphere.
    """


    def __init__(self, atm_model=atm.default_model, param=False, refractivity_at_sea_level=312e-6, curved=False,
                 interpolate_zenith=True, number_of_zenith_bins=1000, distance_increment=100, gdas_file=None):

        """
        Parameters
        ----------

        atm_model : int
            Number of the atmospheric model (from radiotools.atmosphere.models) which provides the desity profile rho(h)
            to calculate the refractivity via N(h) = N(0) * rho(h) / rho(0) (Only if param is False).
        param : bool
            If True, uses ZHAireS parametrisation to calculate N(h) (default: False).
        refractivity_at_sea_level : float
            Refractivity at earth surface, i.e., N(0) (default: 312e-6).
        curved : bool
            If True, initializes tables for the integrated refractivity as function of zenith angle and distance.
        interpolate_zenith : bool
            If True, linear interpolation in zenith angle, if False, select closest zenith angle bin. Only relevant if
            curved is True (default: True).
        number_of_zenith_bins : int
            Number of bins in theta(zenith). Only relevant if curved is True (default: 1000).
        distance_increment : float
            Bin size in distance in meter. Only relevant if curved is True (default: 100).

        """

        self._read_profile_from_gdas = gdas_file is not None

        if self._read_profile_from_gdas:
            self._gdas_file_name = os.path.basename(gdas_file)
            self.parse_gdas_file(gdas_file)
            self._use_param = False
            self._atm_model = None
        else:
            self._refractivity_at_sea_level = refractivity_at_sea_level
            self._use_param = param
            self._atm_model = atm_model

            # set fix values.
            self._height_increment = 10
            self._max_heigth = 4e4
            self._heights = np.arange(0, self._max_heigth, self._height_increment)

            rho0 = atm.get_density(0, allow_negative_heights=False, model=self._atm_model)
            if self._use_param:
                self._refractivity_table = np.array([n_param_ZHAireS(h) - 1 for h in self._heights])
            else:
                self._refractivity_table = np.array([self._refractivity_at_sea_level * \
                    atm.get_density(h, allow_negative_heights=False, model=self._atm_model) \
                     / rho0 for h in self._heights])

        self._refractivity_integrated_table_flat = np.cumsum(self._refractivity_table * self._height_increment)

        self._min_zenith = np.deg2rad(50)
        self._max_zenith = np.deg2rad(89)
        self._distance_increment = distance_increment
        self._number_of_zenith_bins = number_of_zenith_bins
        self._interpolate_zenith = interpolate_zenith

        self._curved = curved
        if self._curved:
            self.get_cached_table_curved_atmosphere()

    def parse_gdas_file(self, gdas_file):
        with open(gdas_file, "r") as f:
            lines = f.readlines()

            # atm_para = [line.strip("\n").split() for line in lines[1:5]]
            self._heights = np.zeros(len(lines) - 6)
            self._refractivity_table = np.zeros(len(lines) - 6)
            for idx, line in enumerate(lines[6:]):
                h, n = line.strip("\n").split()
                self._heights[idx] = float(h)
                self._refractivity_table[idx] = float(n) - 1

            self._height_increment = self._heights[1] - self._heights[0]
            self._max_heigth = np.amax(self._heights)

            # just take the "layer" closet value to 0 (sea level) if its within 1m. This is stupid but should accurate enought.
            null = np.argmin(np.abs(self._heights))
            if np.abs(self._heights[null]) > 1:
                sys.exit("Could not find refractive index at sea level in gdas profile. stop...")

            self._refractivity_at_sea_level = self._refractivity_table[null]


    def read_table_from_file(self, fname):
        logger.info("Read in {} ...".format(fname))
        data = np.load(fname, "r", allow_pickle=True)
        self._refractivity_integrated_table = data["refractivity_integrated_table"]
        self._distance_increment = data["distance_increment"]
        self._distances = data["distances"]
        self._zeniths = data["zeniths"]


    def get_cached_table_curved_atmosphere(self):
        """
        Get table of integrated refractivity as function of zenith angle and distance. For curved atmosphere.
        Will try to read table from numpy file if exsits and will write table to numpy file if it does not.
        """

        basedir = os.path.dirname(os.path.abspath(__file__))
        if self._read_profile_from_gdas:
            fname_tmp = "refractivity_%.0f_%s.npz" % (
                self._refractivity_at_sea_level * 1e6, self._gdas_file_name.replace(".DAT", ""))
        else:
            fname_tmp = "refractivity_%02d_%.0f_%d_%d.npz" % (
                self._atm_model, self._refractivity_at_sea_level * 1e6, self._number_of_zenith_bins, self._distance_increment)

        fname = os.path.join(basedir, fname_tmp)

        if os.path.exists(fname):
            self.read_table_from_file(fname)

        else:
            logger.info("Write {} ...".format(fname))

            # anchors for table binned in tan.
            self._zeniths = np.arctan(np.linspace(np.tan(self._min_zenith),
                                                  np.tan(self._max_zenith), self._number_of_zenith_bins))
            self._distances = []
            self._refractivity_integrated_table = []

            for zdx, zen in enumerate(self._zeniths):
                max_dist = atm.get_distance_for_height_above_ground(self._max_heigth, zen, 0)
                distances = np.arange(0, max_dist, self._distance_increment)

                refractivities_for_distances = np.array([self.get_refractivity_for_height_tabulated(
                    atm.get_height_above_ground(d, zen, observation_level=0)) for d in distances])

                refractivity_integrated_table = np.cumsum(refractivities_for_distances * self._distance_increment)

                self._distances.append(distances)
                self._refractivity_integrated_table.append(refractivity_integrated_table)

            self._distances = np.array(self._distances, dtype=object)
            self._refractivity_integrated_table = np.array(self._refractivity_integrated_table, dtype=object)

            data = {"refractivity_integrated_table": self._refractivity_integrated_table,
                    "distance_increment": self._distance_increment,
                    "distances": self._distances,
                    "zeniths": self._zeniths}

            np.savez(fname, **data)

    def get_refractivity_for_height_tabulated(self, h):
        """
        Get refractivity as function of height above ground from pre-calculated table.
        Linear interpolation between table anchors.
        """

        if h == 0:
            return self._refractivity_at_sea_level

        fidx = (h - self._heights[0]) / self._height_increment
        idx = int(fidx)
        f = fidx - idx

        # if height is out of table (right edge): interpolate
        if not idx < len(self._heights) - 1:
            slope10 = (self._refractivity_table[-1] - self._refractivity_table[-10]) / 10
            return self._refractivity_table[-1] + slope10 * (fidx - len(self._heights))

        return (1-f) * self._refractivity_table[idx] + f * self._refractivity_table[idx+1]


    def get_integrated_refractivity_for_height_tabulated(self, h):
        """
        Get integrated refractivity as function of height above ground from pre-calculated table.
        Linear interpolation between table anchors.
        """
        if h == 0:
            return self._refractivity_at_sea_level

        fidx = (h - self._heights[0]) / self._height_increment
        idx = int(fidx)
        f = fidx - idx

        # if height is out of table (right edge): interpolate
        if not idx < len(self._heights) - 1:
            slope10 = (self._refractivity_integrated_table_flat[-1] - \
                self._refractivity_integrated_table_flat[-10]) / 10
            return self._refractivity_integrated_table_flat[-1] + slope10 * (fidx - len(self._heights))

        return (1-f) * self._refractivity_integrated_table_flat[idx] + f * \
            self._refractivity_integrated_table_flat[idx+1]


    def _get_integrated_refractivity_for_distance(self, d, zenith):
        """
        Get integrated refractivity as function of the zenith angle and distance from ground
        (along a axis with the specified zenith angle) from pre-calculated table.
        Takes clostest zenith angle from table to calculated the refractivity.
        """
        if not self._curved:
            sys.exit("Table not available: please specifiy \"curved=True\"")

        # next zenith bin. checks if zenith is in or out of range
        zenith_idx = self.get_zenith_bin(zenith)

        # if distance is out of table (left edge): fail
        if d < np.amin(self._distances[zenith_idx]):
            raise ValueError(r"Requested distance is out of range: {} ($\theta$ = {})".format(
                d, np.rad2deg(zenith)))

        distance_idx = (d - self._distances[zenith_idx][0]) / self._distance_increment
        idx = int(distance_idx)

        # if distance is out of table (right edge): interpolate
        if not idx < len(self._distances[zenith_idx]) - 1:
            slope10 = (self._refractivity_integrated_table[zenith_idx][-1] - \
                       self._refractivity_integrated_table[zenith_idx][-10]) / 10
            return self._refractivity_integrated_table[zenith_idx][-1] + slope10 \
                 * (distance_idx - len(self._distances[zenith_idx]))

        f = distance_idx - idx
        return (1 - f) * self._refractivity_integrated_table[zenith_idx][idx] \
                + f * self._refractivity_integrated_table[zenith_idx][idx+1]


    def get_zenith_bin(self, zenith):
        """ get index of closest zenith bin """

        if zenith < np.amin(self._zeniths) or zenith > np.amax(self._zeniths):
            raise ValueError("get_zenith_bin zenith out of range: {} not in [{}, {}]".format(
                *np.rad2deg([zenith, np.amin(self._zeniths), np.amax(self._zeniths)])))

        return np.argmin(np.abs(self._zeniths - zenith))


    def get_integrated_refractivity_for_distance(self, d, zenith):
        """
        Get integrated refractivity as function of the zenith angle and distance from ground
        (along a axis with the specified zenith angle) from pre-calculated table.
        If "_interpolate_zenith" is True, a linear interpolation in zenith angle is performed,
        otherwise the clostest zenith angle bin is used for the calculation.
        """
        if not self._curved:
            sys.exit("Table not available: please specifiy \"curved=True\"")

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
            rinterp = rlow + (rup - rlow) / (self._zeniths[bin_up] - self._zeniths[bin_low]) \
                 * (zenith - self._zeniths[bin_low])
        elif rup > rlow:
            rinterp = rup + (rlow - rup) / (self._zeniths[bin_up] - self._zeniths[bin_low]) \
                * (zenith - self._zeniths[bin_low])
        else:
            rinterp = rlow

        return rinterp


    def get_refractivity_between_two_points_from_distance(self, zenith, d1, d2):
        """ Get integrated refractivity along line (zenith) between two distances in curved atmosphere table """
        r1 = self.get_integrated_refractivity_for_distance(d1, zenith=zenith)
        r2 = self.get_integrated_refractivity_for_distance(d2, zenith=zenith)

        return (r2 - r1) / (d2 - d1)


    def get_refractivity_between_two_altitudes(self, h1, h2):
        """ Get integrated refractivity between two altitudes in flat atmosphere table """
        r1 = self.get_integrated_refractivity_for_height_tabulated(h1)
        r2 = self.get_integrated_refractivity_for_height_tabulated(h2)

        return (r2 - r1) / (h2 - h1)


    def get_refractivity_between_two_points_tabulated(self, p1, p2):
        """
        Get integrated refractivity between two positions in curved atmosphere.
        If not curved or zenith angle below table use flat atmosphere.
        If curved, zenith angle above range or SystemExit happens use numerical solution.
        """

        dist = np.linalg.norm(p1 - p2)
        zenith_local = helper.get_local_zenith_angle(p1, p2)
        obs_level_local = helper.get_local_altitude(p2)

        # return flat solution
        if not self._curved or zenith_local < np.amin(self._zeniths):
            return self.get_refractivity_between_two_altitudes(obs_level_local, helper.get_local_altitude(p1))

        try:
            zenith_at_sea_level, distance_to_earth = helper.get_zenith_angle_at_sea_level(
                zenith_local, obs_level_local)
            d2 = dist + distance_to_earth
        except SystemExit:
            logger.warning("Catch SystemExit while calculating zenith at earth, resuming with numerical calculation")
            return self.get_refractivity_between_two_points_numerical(p1, p2)

        if zenith_at_sea_level < np.amin(self._zeniths):
            return self.get_refractivity_between_two_altitudes(obs_level_local, helper.get_local_altitude(p1))

        if zenith_at_sea_level > np.amax(self._zeniths):
            logger.warning("Zenith out of range, perform numerical calculation")
            return self.get_refractivity_between_two_points_numerical(p1, p2)

        return self.get_refractivity_between_two_points_from_distance(zenith_at_sea_level, distance_to_earth, d2)


    def get_refractivity_between_two_points_numerical(self, p1, p2, debug=None):
        """ Get numerical calculated integrated refractivity between two positions in atmosphere """
        return get_refractivity_between_two_points_numerical(p1, p2, table=self, debug=debug)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from scipy import constants
    from collections import defaultdict

    atm_model = 27
    tab_flat = RefractivityTable(atm_model=atm_model, curved=False, number_of_zenith_bins=1000)
    tab = RefractivityTable(atm_model=atm_model, interpolate_zenith=True, curved=True, number_of_zenith_bins=1000)
    at = atm.Atmosphere(atm_model)

    data = defaultdict(list)

    zeniths = np.deg2rad(np.arange(55, 80, 1))
    depths = [400]
    core = np.array([0, 0, 1400])

    positions = np.array([np.linspace(-1000, 1000, 20), np.zeros(20), np.zeros(20)]).T + core
    logger.debug(positions)
    logger.debug(positions.shape)


    for zenith in zeniths:
        logger.debug('zenith = %s', np.rad2deg(zenith))
        shower_axis = helper.spherical_to_cartesian(zenith, 0)

        for depth in depths:
            dist = at.get_distance_xmax_geometric(zenith, depth, observation_level=core[-1])
            if dist < 0:
                continue

            # sea level
            point_on_axis = shower_axis * dist + core
            point_on_axis_height = atm.get_height_above_ground(dist, zenith, observation_level=core[-1]) + core[-1]
            logger.info("Height of point in inital sys: %s", point_on_axis_height)

            for pos in positions:
                r_num = tab.get_refractivity_between_two_points_numerical(point_on_axis, pos)
                r_tab = tab.get_refractivity_between_two_points_tabulated(point_on_axis, pos)
                r_tab_flat = tab_flat.get_refractivity_between_two_altitudes(
                    helper.get_local_altitude(pos), helper.get_local_altitude(point_on_axis))

                zenith_eff = helper.get_local_zenith_angle(point_on_axis, pos)
                obs_level_local = helper.get_local_altitude(pos)
                pos_dist = np.linalg.norm(point_on_axis - pos)

                data["r_num"].append(r_num)
                data["r_tab"].append(r_tab)
                data["r_tab_flat"].append(r_tab_flat)
                data["distances"].append(pos_dist)
                data["zenith_station"].append(zenith_eff)


        data["zeniths"] += [zenith] * len(positions) * len(depths)

    for key in data:
        data[key] = np.array(data[key])

    fig, axs = plt.subplots(2)

    t_num = (data["distances"] / constants.c * (data["r_num"] + 1))
    t_tab = (data["distances"] / constants.c * (data["r_tab"] + 1))
    t_tab_flat = (data["distances"] / constants.c * (data["r_tab_flat"] + 1))
    t_uni = (data["distances"] / constants.c)

    axs[0].plot(np.rad2deg(data["zenith_station"]), t_num * 1e6, "C0o", markersize=10, label="numerical")
    axs[0].plot(np.rad2deg(data["zenith_station"]), t_tab * 1e6, "C1o", label="tabulated")
    axs[0].plot(np.rad2deg(data["zenith_station"]), t_tab_flat * 1e6, "C2o", label="tabulated flat")
    axs[0].plot(np.rad2deg(data["zenith_station"]), t_uni * 1e6, "C3o", label="unity")
    axs[0].set_ylabel(r"$t$ / $\mu$s")

    axs[1].plot(np.rad2deg(data["zenith_station"]), (t_tab - t_num) * 1e9, "C1o", label="tabulated")
    axs[1].plot(np.rad2deg(data["zenith_station"]), (t_tab_flat - t_num) * 1e9, "C2o", label="tabulated flat")
    axs[1].plot(np.rad2deg(data["zenith_station"]), (t_uni - t_num) * 1e9, "C3o", label="unity")
    axs[1].set_ylabel(r"$t_\mathrm{i}$ - $t_\mathrm{num}$ / ns")


    axs[0].legend()
    axs[1].set_xlabel(r"zenith angle / deg")
    plt.tight_layout()
    plt.show()
