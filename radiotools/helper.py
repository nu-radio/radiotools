#!/usr/bin/env python
# -*- coding: utf-8 -*-
from radiotools.atmosphere import models as atm
import numpy as np
import sys
from scipy.signal import correlate

import logging

logger = logging.getLogger('radiotools.helper')


def linear_to_dB(linear):
    """ conversion to decibel scale

    Parameters
    ------------
    linear : float
        quantity in linear units

    Returns
    --------
    dB : float
        quantity in dB units
    """
    return 10 * np.log10(linear)


def dB_to_linear(dB):
    """ conversion from decibel scale to linear scale

    Parameters
    ------------
    dB : float
        quantity in dB units

    Returns
    --------
    linear : float
        quantity in linear units
    """
    return 10 ** (dB / 10.)


def gps_to_datetime(gps):
    """ conversion between GPS seconds and a python datetime object (taking into account leap seconds) """
    from radiotools import leapseconds
    from datetime import datetime, timedelta
    return leapseconds.gps_to_utc(datetime(1980, 1, 6) + timedelta(seconds=gps))


def datetime_to_gps(date):
    from radiotools import leapseconds
    from datetime import datetime
    return (leapseconds.utc_to_gps(date) - datetime(1980, 1, 6)).total_seconds()


def GPS_to_UTC(gps):
    offset = 315964800 + 16
    return gps + offset


def UTC_to_GPS(utc):
    return utc - GPS_to_UTC(0)


def datetime_to_UTC(dt):
    import calendar
    return calendar.timegm(dt.timetuple())


def get_local_zenith(pos):
    """
    Assumes spherical earth. Returns direction of zenith for given position "pos".
    "pos" needs to be given in coordinate system with the origin at sea level.

    Parameters
    ------------
    pos : array (3,)
        coordinates in meter

    Returns
    --------
    zenith : array (3,)
        unity vector of local zenith
    """
    origin = np.array([0, 0, atm.r_e])
    return (origin + pos) / np.linalg.norm(origin + pos)


def get_local_altitude(pos):
    """
    Assumes spherical earth. Returns height above sea level for position "pos".
    "pos" needs to be given in coordinate system with the origin at sea level.

    Parameters
    ------------
    pos : array (3,)
        coordinates in meter

    Returns
    --------
    altitude : float
        height above ground of pos
    """
    pos_tot = np.array([0, 0, atm.r_e]) + pos
    return np.linalg.norm(pos_tot - get_local_zenith(pos) * atm.r_e)


def get_local_zenith_angle(psource, preciever):
    """
    Assumes spherical earth. Returns zenith angle under which a reciever (preciever) sees a source (psource).
    "preciever" and "psource" have to be in the same coordinate system with the origin at sea level.

    Parameters
    ------------
    psource : array (3,)
        coordinates in meter
    psource : array (3,)
        coordinates in meter

    Returns
    --------
    zenith angle : float
        local zenith angle of psource at preciever in rad
    """
    local_zenith = get_local_zenith(preciever)
    line = psource - preciever
    return get_angle(local_zenith, line)


def get_intersection_between_circle_and_line(r, b, c):
    """
    solution from: https://cp-algorithms.com/geometry/circle-line-intersection.html
    calculation in 2 dimension.

    Parameters
    ------------
    r : float
        circle radius in meter
    b, c: float, float
        line parameter

    Returns
    --------
    x0, y0 : float, float
        coordinates of intersection(s)
    """
    a = 1  # without loss of generality set a to 1
    eps = 1.e-6
    r0 = (a ** 2 + b ** 2)
    x0 = -a * c / r0
    y0 = -b * c / r0
    if c ** 2 > r ** 2 * r0 + eps:
        # no intersection
        return 0, 0
    elif np.abs(c ** 2 - r ** 2 * r0) < eps:
        # one intersection
        return x0, y0
    else:
        # two intersections
        d = r ** 2 - c ** 2 / r0
        mult = np.sqrt(d / r0)
        return x0 + b * mult, y0 - a * mult, x0 - b * mult, y0 + a * mult


def get_zenith_angle_at_sea_level(zenith, observer_level):
    """
    Calculates intersections of a line with an anchor at an observation level and zenith angle with a spherical earth.
    Determines distance along line between clostest intersection and the observation level and local zenith angle at that
    intersection.

    Parameters
    ------------
    zenith : float
        zenith angle of line in rad
    observer_level : float
        observation level in meter

    Returns
    --------
    local_zenith : float
        local zenith angle of line at earth surface in rad
    distance : float
        distance between found intersection and observation level in meter
    """

    if observer_level == 0:
        return zenith, 0

    r_e = atm.r_e
    coors = get_intersection_between_circle_and_line(r_e, -np.tan(zenith), (r_e + observer_level) * np.tan(zenith))

    # only accept two intersections found
    if len(coors) > 2:
        # take the closer one
        if coors[1] > coors[3]:
            x, y = coors[0], coors[1]
        else:
            x, y = coors[2], coors[3]

        if 0:
            from matplotlib import pyplot as plt
            earth = spherical_to_cartesian(zenith=np.linspace(0, np.pi * 2, 1000), azimuth=0) * atm.r_e
            xs = np.linspace(-r_e * 0.5, r_e * 0.5)
            line = xs / np.tan(zenith) + r_e + observer_level

            lot1 = np.array([0, 0, 1]) * np.linspace(0, atm.r_e + observer_level + 1000)[:, None]
            lot2 = np.array([x, 0, y]) / np.linalg.norm(np.array([x, 0, y])) * np.linspace(0, atm.r_e + observer_level + 1000)[:, None]

            plt.plot(x, y, "r*")
            plt.plot(earth[0], earth[2])
            plt.plot(xs, line)
            plt.plot(lot1[:, 0], lot1[:, 2], "k--")
            plt.plot(lot2[:, 0], lot2[:, 2], "k--")
            plt.show()

        # calculate
        v1 = np.array([x, 0, y])
        distance = np.linalg.norm(v1 - np.array([0, 0, r_e + observer_level]))
        line = spherical_to_cartesian(zenith, 0)
        local_zenith = get_angle(v1, line)

        return local_zenith, distance
    else:
        sys.exit("Find theta at earth: Not 2 intersections")


def spherical_to_cartesian(zenith, azimuth):
    sinZenith = np.sin(zenith)
    x = sinZenith * np.cos(azimuth)
    y = sinZenith * np.sin(azimuth)
    z = np.cos(zenith)
    if hasattr(zenith, '__len__') and hasattr(azimuth, '__len__'):
        return np.array([x, y, z]).T
    else:
        return np.array([x, y, z])


def cartesian_to_spherical(x, y, z):
    # normlize vector
    norm = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    x2 = x / norm
    y2 = y / norm
    z2 = z / norm
    theta = 0
    if hasattr(x, '__len__') and hasattr(y, '__len__')and hasattr(z, '__len__'):
        theta = np.zeros_like(x)
        theta[z2 < 1] = np.arccos(z2[z2 < 1])
        phi = np.arctan2(y2, x2)
        return theta, phi
    else:
        if (z2 < 1):
            theta = np.arccos(z2)
        phi = np.arctan2(y2, x2)
        return theta, phi


def get_angle(v1, v2):
    """
    Calculates the angle between two vectors.

    Parameters
    ----------
    v1: 3d array or list of 3d arrays
        vector(s) one
    v2: 3d array
        vector two

    Returns: float or list of floats
        angle(s) between vector(s)
    """
    
    if v1.ndim == 2 and v2.ndim == 2:
        arccos = np.array([
            np.dot(v1_, v2_) / (np.linalg.norm(v1_.T, axis=0) * np.linalg.norm(v2_.T, axis=0))
            for v1_, v2_ in zip(v1, v2)])
    else:
        arccos = np.dot(v1, v2) / (np.linalg.norm(v1.T, axis=0) * np.linalg.norm(v2.T, axis=0))
    # catch numerical overlow
    mask1 = arccos > 1
    mask2 = arccos < -1
    mask = np.logical_or(mask1, mask2)
    if (type(mask) != np.bool_):
        arccos[mask1] = 1
        arccos[mask2] = -1
    else:
        if (mask1):
            arccos = 1
        if (mask2):
            arccos = -1
    return np.arccos(arccos)


def get_rotation(v1, v2):
    """
    calculates the rotation matrix to transform vector 1 to vector 2
    """
    v = np.cross(v1, v2)
    phi = get_angle(v1, v2)
    v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    vx2 = np.matmul(v_x, v_x)
    return np.identity(3) + v_x + vx2 * 1. / (1 + np.cos(phi))


def get_normalized_angle(angle, degree=False, interval=np.deg2rad([0, 360])):
    import collections.abc
    if degree:
        interval = np.rad2deg(interval)
    delta = interval[1] - interval[0]
    if(isinstance(angle, (collections.abc.Sequence, np.ndarray))):
        angle[angle >= interval[1]] -= delta
        angle[angle < interval[0]] += delta
    else:
        while (angle >= interval[1]):
            angle -= delta
        while (angle < interval[0]):
            angle += delta
    return angle


def get_declination(magnetic_field_vector):
    declination = np.arccos(np.dot(np.array([0, 1]), magnetic_field_vector[:2] /
                                   np.linalg.norm(magnetic_field_vector[:2])))
    return declination


def get_magnetic_field_vector(site=None):
    """
    get the geomagnetic field vector in Gauss. x points to geographic East and y towards geographic North
    """
    magnetic_fields = {'auger': np.array([0.00871198, 0.19693423, 0.1413841]),
                       'mooresbay': np.array([0.058457, -0.09042, 0.61439]),
                       'summit': np.array([-.037467, 0.075575, -0.539887]),  # Summit station, Greenland
                       'southpole': np.array([-0.14390398, 0.08590658, 0.52081228]),  # position of SP arianna station
                       'lofar': np.array([0.004675, 0.186270, -0.456412])  # values from 2015
                       }  
    if site is None:
        site = 'auger'
    return magnetic_fields[site.lower()]


def get_angle_to_magnetic_field_vector(zenith, azimuth, site=None):
    """
        returns the angle between shower axis and magnetic field
    """
    magnetic_field = get_magnetic_field_vector(site=site)
    v = spherical_to_cartesian(zenith, azimuth)
    return get_angle(magnetic_field, v)


def get_magneticfield_azimuth(magnetic_field_declination):
    return magnetic_field_declination + np.deg2rad(90)


def get_inclination(magnetic_field_vector):
    zenith, azimuth = cartesian_to_spherical(*magnetic_field_vector)
    return np.deg2rad(90) - zenith


def get_magneticfield_zenith(magnetic_field_inclination):
    if (magnetic_field_inclination < 0):
        return magnetic_field_inclination + np.deg2rad(90)
    else:
        return np.deg2rad(90) - magnetic_field_inclination


def get_magnetic_field_vector_from_inc(inclination, declination):
    return spherical_to_cartesian(get_magneticfield_zenith(inclination),
                                  get_magneticfield_azimuth(declination))


def get_lorentzforce_vector(zenith, azimuth, magnetic_field_vector=None):
    if (magnetic_field_vector is None):
        magnetic_field_vector = get_magnetic_field_vector()
    showerAxis = spherical_to_cartesian(zenith, azimuth)
    magnetic_field_vector_normalized = magnetic_field_vector / \
        np.linalg.norm(magnetic_field_vector.T, axis=0, keepdims=True).T
    return np.cross(showerAxis, magnetic_field_vector_normalized)


def get_sine_angle_to_lorentzforce(zenith, azimuth, magnetic_field_vector=None):
    # we use the tanspose of the vector or matrix to be able to always use
    # axis=0
    return np.linalg.norm(get_lorentzforce_vector(zenith, azimuth, magnetic_field_vector).T, axis=0)


def get_chargeexcess_vector(core, zenith, azimuth, stationPosition):
    showerAxis = spherical_to_cartesian(zenith, azimuth)
    magnitues = np.dot((stationPosition - core), showerAxis)
    showerAxis = np.outer(magnitues, showerAxis)
    chargeExcessVector = core - stationPosition + showerAxis
    norm = np.linalg.norm(chargeExcessVector.T, axis=0)
    chargeExcessVector = (chargeExcessVector.T / norm).T
    return np.squeeze(chargeExcessVector)


def get_chargeexcess_correction_factor(core, zenithSd, azimuthSd, stationPositions, a=0.14, magnetic_field_vector=None):
    chargeExcessVector = get_chargeexcess_vector(core, zenithSd, azimuthSd, stationPositions)
    LorentzVector = get_lorentzforce_vector(zenithSd, azimuthSd, magnetic_field_vector)
    correctionFactor = np.linalg.norm((LorentzVector + a * chargeExcessVector).T, axis=0)
    return correctionFactor


def get_polarization_vector_max(trace):
    """ calculates polarization vector of efield trace (vector at maximum pulse position)
    """
    from scipy.signal import hilbert

    h = np.sqrt(np.sum(np.abs(hilbert(trace)) ** 2, axis=0))
    max_pos = h.argmax()
    pol = trace[:, max_pos]
    return pol


def get_interval(trace, scale=0.5):
    h = np.abs(trace)
    max_pos = h.argmax()
    n_samples = trace.T.shape[0]
    h_max = h.max()
    up_pos = max_pos
    low_pos = max_pos
    for i in range(max_pos, n_samples):
        if (h[i] < h_max * scale):
            up_pos = i
            break
    for i in range(0, max_pos)[::-1]:
        if (h[i] < h_max * scale):
            low_pos = i
            break
    return low_pos, up_pos


def get_interval_hilbert(trace, scale=0.5):
    from scipy.signal import hilbert

    d = len(trace.shape)
    if (d == 1):
        h = np.abs(hilbert(trace))
    elif (d == 2):
        h = np.sqrt(np.sum(np.abs(hilbert(trace)) ** 2, axis=0))
    else:
        logger.error("trace has not the correct dimension")
        raise
    return get_interval(h, scale)


def get_FWHM_hilbert(trace):
    return get_interval_hilbert(trace, scale=0.5)


def get_polarization_vector_FWHM(trace):
    """ calculates polarization vector of efield trace,
        all vectors in the FWHM interval are averaged,
        the amplitude is set to the maximum of the hilbert envelope
    """
    from scipy.signal import hilbert

    h = np.sqrt(np.sum(np.abs(hilbert(trace)) ** 2, axis=0))
    max_pos = h.argmax()
    h_max = h.max()
    low_pos, up_pos = get_FWHM_hilbert(trace)
    sign = np.expand_dims(np.sign(trace[:, max_pos]), axis=1) * np.ones(up_pos - low_pos)
    pol = np.mean(sign * np.abs(trace[:, low_pos: up_pos]), axis=-1)
    pol /= np.linalg.norm(pol) * h_max
    return pol


def get_expected_efield_vector(core, zenith, azimuth, stationPositions, a=0.14, magnetic_field_vector=None):
    chargeExcessVector = get_chargeexcess_vector(core, zenith, azimuth, stationPositions)
    LorentzVector = get_lorentzforce_vector(zenith, azimuth, magnetic_field_vector)
    return LorentzVector + a * chargeExcessVector


def get_expected_efield_vector_vxB_vxvxB(station_positions, zenith, azimuth, a=.14):
    """
    also returns the expected electric field vector, but station positions and
    the returned field vectors are in the vxB-vxvxB coordinate system
    """
    alpha = get_angle_to_magnetic_field_vector(zenith, azimuth)
    e_geomagnetic = np.array([-np.sin(alpha), 0, 0])
    e_charge_excess = -a * station_positions / np.linalg.norm(station_positions)
    return e_charge_excess + e_geomagnetic


def get_angle_to_efieldexpectation_in_showerplane(efield, core, zenith, azimuth, stationPositions,
                                                  a=0.14, magnetic_field_vector=None):
    """ calculated the angle between a measured efield vector and the expectation
        from the geomagnetic and chargeexcess emission model. Thereby, the angular
        difference is evaluated in the showerfront, components not perpendicular to
        the shower axis are thus neglected. """
    if (efield.shape != stationPositions.shape):
        logger.error("shape of efield and station positions is not the same.")
        raise
    from CSTransformation import CSTransformation

    cs = CSTransformation(zenith, azimuth)
    efield_transformed = cs.transform_to_vxB_vxvxB(efield)
    #     print "efieldtransformed ", efield_transformed
    if (len(stationPositions.shape) == 1):
        if (efield_transformed[0] > 0):
            efield_transformed *= -1.
    else:
        for i in range(len(efield_transformed)):
            if (efield_transformed[i][0] > 0):
                efield_transformed[i] *= -1.

    efield_expectations = get_expected_efield_vector(core, zenith, azimuth,
                                                     stationPositions, a=a,
                                                     magnetic_field_vector=magnetic_field_vector)
    exp_efields_transformed = cs.transform_to_vxB_vxvxB(efield_expectations)
    #     print exp_efields_transformed
    #     print exp_efields_transformed[..., 1]
    exp_phi = np.arctan2(exp_efields_transformed[..., 1], exp_efields_transformed[..., 0])
    #     print exp_phi
    phi = np.arctan2(efield_transformed[..., 1], efield_transformed[..., 0])
    #     print phi
    diff = exp_phi - phi
    if (len(stationPositions.shape) == 1):
        while (diff > np.pi):
            diff -= 2 * np.pi
        while (diff < -np.pi):
            diff += 2 * np.pi
    else:
        for i in range(len(diff)):
            while (diff[i] > np.pi):
                diff[i] -= 2 * np.pi
            while (diff[i] < -np.pi):
                diff[i] += 2 * np.pi
    return diff


def get_distance_to_showeraxis(core, zenith, azimuth, antennaPosition):
    showerAxis = spherical_to_cartesian(zenith, azimuth)
    showerAxis = core + showerAxis
    num = np.linalg.norm(np.cross(
        antennaPosition - core, antennaPosition - showerAxis).T, axis=0)
    den = np.linalg.norm((showerAxis - core).T, axis=0)
    return num / den


def get_position_at_height(pos, height, zenith, azimuth):
    ez = np.array([0, 0, 1.])
    # t = pos * ez - np.outer(ez, height)
    # print t
    # t = np.array([0, 0, pos[2] - height])
    n = spherical_to_cartesian(zenith, azimuth)
    #     print n
    #     print "(ez * t)", np.inner(ez, t)
    #     print "(ez * n)", np.inner(ez, n)
    scaling = (pos[2] - height) / n[2]
    # print "np.outer(scaling, n)", np.outer(scaling, n)
    pos = pos - np.outer(scaling, n)
    # print pos
    return pos


def get_2d_probability(x, y, xx, yy, xx_error, yy_error, xy_correlation, sigma=False):
    from scipy.stats import multivariate_normal

    cov = np.array(
        [[xx_error ** 2, xx_error * yy_error * xy_correlation], [xx_error * yy_error * xy_correlation, yy_error ** 2]])
    p = multivariate_normal.pdf([x, y], mean=[xx, yy], cov=cov)
    denom = (2 * np.pi * xx_error * yy_error * np.sqrt(1 - xy_correlation ** 2))
    nom = np.exp(-1. / (2 * (1 - xy_correlation ** 2)) *
                 ((x - xx) ** 2 / xx_error ** 2 + (y - yy) ** 2 / yy_error ** 2 - 2 * xy_correlation * (x - xx) * (
                     y - yy) / (xx_error * yy_error)))
    if sigma:
        from scipy.stats import chi2, norm
        # p = norm.cdf(i) - norm.cdf(-i)
        logger.debug("p = %s %s %s sigma = %s" % (nom, denom, nom / denom,  chi2.ppf(nom / denom, 1)))
        logger.debug("p = %s", p)
        logger.debug(norm.ppf(nom / denom))
        return chi2.ppf(nom / denom, 1)
    else:
        return nom / denom


def is_equal(a, b, rel_precision=1e-5):
    if (a + b) != 0:
        if ((0.5 * abs(a - b) / (abs(a + b))) < rel_precision):
            return True
        else:
            return False
    else:
        if a == 0:
            return True
        else:
            if ((0.5 * abs(a - b) / (abs(a) + abs(b))) < rel_precision):
                return True
            else:
                return False


def has_same_direction(zenith1, azimuth1, zenith2, azimuth2, distancecut=20):
    distancecut = np.deg2rad(distancecut)
    axis1 = spherical_to_cartesian(zenith1, azimuth1)
    axis2 = spherical_to_cartesian(zenith2, azimuth2)

    diff = get_angle(axis1, axis2)
    if (diff < distancecut):
        return True
    else:
        return False


def get_cherenkov_angle(h, model=1):
    """ returns the cherenkov angle for the density at height above ground
        assuming that the particle speed is the speed of light """
    from radiotools.atmosphere import models as atm
    return np.arccos(1. / (atm.get_n(h, model=model)))


def get_cherenkov_ellipse(zenith, xmax, model=1):
    """ returns the major and minor axis of the cherenkov cone projected
    on the ground plane
    reference: 10.1016/j.astropartphys.2014.04.004 """
    from radiotools.atmosphere import models as atm
    h = atm.get_vertical_height(xmax, model=model) / np.cos(zenith)
    cherenkov = get_cherenkov_angle(h, model=model)
    ra = (np.tan(zenith + cherenkov) - np.tan(zenith)) * h
    rb = np.tan(cherenkov) / np.cos(zenith) * h
    return ra, rb


def gaisser_hillas1_parametric(x, xmax, nmax=1):
    """ return one parametric form of Gaisser-Hillers function
        Reference: http://en.wikipedia.org/wiki/Gaisser%E2%80%93Hillas_function and
        Darko Veberic (2012). "Lambert W Function for Applications in Physics".
        Computer Physics Communications 183 (12): 2622–2628. arXiv:1209.0735. doi:10.1016/j.cpc.2012.07.008.
    """
    return nmax * (x / xmax) ** xmax * np.exp(xmax - x)


def gaisser_hillas(X, xmax, X0, lam, nmax=1):
    """ returns the Gaisser-Hillers function
        Reference: http://en.wikipedia.org/wiki/Gaisser%E2%80%93Hillas_function
    """
    return nmax * ((X - X0) / (xmax - X0)) ** ((xmax - X0) / lam) * np.exp((xmax - X) / lam)


def is_confined(x, y, station_positions, delta_confinement=0):
    """ returns True if core (x, y coordinate) is confined within stations
        given by 'station_positions'. If the 'delta_confinement' parameter is
        given, the stations need this minimum distance to the core for the core
        to be confined. """
    is_confined = (
        np.bool(np.sum([((x - delta_confinement) > sp_x and (y - delta_confinement) > sp_y) for (sp_x, sp_y, sp_z) in
                        station_positions])) and
        np.bool(np.sum([((x - delta_confinement) > sp_x and (y + delta_confinement) < sp_y) for (sp_x, sp_y, sp_z) in
                        station_positions])) and
        np.bool(np.sum([((x + delta_confinement) < sp_x and (y - delta_confinement) > sp_y) for (sp_x, sp_y, sp_z) in
                        station_positions])) and
        np.bool(np.sum([((x + delta_confinement) < sp_x and (y + delta_confinement) < sp_y) for (sp_x, sp_y, sp_z) in
                        station_positions])))
    return is_confined


def is_confined_weak(x, y, station_positions, delta_confinement=0):
    """ returns True if core (x, y coordinate) is confined within stations
        given by 'station_positions'. If the 'delta_confinement' parameter is
        given, the stations need this minimum distance to the core for the core
        to be confined. The criterion is weaker than in isConfined(). Here, at
        least one station has to be above, below, left or right the core position."""
    is_confined = (
        np.bool(np.sum([((x - delta_confinement) > sp_x) for (sp_x, sp_y, sp_z) in station_positions])) and
        np.bool(np.sum([((x + delta_confinement) < sp_x) for (sp_x, sp_y, sp_z) in station_positions])) and
        np.bool(np.sum([((y - delta_confinement) > sp_y) for (sp_x, sp_y, sp_z) in station_positions])) and
        np.bool(np.sum([((y + delta_confinement) < sp_y) for (sp_x, sp_y, sp_z) in station_positions])))
    return is_confined


def in_hull(p, hull):
    #     """
    #     Test if points in `p` are in `hull`
    #
    #     `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    #     `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    #     coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    #     will be computed
    #     """
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def is_confined2(x, y, station_positions, delta_confinement=0):
    from scipy.spatial import Delaunay
    s1 = station_positions + np.array([delta_confinement, 0, 0])
    s2 = station_positions + np.array([-1. * delta_confinement, 0, 0])
    s3 = station_positions + np.array([0, delta_confinement, 0])
    s4 = station_positions + np.array([0, -1. * delta_confinement, 0])
    positions = np.append(np.append(s1, s2, axis=0), np.append(s3, s4, axis=0), axis=0)
    hull = Delaunay(positions[..., 0:2])
    points = np.array([x, y]).T
    return np.array([in_hull(p, hull) for p in points], dtype=np.bool)


def get_efield_in_shower_plane(ex, ey, ez, zenith, azimuth):
    e_theta = np.cos(zenith) * np.cos(azimuth) * ex + np.cos(zenith) * np.sin(azimuth) * ey - np.sin(zenith) * ez
    e_phi = -np.sin(azimuth) * ex + np.cos(azimuth) * ey
    e_r = np.sin(zenith) * np.cos(azimuth) * ex + np.sin(zenith) * np.sin(azimuth) * ey + np.cos(zenith) * ez
    return e_theta, e_phi, e_r


def get_dirac_pulse(samples, binning=1., low_freq=30., up_freq=80.):
    """ generate dirac pulse """
    from numpy import fft

    ff = fft.rfftfreq(samples, binning * 1e-9) * 1e-6  # frequencies in MHz
    dirac = np.zeros(samples)
    dirac[samples / 2] = 1
    diracfft = fft.rfft(dirac)
    mask = (ff >= low_freq) & (ff <= up_freq)
    diracfft[~mask] = 0
    dirac = fft.irfft(diracfft)
    dirac = dirac / abs(dirac).max()
    return dirac


def rotate_vector_in_2d(v, angle):  # rotate a 2d vector counter-clockwise
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    v_rotated = np.dot(rotation_matrix, v)
    return v_rotated


def get_sd_core_error_ellipse(easting_error, northing_error, error_correlation, p):
    """
        returns semi major and semi minor axis of the confidence region with p-value p
        """
    import scipy.stats

    cov = np.array([[easting_error ** 2, easting_error * northing_error * error_correlation],
                    [easting_error * northing_error * error_correlation, northing_error ** 2]])
    chi_2 = scipy.stats.chi2.isf(1 - p, 2)
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    eigen_vector_1 = np.array([eigen_vectors[0][0], eigen_vectors[1][0], 0])
    eigen_vector_2 = np.array([eigen_vectors[0][1], eigen_vectors[1][1], 0])
    axis_1 = eigen_vector_1 * np.sqrt(chi_2 * eigen_values[0])
    axis_2 = eigen_vector_2 * np.sqrt(chi_2 * eigen_values[1])
    if eigen_values[0] >= eigen_values[1]:
        return [axis_1, axis_2]
    else:
        return [axis_2, axis_1]


def transform_error_ellipse_into_vxB_vxvxB(semi_major_axis, semi_minor_axis, zenith, azimuth):
    """
        accepts semi major and semi minor axis of an ellipse in standard auger coordinates.
        transforms the ellipse into vxB-vxvxB system and projects it onto the vxB-vxvxB_plane.
        returns the semi major and semi minor axis of the result.
    """
    import coordinatesystems

    cs = coordinatesystems.cstrafo(zenith, azimuth)
    axis_1_vxB_vxvxB = cs.transform_to_vxB_vxvxB(semi_major_axis)
    axis_2_vxB_vxvxB = cs.transform_to_vxB_vxvxB(semi_minor_axis)
    a_dot_b = axis_1_vxB_vxvxB[0] * axis_2_vxB_vxvxB[0] + axis_1_vxB_vxvxB[1] * axis_2_vxB_vxvxB[1]
    if a_dot_b == 0:
        return [[axis_1_vxB_vxvxB[0], axis_1_vxB_vxvxB[1]], [axis_2_vxB_vxvxB[0], axis_2_vxB_vxvxB[1]]]
    abs_a_squared = axis_1_vxB_vxvxB[0] ** 2 + axis_1_vxB_vxvxB[1] ** 2
    abs_b_squared = axis_2_vxB_vxvxB[0] ** 2 + axis_2_vxB_vxvxB[1] ** 2
    tan_phi_1 = -.5 * (abs_a_squared - abs_b_squared) / a_dot_b + np.sqrt(
        .25 * ((abs_a_squared - abs_b_squared) / a_dot_b) ** 2 + 1)
    tan_phi_2 = -.5 * (abs_a_squared - abs_b_squared) / a_dot_b - np.sqrt(
        .25 * ((abs_a_squared - abs_b_squared) / a_dot_b) ** 2 + 1)
    phi_1 = np.arctan(tan_phi_1)
    phi_2 = np.arctan(tan_phi_2)
    axis_1 = [axis_1_vxB_vxvxB[0] * np.cos(phi_1) + axis_2_vxB_vxvxB[0] * np.sin(phi_1),
              axis_1_vxB_vxvxB[1] * np.cos(phi_1) + axis_2_vxB_vxvxB[1] * np.sin(phi_1)]
    axis_2 = [axis_1_vxB_vxvxB[0] * np.cos(phi_2) + axis_2_vxB_vxvxB[0] * np.sin(phi_2),
              axis_1_vxB_vxvxB[1] * np.cos(phi_2) + axis_2_vxB_vxvxB[1] * np.sin(phi_2)]
    if axis_1[0] ** 2 + axis_1[1] ** 2 >= axis_2[0] ** 2 + axis_2[1] ** 2:
        return [axis_1, axis_2]
    else:
        return [axis_2, axis_1]


def is_in_quantile(center, station_position, easting_error, northing_error, error_correlation, p):
    """
        returns true if station_position is within the p-quantile around center,
        false otherwise
    """
    import scipy.stats

    cov = np.array([[easting_error ** 2, easting_error * northing_error * error_correlation],
                    [easting_error * northing_error * error_correlation, northing_error ** 2]])
    cov_inv = np.linalg.inv(cov)
    diff = [center[0] - station_position[0], center[1] - station_position[1]]
    c = diff[0] * (cov_inv[0][0] * diff[0] + cov_inv[0][1] * diff[1]) + diff[1] * (
        cov_inv[1][0] * diff[0] + cov_inv[1][1] * diff[1])
    if c <= scipy.stats.chi2.isf(1 - p, 2):
        return True
    else:
        return False


def get_ellipse_tangents_through_point(point, semi_major_axis, semi_minor_axis):
    """
        determines the points where the tangents to an ellipse with given semi major and semi minor axis that go through
        point touch the ellipse. returns none if point is inside the ellipse
    """
    theta = np.arctan2(semi_major_axis[1], semi_major_axis[0])
    point_r = rotate_vector_in_2d([point[0], point[1], 0], -theta)
    r_major = np.sqrt(semi_major_axis[0] ** 2 + semi_major_axis[1] ** 2)
    r_minor = np.sqrt(semi_minor_axis[0] ** 2 + semi_minor_axis[1] ** 2)
    divisor = point_r[1] ** 2 * r_major ** 2 + r_minor ** 2 * point_r[0] ** 2
    if (point_r[0] / r_major) ** 2 + (point_r[1] / r_minor) ** 2 <= 1:
        return None
    square_root_term = (r_minor ** 2 * point_r[0] * r_major ** 2 / divisor) ** 2 + \
                       (point_r[1] ** 2 * r_major ** 4 - r_minor ** 2 * r_major ** 4) / divisor
    tan_1_x = r_minor ** 2 * point_r[0] / divisor + np.sqrt(square_root_term)
    tan_2_x = r_minor ** 2 * point_r[0] / divisor - np.sqrt(square_root_term)
    tan_1_y = r_minor ** 2 / point_r[1] - (r_minor / r_major) ** 2 * tan_1_x * point_r[0] / point_r[1]
    tan_2_y = r_minor ** 2 / point_r[1] - (r_minor / r_major) ** 2 * tan_2_x * point_r[0] / point_r[1]
    tan_1_r = np.array([tan_1_x, tan_1_y, 0])
    tan_2_r = np.array([tan_2_x, tan_2_y, 0])
    tan_1 = rotate_vector_in_2d(tan_1_r, theta)
    tan_2 = rotate_vector_in_2d(tan_2_r, theta)
    return [[tan_1[0], tan_1[1], 0], [tan_2[0], tan_2[1], 0]]


def covariance_to_correlation(M):
    """ converts covariance matrix into correlation matrix
    """
    D = np.diagflat(np.diag(M)) ** 0.5
    Dinv = np.linalg.inv(D)
    return np.dot(Dinv, np.dot(M, Dinv))


def get_normalized_xcorr(trace1, trace2, mode='full'):
    return correlate(trace1, trace2, mode=mode, method='auto') / (np.sum(trace1 ** 2) * np.sum(trace2 ** 2)) ** 0.5


def linreg(x, y):
    '''
    Linear regression: returns the offset a and slope b for the function y_lin(x) = a + b*x
    that approximates the distribtion y(x) the best (sum of squares of residuals is minimized).

    input:
        x: array-like, values where y-values are valid
        y: array-like, must have same length as x, values y(x)

    output:
        a = offset of linear function resulting from regression
        b = slope of linear function resulting from regression
    '''
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b = SS_xy / SS_xx  # slope
    a = m_y - b * m_x  # zero-offset

    return(a, b)


def pretty_time_delta(seconds):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%dd%dh%dm%ds' % (days, hours, minutes, seconds)
    elif hours > 0:
        return '%dh%dm%ds' % (hours, minutes, seconds)
    elif minutes > 0:
        return '%dm%ds' % (minutes, seconds)
    else:
        return '%ds' % (seconds,)


def FC_limits(counts):
    """
    returns the 68%CL Feldman-Cousins limits for 0 background.
    
    Parameters
    ----------
    counts: float
        the number of counts/events
        
    Returns tuple of floats
        lower_bound, upper bound
    """

    from scipy.interpolate import interp1d

    count_list = np.arange(0, 21)
    lower_limits = [0.00,
                    0.37,
                    0.74,
                    1.10,
                    2.34,
                    2.75,
                    3.82,
                    4.25,
                    5.30,
                    6.33,
                    6.78,
                    7.81,
                    8.83,
                    9.28,
                    10.30,
                    11.32,
                    12.33,
                    12.79,
                    13.81,
                    14.82,
                    15.83]
    upper_limits = [1.29,
                    2.75,
                    4.25,
                    5.30,
                    6.78,
                    7, 81,
                    9.28,
                    10.30,
                    11.32,
                    12.79,
                    13.81,
                    14.82,
                    16.29,
                    17.30,
                    18.32,
                    19.32,
                    20.80,
                    21.81,
                    22.82,
                    25.30]

    if counts > count_list[-1]:

        return (counts - np.sqrt(counts), counts + np.sqrt(counts))

    elif counts < 0:

        return (0.00, 1.29)

    low_interp = interp1d(count_list, lower_limits)
    up_interp = interp1d(count_list, upper_limits)

    return (low_interp(counts), up_interp(counts))

