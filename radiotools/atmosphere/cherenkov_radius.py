# functions to estimate calculate the cherenkov radius of the radio emission of an air shower
# code written by Felix Schlueter for the RadioAnalysis framework
# added to radiotools by Lukas Guelzow

import numpy as np
import warnings

from radiotools.atmosphere import models as atm


def get_cherenkov_radius_from_depth(zenith, depth, obs_level, n0, model=None, at=None):
    """ Calculates the radius of the (Cherenkov) cone with an apex at a given depth along a 
        shower axis with a given zenith angle. The open angle of the cone equals the 
        Cherenkov angle for a value of the refractive index at this position.

    Paramter:

    zenith : double
        Zenith angle (in radian) under which a shower is observed 
    
    depth : double
        Slant depth (in g/cm^2), i.e., shower maximum of the observed shower
    
    obs_level : double
        Altitude (in meter) of the plane at which the shower is observed 

    n0 : double
        Refractive index at sea level (!= obs_level)

    model : int
        Model index for the atmospheric (density) profile model. Needed when no "at" is given

    at : radiotools.atmosphere.models.Atmosphere
        Atmospheric (density) profile model. Provides the density profile of the atmosphere in the typical 5-layer param.

    Return : cherenkov Radius

    """
    if at is None:
        at = atm.Atmosphere(model=model)

    d = at.get_distance_xmax_geometric(zenith, depth, obs_level)
    return get_cherenkov_radius_from_distance(zenith, d, obs_level, n0, at.model)


def get_cherenkov_radius_from_height(zenith, height, obs_level, n0, model):
    """ Calculates the radius of the (Cherenkov) cone with an apex at a given height above sea level on a
        shower axis with a given zenith angle. The open angle of the cone equals the 
        Cherenkov angle for a value of the refractive index at this position.

    Paramter:

    zenith : double
        Zenith angle (in radian) under which a shower is observed 
    
    height : double
        Height above sea level (in m) of the apex, i.e., shower maximum of the observed shower
    
    obs_level : double
        Altitude (in meter) of the plane at which the shower is observed 

    n0 : double
        Refractive index at sea level (!= obs_level)

    model : int
        Model index for the atmospheric (density) profile model.

    Return : cherenkov Radius

    """

    angle = get_cherenkov_angle(height, n0, model)
    dmax = atm.get_distance_for_height_above_ground(
        height - obs_level, zenith, observation_level=obs_level)
    return cherenkov_radius(angle, dmax)


def get_cherenkov_radius_from_distance(zenith, d, obs_level, n0, model):
    """ Calculates the radius of the (Cherenkov) cone with an apex at a given distance from ground 
        along the shower axis with a given zenith angle. The open angle of the cone equals the 
        Cherenkov angle for a value of the refractive index at this position.

    Paramter:

    zenith : double
        Zenith angle (in radian) under which a shower is observed 
    
    d : double
        Distance from ground to the apex, i.e., shower maximum of the observed shower along the shower axis (in m)
    
    obs_level : double
        Altitude (in meter) of the plane at which the shower is observed 

    n0 : double
        Refractive index at sea level (!= obs_level)

    model : int
        Model index for the atmospheric (density) profile model.

    Return : cherenkov Radius

    """
    height = atm.get_height_above_ground(
        d, zenith, observation_level=obs_level) + obs_level
    angle = get_cherenkov_angle(height, n0, model)
    return cherenkov_radius(angle, d)


def get_cherenkov_angle(height, n0, model):
    """ Return cherenkov angle for given height above sea level, 
        refractive index at sea level and atmospheric model. 

    Paramter:

    height : double
        Height above sea level (in m)
    
    n0 : double
        Refractive index at sea level (!= obs_level)

    model : int
        Model index for the atmospheric (density) profile model.

    Return : cherenkov angle

    """
    n = atm.get_n(height, n0=n0, model=model)
    return cherenkov_angle(n)


def cherenkov_angle(n):
    """ Return cherenkov angle for given refractive index.

    Paramter:

    n : double
        Refractive index

    Return : cherenkov angle

    """
    return np.arccos(1 / n)


def cherenkov_radius(angle, d):
    """ Return (cherenkov) radius

    Paramters
    ---------

    angle : double
        (Opening) angle of the cone (in rad)

    d : double
        Heigth of the cone (typically called distance, in meter)

    Returns
    -------
    
    radius : double
        (Cherenkov) radius

    """
    return np.tan(angle) * d