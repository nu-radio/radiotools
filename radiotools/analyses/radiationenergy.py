import functools

import numpy as np
from radiotools.atmosphere import models as atm

# see Glaser et al., JCAP 09(2016)024 for the derivation of the formulas

average_xmax = 669.40191244545326  # 1 EeV, 50% proton, 50% iron composition
average_zenith = np.deg2rad(45)


@functools.lru_cache(maxsize=16)
def get_average_density(model=1, zenith=average_zenith, xmax=average_xmax):
    """ get average density

    Parameters
    ----------
    model : int
        atmospheric model

    Returns
    -------
    float
        air density for a mean xmax and zenith angle (default: 1, US standard after Linsley)

    """
    atmc = atm.Atmosphere(model=model)
    return atmc.get_density(zenith, xmax) * 1e-3  # in kg/m^3


def get_clipping(dxmax):
    """ get clipping correction

    Parameters
    ----------
    dxmax : float
        distance to shower maximum in g/cm^2

    Returns
    -------
    float
        fraction of radiation energy that is radiated in the atmosphere

    """
    return 1 - np.exp(-8.7 * (dxmax * 1e-3 + 0.29) ** 1.89)


def get_a(rho):
    """ get relative charge excess strength

    Parameters
    ----------
    rho : float
        density at shower maximum in kg/m^3

    Returns
    -------
    float
        relative charge excess strength a
    """
    return -0.23604683 + 0.43426141 * np.exp(1.11141046 * (rho - get_average_density()))


def get_a_zenith(zenith):
    """ get relative charge excess strength wo Xmax information

    Parameters
    ----------
    zentith : float
        zenith angle in rad according to radiotools default coordinate system

    Returns
    --------
    float
        relative charge excess strength a
    """
    rho = atmc.get_density(zenith, average_xmax) * 1e-3
    return -0.24304254 + 0.4511355 * np.exp(1.1380946 * (rho - get_average_density()))


def get_S(Erad, sinalpha, density, p0=0.250524463912, p1=-2.95290494,
          b_scale=1., b=1.8):
    """ get corrected radiation energy (S_RD)

    Parameters
    ----------
    Erad : float
        radiation energy (in eV)
    sinalpha: float
        sine of angle between shower axis and geomagnetic field
    density : float
        density at shower maximum in kg/m^3

    Returns
    --------
    float:
        corrected radiation energy (in eV)
    """
    a = get_a(density) * b_scale ** (-0.5 * b)
    return Erad / (a ** 2 + (1 - a ** 2) * sinalpha ** 2 * b_scale ** b) / \
            (1 - p0 + p0 * np.exp(p1 * (density - get_average_density()))) ** 2


def get_S_zenith(erad, sinalpha, zeniths, b_scale=1., p0=0.239, p1=-3.13):
    """ get corrected radiation energy (S_RD) wo xmax information

    Parameters
    ----------
    Erad : float
        radiation energy (in eV)
    sinalpha: float
        sine of angle between shower axis and geomagnetic field
    density : float
        density at shower maximum in kg/m^3

    Returns
    --------
    float:
        corrected radiation energy wo Xmax information (in eV)
    """
    rho = atmc.get_density(zeniths, average_xmax) * 1e-3
    return get_S(erad, sinalpha, rho, p0=p0, p1=p1, b_scale=b_scale)


def get_radiation_energy(Srd, sinalpha, density, p0=0.250524463912,
                         p1=-2.95290494, b_scale=1., b=1.8):
    """ get radiation energy (S_RD)

    Parameters
    ----------
    Srd : float
        corrected radiation energy (in eV)
    sinalpha: float
        sine of angle between shower axis and geomagnetic field
    density : float
        density at shower maximum in kg/m^3

    Returns
    --------
    float:
        radiation energy (in eV)
    """
    a = get_a(density) * b_scale ** (-0.5 * b)
    return Srd * (a ** 2 + (1 - a ** 2) * sinalpha ** 2 * b_scale ** b) * \
            (1 - p0 + p0 * np.exp(p1 * (density - get_average_density()))) ** 2
