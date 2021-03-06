# Python 2 and 3: backward-compatible
from __future__ import absolute_import, division, print_function  # , unicode_literals
from past.builtins import xrange

from scipy import optimize, interpolate, integrate

import numpy as np
import os
import sys

default_curved = True
default_model = 17

r_e = 6.371 * 1e6  # radius of Earth

"""
    All functions use "grams" and "meters", only the functions that receive and
    return "atmospheric depth" use the unit "g/cm^2"

    atmospheric density models as used in CORSIKA. The parameters are documented in the CORSIKA manual
    the parameters for the Auger atmospheres are documented in detail in GAP2011-133
    The May and October atmospheres describe the annual average best.
"""
h_max = 112829.2  # height above sea level where the mass overburden vanishes
atm_models = {  # US standard after Linsley
              1: {'a': 1e4 * np.array([-186.555305, -94.919, 0.61289, 0., 0.01128292]),
                  'b': 1e4 * np.array([1222.6562, 1144.9069, 1305.5948, 540.1778, 1.]),
                  'c': 1e-2 * np.array([994186.38, 878153.55, 636143.04, 772170.16, 1.e9]),
                  'h': 1e3 * np.array([4., 10., 40., 100.])
                  },
              # southpole January after Lipari
              15: {'a': 1e4 * np.array([-113.139, -79.0635, -54.3888, 0., 0.00421033]),
                   'b': 1e4 * np.array([1133.1, 1101.2, 1085., 1098., 1.]),
                   'c': 1e-2 * np.array([861730., 826340., 790950., 682800., 2.6798156e9]),
                   'h': 1e3 * np.array([2.67, 5.33, 8., 100.])
                  },
              # US standard after Keilhauer
              17: {'a': 1e4 * np.array([-149.801663, -57.932486, 0.63631894, 4.35453690e-4, 0.01128292]),
                   'b': 1e4 * np.array([1183.6071, 1143.0425, 1322.9748, 655.67307, 1.]),
                   'c': 1e-2 * np.array([954248.34, 800005.34, 629568.93, 737521.77, 1.e9]),
                   'h': 1e3 * np.array([7., 11.4, 37., 100.])
                  },
              # Malargue January
              18: {'a': 1e4 * np.array([-136.72575606, -31.636643044, 1.8890234035, 3.9201867984e-4, 0.01128292]),
                   'b': 1e4 * np.array([1174.8298334, 1204.8233453, 1637.7703583, 735.96095023, 1.]),
                   'c': 1e-2 * np.array([982815.95248, 754029.87759, 594416.83822, 733974.36972, 1e9]),
                   'h': 1e3 * np.array([9.4, 15.3, 31.6, 100.])
                  },
              # Malargue February
              19: {'a': 1e4 * np.array([-137.25655862, -31.793978896, 2.0616227547, 4.1243062289e-4, 0.01128292]),
                   'b': 1e4 * np.array([1176.0907565, 1197.8951104, 1646.4616955, 755.18728657, 1.]),
                   'c': 1e-2 * np.array([981369.6125, 756657.65383, 592969.89671, 731345.88332, 1.e9]),
                   'h': 1e3 * np.array([9.2, 15.4, 31., 100.])
                  },
              # Malargue March
              20: {'a': 1e4 * np.array([-132.36885162, -29.077046629, 2.090501509, 4.3534337925e-4, 0.01128292]),
                   'b': 1e4 * np.array([1172.6227784, 1215.3964677, 1617.0099282, 769.51991638, 1.]),
                   'c': 1e-2 * np.array([972654.0563, 742769.2171, 595342.19851, 728921.61954, 1.e9]),
                   'h': 1e3 * np.array([9.6, 15.2, 30.7, 100.])
                  },
              # Malargue April
              21: {'a': 1e4 * np.array([-129.9930412, -21.847248438, 1.5211136484, 3.9559055121e-4, 0.01128292]),
                   'b': 1e4 * np.array([1172.3291878, 1250.2922774, 1542.6248413, 713.1008285, 1.]),
                   'c': 1e-2 * np.array([962396.5521, 711452.06673, 603480.61835, 735460.83741, 1.e9]),
                   'h': 1e3 * np.array([10., 14.9, 32.6, 100.])
                  },
              # Malargue May
              22: {'a': 1e4 * np.array([-125.11468467, -14.591235621, 0.93641128677, 3.2475590985e-4, 0.01128292]),
                   'b': 1e4 * np.array([1169.9511302, 1277.6768488, 1493.5303781, 617.9660747, 1.]),
                   'c': 1e-2 * np.array([947742.88769, 685089.57509, 609640.01932, 747555.95526, 1.e9]),
                   'h': 1e3 * np.array([10.2, 15.1, 35.9, 100.])
                  },
              # Malargue June
              23: {'a': 1e4 * np.array([-126.17178851, -7.7289852811, 0.81676828638, 3.1947676891e-4, 0.01128292]),
                   'b': 1e4 * np.array([1171.0916276, 1295.3516434, 1455.3009344, 595.11713507, 1.]),
                   'c': 1e-2 * np.array([940102.98842, 661697.57543, 612702.0632, 749976.26832, 1.e9]),
                   'h': 1e3 * np.array([10.1, 16., 36.7, 100.])
                  },
              # Malargue July
              24: {'a': 1e4 * np.array([-126.17216789, -8.6182537514, 0.74177836911, 2.9350702097e-4, 0.01128292]),
                   'b': 1e4 * np.array([1172.7340688, 1258.9180079, 1450.0537141, 583.07727715, 1.]),
                   'c': 1e-2 * np.array([934649.58886, 672975.82513, 614888.52458, 752631.28536, 1.e9]),
                   'h': 1e3 * np.array([9.6, 16.5, 37.4, 100.])
                  },
              # Malargue August
              25: {'a': 1e4 * np.array([-123.27936204, -10.051493041, 0.84187346153, 3.2422546759e-4, 0.01128292]),
                   'b': 1e4 * np.array([1169.763036, 1251.0219808, 1436.6499372, 627.42169844, 1.]),
                   'c': 1e-2 * np.array([931569.97625, 678861.75136, 617363.34491, 746739.16141, 1.e9]),
                   'h': 1e3 * np.array([9.6, 15.9, 36.3, 100.])
                  },
              # Malargue September
              26: {'a': 1e4 * np.array([-126.94494665, -9.5556536981, 0.74939405052, 2.9823116961e-4, 0.01128292]),
                   'b': 1e4 * np.array([1174.8676453, 1251.5588529, 1440.8257549, 606.31473165, 1.]),
                   'c': 1e-2 * np.array([936953.91919, 678906.60516, 618132.60561, 750154.67709, 1.e9]),
                   'h': 1e3 * np.array([9.5, 15.9, 36.3, 100.])
                  },
              # Malargue October
              27: {'a': 1e4 * np.array([-133.13151125, -13.973209265, 0.8378263431, 3.111742176e-4, 0.01128292]),
                   'b': 1e4 * np.array([1176.9833473, 1244.234531, 1464.0120855, 622.11207419, 1.]),
                   'c': 1e-2 * np.array([954151.404, 692708.89816, 615439.43936, 747969.08133, 1.e9]),
                   'h': 1e3 * np.array([9.5, 15.5, 36.5, 100.])
                  },
              # Malargue November
              28: {'a': 1e4 * np.array([-134.72208165, -18.172382908, 1.1159806845, 3.5217025515e-4, 0.01128292]),
                   'b': 1e4 * np.array([1175.7737972, 1238.9538504, 1505.1614366, 670.64752105, 1.]),
                   'c': 1e-2 * np.array([964877.07766, 706199.57502, 610242.24564, 741412.74548, 1.e9]),
                   'h': 1e3 * np.array([9.6, 15.3, 34.6, 100.])
                  },
              # Malargue December
              29: {'a': 1e4 * np.array([-135.40825209, -22.830409026, 1.4223453493, 3.7512921774e-4, 0.01128292]),
                   'b': 1e4 * np.array([1174.644971, 1227.2753683, 1585.7130562, 691.23389637, 1.]),
                   'c': 1e-2 * np.array([973884.44361, 723759.74682, 600308.13983, 738390.20525, 1.e9]),
                   'h': 1e3 * np.array([9.6, 15.6, 33.3, 100.])
                  },
              # South Pole April (De Ridder)
              33: {'a': 1e4 * np.array([-69.7259, -2.79781, 0.262692, -.0000841695, 0.00207722]),
                   'b': 1e4 * np.array([1111.70, 1128.64, 1413.98, 587.688, 1]),
                   'c': 1e-2 * np.array([766099., 641716., 588082., 693300., 5430320300]),
                   'h': 1e3 * np.array([7.6, 22.0, 40.4, 100.])
                  }
             }


def get_auger_monthly_model(month):
    """ Helper function to get the correct model number for monthly Auger atmospheres """
    return month + 17


def get_height_above_ground(d, zenith, observation_level=0):
    """ returns the perpendicular height above ground for a distance d from ground at a given zenith angle """
    r = r_e + observation_level
    x = d * np.sin(zenith)
    y = d * np.cos(zenith) + r
    return (x ** 2 + y ** 2) ** 0.5 - r


def get_distance_for_height_above_ground(h, zenith, observation_level=0):
    """ inverse of get_height_above_ground() """
    r = r_e + observation_level
    return (h ** 2 + 2 * r * h + r ** 2 * np.cos(zenith) ** 2) ** 0.5 - r * np.cos(zenith)


def get_vertical_height(slant_depth, model=default_model):
    """ input: atmosphere (slant depth) above in g/cm^2 [e.g. Xmax]
        output: height in m """
    return _get_vertical_height(slant_depth * 1e4, model=model)


def _get_vertical_height(at, model=default_model):
    """ get vertical height from atmosphere, i.e., mass overburden """
    if not hasattr(at, "__len__"):
        T = _get_i_at(at, model=model)
    else:
        T = np.zeros(len(at))
        for i, at in enumerate(at):
            T[i] = _get_i_at(at, model=model)
    return T


def _get_i_at(at, model=default_model):
    """ get vertical height from atmosphere, i.e., mass overburden for different layer """
    a = atm_models[model]['a']
    b = atm_models[model]['b']
    c = atm_models[model]['c']
    layers = atm_models[model]['h']

    if at > _get_atmosphere(layers[0], model=model):
        i = 0
    elif at > _get_atmosphere(layers[1], model=model):
        i = 1
    elif at > _get_atmosphere(layers[2], model=model):
        i = 2
    elif at > _get_atmosphere(layers[3], model=model):
        i = 3
    else:
        i = 4

    if i == 4:
        h = -1. * c[i] * (at - a[i]) / b[i]
    else:
        h = -1. * c[i] * np.log((at - a[i]) / b[i])

    return h


def get_atmosphere(h, model=default_model):
    """ returns the (vertical) amount of atmosphere above the height h above see level
    in units of g/cm^2
    input: height above sea level in meter"""
    if hasattr(h, "__len__"):
        return _get_atmosphere(h, model=model) * 1e-4
    else:
        return _get_atmosphere_float(h, model=model) * 1e-4


def _get_atmosphere(h, model=default_model):
    a = atm_models[model]['a']
    b = atm_models[model]['b']
    c = atm_models[model]['c']
    layers = atm_models[model]['h']
    y = np.where(h < layers[0], a[0] + b[0] * np.exp(-1 * h / c[0]), a[1] + b[1] * np.exp(-1 * h / c[1]))
    y = np.where(h < layers[1], y, a[2] + b[2] * np.exp(-1 * h / c[2]))
    y = np.where(h < layers[2], y, a[3] + b[3] * np.exp(-1 * h / c[3]))
    y = np.where(h < layers[3], y, a[4] - b[4] * h / c[4])
    y = np.where(h < h_max, y, 0)
    return y


def _get_atmosphere_float(h, model=default_model):
    if h > h_max:
        return 0

    a = atm_models[model]['a']
    b = atm_models[model]['b']
    c = atm_models[model]['c']
    layers = atm_models[model]['h']

    idx = np.argmin(np.abs(layers - h))
    if h > layers[idx]:
        idx += 1

    if idx == 4:
        return a[4] - b[4] * h / c[4]
    else:
        return a[idx] + b[idx] * np.exp(-1 * h / c[idx])


def get_density(h, allow_negative_heights=True, model=default_model):
    """ returns the atmospheric density [g/m^3] for the height h above see level"""
    b = atm_models[model]['b']
    c = atm_models[model]['c']
    layers = atm_models[model]['h']

    if hasattr(h, "__len__"):
        y = np.zeros_like(h, dtype=np.float)
        if not allow_negative_heights:
            y *= np.nan  # set all requested densities for h < 0 to nan
            y = np.where(h < 0, y, b[0] * np.exp(-1 * h / c[0]) / c[0])
        else:
            y = b[0] * np.exp(-1 * h / c[0]) / c[0]

        y = np.where(h < layers[0], y, b[1] * np.exp(-1 * h / c[1]) / c[1])
        y = np.where(h < layers[1], y, b[2] * np.exp(-1 * h / c[2]) / c[2])
        y = np.where(h < layers[2], y, b[3] * np.exp(-1 * h / c[3]) / c[3])
        y = np.where(h < layers[3], y, b[4] / c[4])
        y = np.where(h < h_max, y, 0)

        return y
    else:
        if h < 0 and not allow_negative_heights:
            return np.nan

        b = atm_models[model]['b']
        c = atm_models[model]['c']
        layers = atm_models[model]['h']

        idx = np.argmin(np.abs(layers - h))
        if h > layers[idx]:
            idx += 1

        if idx == 4:
            return b[4] / c[4]
        else:
            return b[idx] * np.exp(-1 * h / c[idx]) / c[idx]


def get_density_for_distance(d, zenith, observation_level=0, model=default_model):
    """ returns the atmospheric density [g/m^3] for a given distance and zenith angle assuming a curved atmosphere"""
    h = get_height_above_ground(d, zenith, observation_level=observation_level) + observation_level
    return get_density(h, model=model)


def get_density_from_barometric_formula(hh):
    """ returns the atmospheric density [g/m^3] for the height h abolve see level
    according to https://en.wikipedia.org/wiki/Barometric_formula"""
    if isinstance(hh, float):
        hh = np.array([hh])
    R = 8.31432  # universal gas constant for air: 8.31432 N m/(mol K)
    g0 = 9.80665  # gravitational acceleration (9.80665 m/s2)
    M = 0.0289644  # molar mass of Earth's air (0.0289644 kg/mol)
    rhob = [1.2250, 0.36391, 0.08803, 0.01322, 0.00143, 0.00086, 0.000064]
    Tb = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65]
    Lb = [-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002]
    hb = [0, 11000, 20000, 32000, 47000, 51000, 71000]

    def rho1(h, i):  # for Lb != 0
        return rhob[i] * (Tb[i] / (Tb[i] + Lb[i] * (h - hb[i]))) ** (1 + (g0 * M) / (R * Lb[i]))

    def rho2(h, i):  # for Lb == 0
        return rhob[i] * np.exp(-g0 * M * (h - hb[i]) / (R * Tb[i]))

    densities = np.zeros_like(hh)
    for i, h in enumerate(hh):
        if (h < 0):
            densities[i] = np.nan
        elif(h > 86000):
            densities[i] = 0
        else:
            t = h - hb
            index = np.argmin(t[t >= 0])
            if Lb[index] == 0:
                densities[i] = rho2(h, index)
            else:
                densities[i] = rho1(h, index)

    return densities * 1e3


def get_atmosphere_upper_limit(model=default_model):
        """ returns the altitude where the mass overburden vanishes """
        from functools import partial
        return optimize.newton(partial(_get_atmosphere, model=model), x0=112.8e3)


def get_n(h, n0=(1 + 2.92e-4), allow_negative_heights=False, model=default_model):
    return (n0 - 1) * get_density(h, allow_negative_heights=allow_negative_heights,
                                  model=model) / get_density(0, model=model) + 1


class Atmosphere():

    def __init__(self, model=17, n_taylor=5, curved=True, number_of_zeniths=201, zenith_numeric=np.deg2rad(80)):
        print("model is ", model)
        self.model = model
        self.curved = curved
        self.n_taylor = n_taylor
        self.__zenith_numeric = zenith_numeric
        self.b = atm_models[model]['b']
        self.c = atm_models[model]['c']
        self.number_of_zeniths = number_of_zeniths
        hh = atm_models[model]['h']
        self.h = np.append([0], hh)

        if curved:
            folder = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(folder, "constants_%02i_%i.npz" % (self.model, n_taylor))
            print("searching constants at ", filename)
            if os.path.exists(filename):
                print("reading constants from ", filename)

                with np.load(filename, "rb") as fin:
                    self.a = fin["a"]

                if(len(self.a) != self.number_of_zeniths):
                    os.remove(filename)
                    print("constants outdated, please rerun to calculate new constants")
                    sys.exit(0)

                zeniths = np.arccos(np.linspace(0, 1, self.number_of_zeniths))
                mask = zeniths < np.deg2rad(90)
                self.a_funcs = [interpolate.interp1d(zeniths[mask], self.a[..., i][mask], kind='cubic') for i in xrange(5)]

            else:
                self.a = self.__calculate_a()
                np.savez(filename, a=self.a)
                print("all constants calculated, exiting now... please rerun your analysis")
                sys.exit(0)

    def __calculate_a(self,):
        zeniths = np.arccos(np.linspace(0, 1, self.number_of_zeniths))
        a = np.zeros((self.number_of_zeniths, 5))
        self.curved = True
        self.__zenith_numeric = 0
        for iZ, z in enumerate(zeniths):
            print("calculating constants for %.02f deg zenith angle (iZ = %i, nT = %i)..." % (np.rad2deg(z), iZ, self.n_taylor))
            a[iZ] = self.__get_a(z)
            print("\t... a  = ", a[iZ], " iZ = ", iZ)

        return a

    def __get_a(self, zenith):
        a = np.zeros(5)
        b = self.b
        c = self.c
        h = self.h
        a[0] = self._get_atmosphere_numeric([zenith], h_low=h[0]) - b[0] * self._get_dldh(h[0], zenith, 0)
        a[1] = self._get_atmosphere_numeric([zenith], h_low=h[1]) - b[1] * np.exp(-h[1] / c[1]) * self._get_dldh(h[1], zenith, 1)
        a[2] = self._get_atmosphere_numeric([zenith], h_low=h[2]) - b[2] * np.exp(-h[2] / c[2]) * self._get_dldh(h[2], zenith, 2)
        a[3] = self._get_atmosphere_numeric([zenith], h_low=h[3]) - b[3] * np.exp(-h[3] / c[3]) * self._get_dldh(h[3], zenith, 3)
        a[4] = self._get_atmosphere_numeric([zenith], h_low=h[4]) + b[4] * h[4] / c[4] * self._get_dldh(h[4], zenith, 4)
        return a

    def _get_dldh(self, h, zenith, iH):
        if iH < 4:
            c = self.c[iH]
            st = np.sin(zenith)
            ct = np.cos(zenith)
            dldh = np.ones_like(zenith) / ct
            if self.n_taylor >= 1:
                dldh += -(st ** 2 / ct ** 3 * (c + h) / r_e)
            if self.n_taylor >= 2:
                tmp = 3. / 2. * st ** 2 * (2 * c ** 2 + 2 * c * h + h ** 2) / (r_e ** 2 * ct ** 5)
                dldh += tmp
            if self.n_taylor >= 3:
                t1 = 6 * c ** 3 + 6 * c ** 2 * h + 3 * c * h ** 2 + h ** 3
                tmp = st ** 2 / (2 * r_e ** 3 * ct ** 7) * (ct ** 2 - 5) * t1
                dldh += tmp
            if self.n_taylor >= 4:
                t1 = 24 * c ** 4 + 24 * c ** 3 * h + 12 * c ** 2 * h ** 2 + 4 * c * h ** 3 + h ** 4
                tmp = -1. * st ** 2 * 5. / (8. * r_e ** 4 * ct ** 9) * (3 * ct ** 2 - 7) * t1
                dldh += tmp
            if self.n_taylor >= 5:
                t1 = 120 * c ** 5 + 120 * c ** 4 * h + 60 * c ** 3 * h ** 2 + 20 * c ** 2 * h ** 3 + 5 * c * h ** 4 + h ** 5
                tmp = st ** 2 * (ct ** 4 - 14. * ct ** 2 + 21.) * (-3. / 8.) / (r_e ** 5 * ct ** 11) * t1
                dldh += tmp
        elif(iH == 4):
            c = self.c[iH]
            st = np.sin(zenith)
            ct = np.cos(zenith)
            dldh = np.ones_like(zenith) / ct
            if self.n_taylor >= 1:
                dldh += (-0.5 * st ** 2 / ct ** 3 * h / r_e)
            if self.n_taylor >= 2:
                dldh += 0.5 * st ** 2 / ct ** 5 * (h / r_e) ** 2
            if self.n_taylor >= 3:
                dldh += 1. / 8. * (st ** 2 * (ct ** 2 - 5) * h ** 3) / (r_e ** 3 * ct ** 7)
            if self.n_taylor >= 4:
                tmp2 = -1. / 8. * st ** 2 * (3 * ct ** 2 - 7) * (h / r_e) ** 4 / ct ** 9
                dldh += tmp2
            if self.n_taylor >= 5:
                tmp2 = -1. / 16. * st ** 2 * (ct ** 4 - 14 * ct ** 2 + 21) * (h / r_e) ** 5 / ct ** 11
                dldh += tmp2
        else:
            print("ERROR, height index our of bounds")
            sys.exit(-1)

        return dldh

    def __get_method_mask(self, zenith):
        if not self.curved:
            return np.ones_like(zenith, dtype=np.bool), np.zeros_like(zenith, dtype=np.bool), np.zeros_like(zenith, dtype=np.bool)
        mask_flat = np.zeros_like(zenith, dtype=np.bool)
        mask_taylor = zenith < self.__zenith_numeric
        mask_numeric = zenith >= self.__zenith_numeric
        return mask_flat, mask_taylor, mask_numeric

    def __get_height_masks(self, hh):
        # mask0 = (hh >= 0) & (hh < atm_models[self.model]['h'][0])
        mask0 = (hh < atm_models[self.model]['h'][0])
        mask1 = (hh >= atm_models[self.model]['h'][0]) & (hh < atm_models[self.model]['h'][1])
        mask2 = (hh >= atm_models[self.model]['h'][1]) & (hh < atm_models[self.model]['h'][2])
        mask3 = (hh >= atm_models[self.model]['h'][2]) & (hh < atm_models[self.model]['h'][3])
        mask4 = (hh >= atm_models[self.model]['h'][3]) & (hh < h_max)
        mask5 = hh >= h_max
        return np.array([mask0, mask1, mask2, mask3, mask4, mask5])

    def __get_X_masks(self, X, zenith):
        mask0 = X > self._get_atmosphere(zenith, atm_models[self.model]['h'][0])
        mask1 = (X <= self._get_atmosphere(zenith, atm_models[self.model]['h'][0])) & \
                (X > self._get_atmosphere(zenith, atm_models[self.model]['h'][1]))
        mask2 = (X <= self._get_atmosphere(zenith, atm_models[self.model]['h'][1])) & \
                (X > self._get_atmosphere(zenith, atm_models[self.model]['h'][2]))
        mask3 = (X <= self._get_atmosphere(zenith, atm_models[self.model]['h'][2])) & \
                (X > self._get_atmosphere(zenith, atm_models[self.model]['h'][3]))
        mask4 = (X <= self._get_atmosphere(zenith, atm_models[self.model]['h'][3])) & \
                (X > self._get_atmosphere(zenith, h_max))
        mask5 = X <= 0
        return np.array([mask0, mask1, mask2, mask3, mask4, mask5])

    def __get_arguments(self, mask, *args):
        tmp = []
        ones = np.ones(np.array(mask).size)
        for a in args:
            if np.shape(a) == ():
                tmp.append(a * ones)
            else:
                tmp.append(a[mask])
        return tmp

    def get_atmosphere(self, zenith, h_low=0., h_up=np.infty, observation_level=0):
        """ returns the atmosphere for an air shower with given zenith angle (in g/cm^2) """
        return self._get_atmosphere(zenith, h_low=h_low, h_up=h_up, observation_level=observation_level) * 1e-4

    def _get_atmosphere(self, zenith, h_low=0., h_up=np.infty, observation_level=0):
        mask_flat, mask_taylor, mask_numeric = self.__get_method_mask(zenith)
        mask_finite = np.array((h_up * np.ones_like(zenith)) < h_max)
        is_mask_finite = np.sum(mask_finite)

        tmp = np.zeros_like(zenith)
        if np.sum(mask_numeric):
            # print("getting numeric")
            tmp[mask_numeric] = self._get_atmosphere_numeric(*self.__get_arguments(mask_numeric, zenith, h_low, h_up, observation_level))

        if np.sum(mask_taylor):
            # print("getting taylor")
            tmp[mask_taylor] = self._get_atmosphere_taylor(*self.__get_arguments(mask_taylor, zenith, h_low))
            if(is_mask_finite):
                # print("\t is finite")
                mask_tmp = np.squeeze(mask_finite[mask_taylor])
                tmp2 = self._get_atmosphere_taylor(*self.__get_arguments(mask_taylor, zenith, h_up))
                tmp[mask_tmp] = tmp[mask_tmp] - np.array(tmp2)

        if np.sum(mask_flat):
            # print("getting flat atm")
            tmp[mask_flat] = self._get_atmosphere_flat(*self.__get_arguments(mask_flat, zenith, h_low))
            if(is_mask_finite):
                mask_tmp = np.squeeze(mask_finite[mask_flat])
                tmp2 = self._get_atmosphere_flat(*self.__get_arguments(mask_flat, zenith, h_up))
                tmp[mask_tmp] = tmp[mask_tmp] - np.array(tmp2)

        return tmp

    def __get_a_from_interpolation(self, zeniths):
        a = np.zeros((len(zeniths), 5))
        for i in xrange(5):
            a[..., i] = self.a_funcs[i](zeniths)
        return a

    def plot_a(self):
        import matplotlib.pyplot as plt
        zeniths = np.arccos(np.linspace(0, 1, self.number_of_zeniths))
        mask = zeniths < np.deg2rad(83)
        fig, ax = plt.subplots(1, 1)
        x = np.rad2deg(zeniths[mask])
        # mask2 = np.array([0, 1] * (np.sum(mask) / 2), dtype=np.bool)

        ax.plot(x, self.a[..., 0][mask], ".", label="a0")
        ax.plot(x, self.a[..., 1][mask], ".", label="a1")
        ax.plot(x, self.a[..., 2][mask], ".", label="a2")
        ax.plot(x, self.a[..., 3][mask], ".", label="a3")
        ax.plot(x, self.a[..., 4][mask], ".", label="a4")
        ax.set_xlim(0, 84)
        ax.legend()
        plt.tight_layout()

        for i in xrange(5):
            y = self.a[..., i][mask]
            f2 = interpolate.interp1d(x, y, kind='cubic')
            xxx = np.linspace(0, 81, 100)
            ax.plot(xxx, f2(xxx), "-")

        ax.set_ylim(-1e8, 1e8)
        plt.show()

    def _get_atmosphere_taylor(self, zenith, h_low=0.):
        b = self.b
        c = self.c
        a = self.__get_a_from_interpolation(zenith)

        masks = self.__get_height_masks(h_low)
        tmp = np.zeros_like(zenith)
        for iH, mask in enumerate(masks):
            if(np.sum(mask)):
                if(np.array(h_low).size == 1):
                    h = h_low
                else:
                    h = h_low[mask]

                if iH < 4:
                    dldh = self._get_dldh(h, zenith[mask], iH)
                    tmp[mask] = np.array([a[..., iH][mask] + b[iH] * np.exp(-1 * h / c[iH]) * dldh]).squeeze()
                elif iH == 4:
                    dldh = self._get_dldh(h, zenith[mask], iH)
                    tmp[mask] = np.array([a[..., iH][mask] - b[iH] * h / c[iH] * dldh])
                else:
                    tmp[mask] = np.zeros(np.sum(mask))
        return tmp

    def _get_atmosphere_numeric(self, zenith, h_low=0, h_up=np.infty, observation_level=0):
        zenith = np.array(zenith)
        tmp = np.zeros_like(zenith)

        for i in xrange(len(tmp)):

            t_h_low = h_low if np.array(h_low).size == 1 else h_low[i]
            t_h_up = h_up if np.array(h_up).size == 1 else h_up[i]
            z = zenith[i]
            if hasattr(observation_level, "__len__"):
                o = observation_level[i]
            else:
                o = observation_level

            if t_h_up <= t_h_low:
                print("WARNING _get_atmosphere_numeric(): upper limit less than lower limit")
                return np.nan

            if t_h_up == np.infty:
                t_h_up = h_max

            d_low = get_distance_for_height_above_ground(t_h_low - o, z, o)
            d_up = get_distance_for_height_above_ground(t_h_up - o, z, o)

            full_atm = integrate.quad(self._get_density_for_distance, d_low, d_up, args=(z, o), limit=500)[0]

            tmp[i] = full_atm

        return tmp

    def _get_atmosphere_flat(self, zenith, h=0):
        y = _get_atmosphere(h, model=self.model)
        return y / np.cos(zenith)

    def get_vertical_height(self, zenith, xmax):
        """ returns the (vertical) height above see level [in meters] as a function
        of zenith angle and Xmax [in g/cm^2]
        """
        return self._get_vertical_height(zenith, xmax * 1e4)

    def _get_vertical_height(self, zenith, X):
        mask_flat, mask_taylor, mask_numeric = self.__get_method_mask(zenith)
        tmp = np.zeros_like(zenith)
        # print(zenith, X)
        if np.sum(mask_numeric):
            print("get vertical height numeric", zenith)
            tmp[mask_numeric] = self._get_vertical_height_numeric(*self.__get_arguments(mask_numeric, zenith, X))
        if np.sum(mask_taylor):
            tmp[mask_taylor] = self._get_vertical_height_numeric_taylor(*self.__get_arguments(mask_taylor, zenith, X))
        if np.sum(mask_flat):
            print("get vertical height flat")
            tmp[mask_flat] = self._get_vertical_height_flat(*self.__get_arguments(mask_flat, zenith, X))
        # print(tmp)
        return tmp

    def _get_vertical_height_numeric(self, zenith, X):
        height = np.zeros_like(zenith)
        zenith = np.array(zenith)

        # returns atmosphere between xmax and d
        def ftmp(d, zenith, xmax):
            h = get_height_above_ground(d, zenith, observation_level=0)  # height above sea level
            tmp = self._get_atmosphere_numeric([zenith], h_low=h)
            dtmp = tmp - xmax
            return dtmp

        for i in xrange(len(height)):
            x0 = get_distance_for_height_above_ground(self._get_vertical_height_flat(zenith[i], X[i]), zenith[i])

            # finding root e.g., distance for given xmax (when difference is 0)
            dxmax_geo = optimize.brentq(ftmp, -1e3, x0 + 1e4, xtol=1e-6, args=(zenith[i], X[i]))

            height[i] = get_height_above_ground(dxmax_geo, zenith[i], observation_level=0)

        return height

    def _get_vertical_height_numeric_taylor(self, zenith, X):
        height = np.zeros_like(zenith)
        zenith = np.array(zenith)

        # returns atmosphere between xmax and d
        def ftmp(d, zenith, xmax):
            h = get_height_above_ground(d, zenith, observation_level=0)
            tmp = self._get_atmosphere_taylor(np.array([zenith]), h_low=np.array([h]))
            dtmp = tmp - xmax
            return dtmp

        for i in xrange(len(height)):
            if(X[i] < 0):
                X[i] = 0

            x0 = get_distance_for_height_above_ground(self._get_vertical_height_flat(zenith[i], X[i]), zenith[i])

            # finding root e.g., distance for given xmax (when difference is 0)
            dxmax_geo = optimize.brentq(ftmp, -1e3, x0 + 1e4, xtol=1e-6, args=(zenith[i], X[i]))

            height[i] = get_height_above_ground(dxmax_geo, zenith[i])

        return height

    def _get_vertical_height_flat(self, zenith, X):
        return _get_vertical_height(X * np.cos(zenith), model=self.model)

    def get_density(self, zenith, xmax):
        """ returns the atmospheric density as a function of zenith angle
        and shower maximum Xmax (in g/cm^2) """
        return self._get_density(zenith, xmax * 1e4)

    def _get_density(self, zenith, xmax):
        """ returns the atmospheric density as a function of zenith angle
        and shower maximum Xmax """
        h = self._get_vertical_height(zenith, xmax)
        rho = get_density(h, model=self.model)
        return rho

    def _get_density_for_distance(self, d, zenith, observation_level=0):
        h = get_height_above_ground(d, zenith, observation_level) + observation_level
        return get_density(h, model=self.model)

    def get_distance_xmax(self, zenith, xmax, observation_level=1564.):
        """ input:
            - xmax in g/cm^2
            - zenith in radians
            output: distance to xmax in g/cm^2
        """
        dxmax = self._get_distance_xmax(zenith, xmax * 1e4, observation_level=observation_level)
        return dxmax * 1e-4

    def _get_distance_xmax(self, zenith, xmax, observation_level=1564.):
        return self._get_atmosphere(zenith, h_low=observation_level) - xmax

    def get_distance_xmax_geometric(self, zenith, xmax, observation_level=1564.):
        """ input:
            - xmax in g/cm^2
            - zenith in radians
            output: distance to xmax in m
        """
        return self._get_distance_xmax_geometric(zenith, xmax * 1e4,
                                                 observation_level=observation_level)

    def _get_distance_xmax_geometric(self, zenith, xmax, observation_level=1564.):
        h = self._get_vertical_height(zenith, xmax) - observation_level
        return get_distance_for_height_above_ground(h, zenith, observation_level)

    def get_xmax_from_distance(self, distance, zenith, observation_level=1564.):
        """ input:
            - distance to xmax in m
            - zenith in radians
            output: xmax g/cm^2
        """
        h_xmax = get_height_above_ground(distance, zenith, observation_level) + observation_level
        return self.get_atmosphere(zenith, h_low=observation_level) - \
               self.get_atmosphere(zenith, h_low=observation_level, h_up=h_xmax)

    def get_viewing_angle(self, zenith, r, xmax=600, observation_level=1564.):
        """
        calculates the viewing angle, i.e. the angle between the shower axis and the line of sight from 
        the observer to xmax.
        
        Parameters
        ----------
        zenith: float
            zenith angle
        r: float
            radial distance (distance of observer perpendicular to shower axis)
        xmax: float
        observation_level: float
        """

        dxmax = self.get_distance_xmax_geometric(zenith, xmax, observation_level)
        return np.arctan(r / dxmax)

    def get_radial_distane_from_viewing_angle(self, zenith, viewing_angle, xmax=600, observation_level=1564.):
        """
        calculates the radial distance from the observer position (defined by xmax and the viewing angle) to the shower
        axis.
        
        Parameters
        ----------
        zenith: float
            zenith angle
        viewing_angle: float
            the viewing angle, i.e. the angle between the shower axis and the line of sight from 
            the observer to xmax.
        xmax: float
        observation_level: float
        """
        dxmax = self.get_distance_xmax_geometric(zenith, xmax, observation_level)
        return dxmax * np.tan(viewing_angle)

#     def __get_distance_xmax_geometric_flat(self, xmax, observation_level=1564.):
# #         _get_vertical_height(xmax, self.model)
# #         dxmax = self._get_distance_xmax(xmax, observation_level=observation_level)
# #         txmax = _get_atmosphere(observation_level, model=self.model) - dxmax * np.cos(self.zenith)
#         return height / np.cos(self.zenith)

#         full = _get_atmosphere(observation_level, model=self.model) / np.cos(self.zenith)

# def get_distance_xmax_geometric2(xmax, zenith, observation_level=1564.,
#                                 model=1, curved=False):
#     """ input:

#     def get_atmosphere3(self, h_low=0., h_up=np.infty):
