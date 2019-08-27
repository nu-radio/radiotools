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
    if np.shape(at) == ():
        T = _get_i_at(at, model=model)
    else:
        T = np.zeros(len(at))
        for i, at in enumerate(at):
            T[i] = _get_i_at(at, model=model)
    return T


def _get_i_at(at, model=default_model):
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
    return _get_atmosphere(h, model=model) * 1e-4


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


def get_density(h, allow_negative_heights=True, model=default_model):
    """ returns the atmospheric density [g/m^3] for the height h above see level"""
    b = atm_models[model]['b']
    c = atm_models[model]['c']
    layers = atm_models[model]['h']

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

def get_density_for_distance(d, zenith, observation_level=0, model=default_model):
    """ returns the atmospheric density [g/m^3] for a given distance and zenith angle assuming a curved atmosphere"""
    h = get_height_above_ground(d, zenith, observation_level=observation_level)
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


def get_n(h, n0=(1 + 2.92e-4), allow_negative_heights=False,
          model=1):
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
                    self.a, self.d = fin["a"], fin["d"]

                if(len(self.a) != self.number_of_zeniths):
                    os.remove(filename)
                    print("constants outdated, please rerun to calculate new constants")
                    sys.exit(0)

                zeniths = np.arccos(np.linspace(0, 1, self.number_of_zeniths))
                mask = zeniths < np.deg2rad(90)
                self.a_funcs = [interpolate.interp1d(zeniths[mask], self.a[..., i][mask], kind='cubic') for i in xrange(5)]

            else:
                self.d = np.zeros(self.number_of_zeniths)   # self.d = self.__calculate_d()
                self.a = self.__calculate_a()
                np.savez(filename, a=self.a, d=self.d)
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

    def get_atmosphere(self, zenith, h_low=0., h_up=np.infty):
        """ returns the atmosphere for an air shower with given zenith angle (in g/cm^2) """
        return self._get_atmosphere(zenith, h_low=h_low, h_up=h_up) * 1e-4

    def _get_atmosphere(self, zenith, h_low=0., h_up=np.infty):
        mask_flat, mask_taylor, mask_numeric = self.__get_method_mask(zenith)
        mask_finite = np.array((h_up * np.ones_like(zenith)) < h_max)
        is_mask_finite = np.sum(mask_finite)
        tmp = np.zeros_like(zenith)
        if np.sum(mask_numeric):
            # print("getting numeric")
            tmp[mask_numeric] = self._get_atmosphere_numeric(*self.__get_arguments(mask_numeric, zenith, h_low, h_up))
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

    def __get_zenith_a_indices(self, zeniths):
        n = self.number_of_zeniths - 1
        cosz_bins = np.linspace(0, n, self.number_of_zeniths, dtype=np.int)
        cosz = np.array(np.round(np.cos(zeniths) * n), dtype=np.int)
        tmp = np.squeeze([np.argwhere(t == cosz_bins) for t in cosz])
        return tmp

    def __get_a_from_cache(self, zeniths):
        n = self.number_of_zeniths - 1
        cosz_bins = np.linspace(0, n, self.number_of_zeniths, dtype=np.int)
        cosz = np.array(np.round(np.cos(zeniths) * n), dtype=np.int)
        a_indices = np.squeeze([np.argwhere(t == cosz_bins) for t in cosz])
        cosz_bins_num = np.linspace(0, 1, self.number_of_zeniths)
        a = ((self.a[a_indices]).T * (cosz_bins_num[a_indices] / np.cos(zeniths))).T
        return a

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
        # a_indices = self.__get_zenith_a_indices(zenith)
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

    def _get_atmosphere_numeric(self, zenith, h_low=0, h_up=np.infty):
        zenith = np.array(zenith)
        tmp = np.zeros_like(zenith)

        for i in xrange(len(tmp)):

            t_h_low = h_low if np.array(h_low).size == 1 else h_low[i]
            t_h_up = h_up if np.array(h_up).size == 1 else h_up[i]
            z = zenith[i]

            if t_h_up <= t_h_low:
                print("WARNING _get_atmosphere_numeric(): upper limit less than lower limit")
                return np.nan

            if t_h_up == np.infty:
                t_h_up = h_max

            d_low = get_distance_for_height_above_ground(t_h_low, z)
            d_up = get_distance_for_height_above_ground(t_h_up, z)

            full_atm = integrate.quad(self._get_density_for_distance,
                                      d_low, d_up, args=(z,),
                                      limit=500)[0]
            tmp[i] = full_atm

        return tmp

    def _get_atmosphere_flat(self, zenith, h=0):
        a = atm_models[self.model]['a']
        b = atm_models[self.model]['b']
        c = atm_models[self.model]['c']
        layers = atm_models[self.model]['h']
        y = np.where(h < layers[0], a[0] + b[0] * np.exp(-1 * h / c[0]), a[1] + b[1] * np.exp(-1 * h / c[1]))
        y = np.where(h < layers[1], y, a[2] + b[2] * np.exp(-1 * h / c[2]))
        y = np.where(h < layers[2], y, a[3] + b[3] * np.exp(-1 * h / c[3]))
        y = np.where(h < layers[3], y, a[4] - b[4] * h / c[4])
        y = np.where(h < h_max, y, 0)
        return y / np.cos(zenith)

    def get_vertical_height(self, zenith, xmax):
        """ returns the (vertical) height above see level [in meters] as a function
        of zenith angle and Xmax [in g/cm^2]
        """
        return self._get_vertical_height(zenith, xmax * 1e4)

    def _get_vertical_height(self, zenith, X):
        mask_flat, mask_taylor, mask_numeric = self.__get_method_mask(zenith)
        tmp = np.zeros_like(zenith)
        if np.sum(mask_numeric):
            print("get vertical height numeric", zenith)
            tmp[mask_numeric] = self._get_vertical_height_numeric(*self.__get_arguments(mask_numeric, zenith, X))
        if np.sum(mask_taylor):
            tmp[mask_taylor] = self._get_vertical_height_numeric_taylor(*self.__get_arguments(mask_taylor, zenith, X))
        if np.sum(mask_flat):
            print("get vertical height flat")
            tmp[mask_flat] = self._get_vertical_height_flat(*self.__get_arguments(mask_flat, zenith, X))
        return tmp

    def __calculate_d(self):
        zeniths = np.arccos(np.linspace(0, 1, self.number_of_zeniths))
        d = np.zeros((self.number_of_zeniths, 4))
        self.curved = True
        self.__zenith_numeric = 0
        for iZ, z in enumerate(zeniths):
            z = np.array([z])
            print("calculating constants for %.02f deg zenith angle (iZ = %i, nT = %i)..." % (np.rad2deg(z), iZ, self.n_taylor))
            d[iZ][0] = 0
            X1 = self._get_atmosphere(z, self.h[1])
            d[iZ][1] = self._get_vertical_height_numeric(z, X1) - self._get_vertical_height_taylor_wo_constants(z, X1)
            X2 = self._get_atmosphere(z, self.h[2])
            d[iZ][2] = self._get_vertical_height_numeric(z, X2) - self._get_vertical_height_taylor_wo_constants(z, X2)
            X3 = self._get_atmosphere(z, self.h[3])
            d[iZ][3] = self._get_vertical_height_numeric(z, X3) - self._get_vertical_height_taylor_wo_constants(z, X3)
            print("\t... d  = ", d[iZ], " iZ = ", iZ)
        return d

    def _get_vertical_height_taylor(self, zenith, X):
        tmp = self._get_vertical_height_taylor_wo_constants(zenith, X)
        masks = self.__get_X_masks(X, zenith)
        d = self.d[self.__get_zenith_a_indices(zenith)]
        for iX, mask in enumerate(masks):
            if(np.sum(mask)):
                if iX < 4:
                    tmp[mask] += d[mask][..., iX]

        return tmp

    def _get_vertical_height_taylor_wo_constants(self, zenith, X):
        b = self.b
        c = self.c
        ct = np.cos(zenith)
        T0 = self._get_atmosphere(zenith)
        masks = self.__get_X_masks(X, zenith)
        # Xs = [self._get_atmosphere(zenith, h) for h in self.h]
        # d = np.array([self._get_vertical_height_numeric(zenith, t) for t in Xs])
        tmp = np.zeros_like(zenith)
        for iX, mask in enumerate(masks):
            if(np.sum(mask)):
                if iX < 4:
                    xx = X[mask] - T0[mask]
                    if self.n_taylor >= 1:
                        tmp[mask] = -c[iX] / b[iX] * ct[mask] * xx
                    if self.n_taylor >= 2:
                        tmp[mask] += -0.5 * c[iX] * (ct[mask] ** 2 * c[iX] - ct[mask] ** 2 * r_e - c[iX]) / (r_e * b[iX] ** 2) * xx ** 2
                    if self.n_taylor >= 3:
                        tmp[mask] += -1. / 6. * c[iX] * ct[mask] * (3 * ct[mask] ** 2 * c[iX] ** 2 - 4 * ct[mask] ** 2 * r_e * c[iX] + 2 * r_e ** 2 * ct[mask] ** 2 - 3 * c[iX] ** 2 + 4 * r_e * c[iX]) / (r_e ** 2 * b[iX] ** 3) * xx ** 3
                    if self.n_taylor >= 4:
                        tmp[mask] += -1. / (24. * r_e ** 3 * b[iX] ** 4) * c[iX] * (15 * ct[mask] ** 4 * c[iX] ** 3 - 25 * c[iX] ** 2 * r_e * ct[mask] ** 4 + 18 * c[iX] * r_e ** 2 * ct[mask] ** 4 - 6 * r_e ** 3 * ct[mask] ** 4 - 18 * c[iX] ** 3 * ct[mask] ** 2 + 29 * c[iX] ** 2 * r_e * ct[mask] ** 2 - 18 * c[iX] * r_e ** 2 * ct[mask] ** 2 + 3 * c[iX] ** 3 - 4 * c[iX] ** 2 * r_e) * xx ** 4
                    if self.n_taylor >= 5:
                        tmp[mask] += -1. / (120. * r_e ** 4 * b[iX] ** 5) * c[iX] * ct[mask] * (ct[mask] ** 4 * (105 * c[iX] ** 4 - 210 * c[iX] ** 3 * r_e + 190 * c[iX] ** 2 * r_e ** 2 - 96 * c[iX] * r_e ** 3 + 24 * r_e ** 4) + ct[mask] ** 2 * (-150 * c[iX] ** 4 + 288 * c[iX] ** 3 * r_e - 242 * c[iX] ** 2 * r_e ** 2 + 96 * c[iX] * r_e ** 3) + 45 * c[iX] ** 4 - 78 * r_e * c[iX] ** 3 + 52 * r_e ** 2 * c[iX] ** 2) * xx ** 5
                    if self.n_taylor >= 6:
                        tmp[mask] += -1. / (720. * r_e ** 5 * b[iX] ** 6) * c[iX] * (ct[mask] ** 6 * (945 * c[iX] ** 5 - 2205 * c[iX] ** 4 * r_e + 2380 * c[iX] ** 3 * r_e ** 2 - 1526 * c[iX] ** 2 * r_e ** 3 + 600 * c[iX] * r_e ** 4 - 120 * r_e ** 5) + ct[mask] ** 4 * (-1575 * c[iX] ** 5 + 3528 * c[iX] ** 4 * r_e - 3600 * c[iX] ** 3 * r_e ** 2 + 2074 * c[iX] ** 2 * r_e ** 3 - 600 * c[iX] * r_e ** 4) + ct[mask] ** 2 * (675 * c[iX] ** 5 - 1401 * c[iX] ** 4 * r_e - 1272 * c[iX] ** 3 * r_e ** 2 - 548 * c[iX] ** 2 * r_e ** 3) - 45 * c[iX] ** 5 + 78 * c[iX] ** 4 * r_e - 52 * c[iX] ** 3 * r_e ** 2) * xx ** 6
                elif iX == 4:
                    print("iX == 4", iX)
                    # numeric fallback
                    tmp[mask] = self._get_vertical_height_numeric(zenith, X)
                else:
                    print("iX > 4", iX)
                    tmp[mask] = np.ones_like(mask) * h_max
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

#     def __get_density2_curved(self, xmax):
#         dxmax_geo = self._get_distance_xmax_geometric(xmax, observation_level=0)
#         return self._get_density_for_distance(dxmax_geo)

    def _get_density_for_distance(self, d, zenith):
        h = get_height_above_ground(d, zenith)
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

#     def __get_distance_xmax_geometric_flat(self, xmax, observation_level=1564.):
# #         _get_vertical_height(xmax, self.model)
# #         dxmax = self._get_distance_xmax(xmax, observation_level=observation_level)
# #         txmax = _get_atmosphere(observation_level, model=self.model) - dxmax * np.cos(self.zenith)
# #         height = _get_vertical_height(txmax)
# #         return (height - observation_level) / np.cos(self.zenith)
# #
#         height = _get_vertical_height(xmax * np.cos(self.zenith)) - observation_level
#         return height / np.cos(self.zenith)

#         full = _get_atmosphere(observation_level, model=self.model) / np.cos(self.zenith)
#         dxmax = full - xmax
#         height = _get_vertical_height(_get_atmosphere(0, model=self.model) - dxmax * np.cos(self.zenith))
#         return height / np.cos(self.zenith)

# def get_distance_xmax_geometric2(xmax, zenith, observation_level=1564.,
#                                 model=1, curved=False):
#     """ input:
#         - xmax in g/cm^2
#         - zenith in radians
#         output: distance to xmax in m
#     """
#     return _get_distance_xmax_geometric2(zenith, xmax * 1e4,
#                                         observation_level=observation_level,
#                                         model=model, curved=curved)
# def _get_distance_xmax_geometric2(zenith, xmax, observation_level=1564.,
#                                  model=default_model,
#                                  curved=default_curved):
#     if curved:
#         x0 = _get_distance_xmax_geometric(zenith, xmax,
#                                           observation_level=observation_level,
#                                           model=model, curved=False)
#
#         def ftmp(d, dxmax, zenith, observation_level):
#             h = get_height_above_ground(d, zenith, observation_level=observation_level)
#             h += observation_level
#             dtmp = _get_atmosphere2(zenith, h_low=observation_level, h_up=h, model=model) - dxmax
#             print "d = %.5g, h = %.5g, dtmp = %.5g" % (d, h, dtmp)
#             return dtmp
#
#         dxmax = _get_distance_xmax(xmax, zenith, observation_level=observation_level, curved=True)
#         print "distance to xmax = ", dxmax
#         tolerance = max(1e-3, x0 * 1.e-6)
#         dxmax_geo = optimize.newton(ftmp, x0=x0, maxiter=100, tol=tolerance,
#                                     args=(dxmax, zenith, observation_level))
#         # print "x0 = %.7g, dxmax_geo = %.7g" % (x0, dxmax_geo)
#         return dxmax_geo
#     else:
#         dxmax = _get_distance_xmax(xmax, zenith, observation_level=observation_level,
#                                    model=model, curved=False)
#         xmax = _get_atmosphere(observation_level, model=model) - dxmax * np.cos(zenith)
#         height = _get_vertical_height(xmax)
#         return (height - observation_level) / np.cos(zenith)
#     def _get_atmosphere2(self, zenith, h_low=0., h_up=np.infty):
#         if use_curved(zenith, self.curved):
#             if h_up <= h_low:
#                 print "WARNING: upper limit less than lower limit"
#                 return np.nan
#             if h_up == np.infty:
#                 h_up = h_max
#             b = h_up
#             d_low = get_distance_for_height_above_ground(h_low, zenith)
#             d_up = get_distance_for_height_above_ground(b, zenith)
#             d_up_1 = d_low + 2.e3
#             if d_up_1 > d_up:
#                 full_atm = integrate.quad(self._get_density_for_distance,
#                                           zenith, d_low, d_up, limit=100, epsabs=1e-2)[0]
#             else:
#                 full_atm = integrate.quad(self._get_density_for_distance,
#                                           zenith, d_low, d_up_1, limit=100, epsabs=1e-4)[0]
#                 full_atm += integrate.quad(self._get_density_for_distance,
#                                            zenith, d_up_1, d_up, limit=100, epsabs=1e-2)[0]
#             return full_atm
#         else:
#             return (_get_atmosphere(h_low, model=self.model) - _get_atmosphere(h_up, model=self.model)) / np.cos(zenith)

#     def get_atmosphere3(self, h_low=0., h_up=np.infty):
#         return self._get_atmosphere3(h_low=h_low, h_up=h_up) * 1e-4
#
#     def _get_atmosphere3(self, h_low=0., h_up=np.infty):
#         a = self.a
#         b = self.b
#         c = self.c
#         h = h_low
#         layers = atm_models[self.model]['h']
#         dldh = self._get_dldh(h)
#         y = np.where(h < layers[0], a[0] + b[0] * np.exp(-1 * h / c[0]) * dldh[0], a[1] + b[1] * np.exp(-1 * h / c[1]) * dldh[1])
#         y = np.where(h < layers[1], y, a[2] + b[2] * np.exp(-1 * h / c[2]) * dldh[2])
#         y = np.where(h < layers[2], y, a[3] + b[3] * np.exp(-1 * h / c[3]) * dldh[3])
#         y = np.where(h < layers[3], y, a[4] - b[4] * h / c[4] * dldh[4])
#         y = np.where(h < h_max, y, 0)
#         return y
