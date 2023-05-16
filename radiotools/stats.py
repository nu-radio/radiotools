"""
Statistic functions
"""
import numpy as np
from numpy import arange, array, bincount, ndarray, ones, where
from numpy.random import seed, random, randint


class WalkerRandomSampling(object):
    """Walker's alias method for random objects with different probablities.
    
    from https://gist.github.com/ntamas/1109133
    __author__ = "Tamas Nepusz, Denis Bzowy"
    __version__ = "27jul2011"
    
    Based on the implementation of Denis Bzowy at the following URL:
    http://code.activestate.com/recipes/576564-walkers-alias-method-for-random-objects-with-diffe/
    """

    def __init__(self, weights, keys=None):
        """Builds the Walker tables ``prob`` and ``inx`` for calls to `random()`.
        The weights (a list or tuple or iterable) can be in any order and they
        do not even have to sum to 1."""
        n = self.n = len(weights)
        if keys is None:
            self.keys = keys
        else:
            self.keys = array(keys)

        if isinstance(weights, (list, tuple)):
            weights = array(weights, dtype=float)
        elif isinstance(weights, ndarray):
            if weights.dtype != float:
                weights = weights.astype(float)
        else:
            weights = array(list(weights), dtype=float)

        if weights.ndim != 1:
            raise ValueError("weights must be a vector")

        weights = weights * n / weights.sum()

        inx = -ones(n, dtype=int)
        short = where(weights < 1)[0].tolist()
        long = where(weights > 1)[0].tolist()
        while short and long:
            j = short.pop()
            k = long[-1]

            inx[j] = k
            weights[k] -= (1 - weights[j])
            if weights[k] < 1:
                short.append(k)
                long.pop()

        self.prob = weights
        self.inx = inx

    def random(self, count=None):
        """Returns a given number of random integers or keys, with probabilities
        being proportional to the weights supplied in the constructor.

        When `count` is ``None``, returns a single integer or key, otherwise
        returns a NumPy array with a length given in `count`.
        """
        if count is None:
            u = random()
            j = randint(self.n)
            k = j if u <= self.prob[j] else self.inx[j]
            return self.keys[k] if self.keys is not None else k

        u = random(count)
        j = randint(self.n, size=count)
        k = where(u <= self.prob[j], j, self.inx[j])
        return self.keys[k] if self.keys is not None else k



def mid(x):
    """
    Midpoints of a given array
    """
    return (x[:-1] + x[1:]) / 2.


def mean_and_variance(y, weights):
    """
    Weighted mean and variance
    """
    w_sum = sum(weights)
    m = np.dot(y, weights) / w_sum
    v = np.dot((y - m) ** 2, weights) / w_sum
    return m, v


def quantile_1d(data, weights, quant):
    # from https://github.com/nudomarinero/wquantiles/blob/master/weighted.py
    """
    Compute the weighted quantile of a 1D numpy array.
    Parameters
    ----------
    data : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    quant : float
        Quantile to compute. It must have a value between 0 and 1.
    Returns
    -------
    quantile_1d : float
        The output value.
    """
    # Check the data
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    nd = data.ndim
    if nd != 1:
        raise TypeError("data must be a one dimensional array")
    ndw = weights.ndim
    if ndw != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if (quant > 1.) or (quant < 0.):
        raise ValueError("quantile must have a value between 0. and 1.")
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    sn = np.cumsum(sorted_weights)
    # TODO: Check that the weights do not sum zero
    # assert Sn != 0, "The sum of the weights must not be zero"
    pn = (sn - 0.5 * sorted_weights) / np.sum(sorted_weights)
    # Get the value of the weighted median
    # noinspection PyTypeChecker
    return np.interp(quant, pn, sorted_data)


def quantile(data, weights, quant):  # pylint: disable=R1710
    # from https://github.com/nudomarinero/wquantiles/blob/master/weighted.py
    """
    Weighted quantile of an array with respect to the last axis.
    Parameters
    ----------
    data : ndarray
        Input array.
    weights : ndarray
        Array with the weights. It must have the same size of the last
        axis of `data`.
    quant : float
        Quantile to compute. It must have a value between 0 and 1.
    Returns
    -------
    quantile : float
        The output value.
    """
    # TODO: Allow to specify the axis
    nd = data.ndim
    if nd == 0:
        TypeError("data must have at least one dimension")
    elif nd == 1:
        return quantile_1d(data, weights, quant)
    elif nd > 1:
        n = data.shape
        imr = data.reshape((np.prod(n[:-1]), n[-1]))
        result = np.apply_along_axis(quantile_1d, -1, imr, weights, quant)
        return result.reshape(n[:-1])


def median(data, weights):
    # from https://github.com/nudomarinero/wquantiles/blob/master/weighted.py
    """
    Weighted median of an array with respect to the last axis.
    Alias for `quantile(data, weights, 0.5)`.
    """
    return quantile(data, weights, 0.5)


def binned_mean(x, y, bins, weights=None):
    """
    <y>_i : mean of y in bins of x
    """
    dig = np.digitize(x, bins)
    n = len(bins) - 1
    my = np.zeros(n)
    if weights is None:
        weights = np.ones(len(x))  # use weights=1 if none given

    for i in range(n):
        idx = (dig == i + 1)
        try:
            my[i] = np.average(y[idx], weights=weights[idx])
        except ZeroDivisionError:
            my[i] = np.nan

    return my


def binned_mean_and_variance(x, y, bins, weights=None):
    """
    <y>_i, sigma(y)_i : mean and variance of y in bins of x
    This is effectively a ROOT.TProfile
    """
    dig = np.digitize(x, bins)
    n = len(bins) - 1
    my, vy = np.zeros(n), np.zeros(n)

    for i in range(n):
        idx = (dig == i + 1)

        if not idx.any():  # check for empty bin
            my[i] = np.nan
            vy[i] = np.nan
            continue

        if weights is None:
            my[i] = np.mean(y[idx])
            vy[i] = np.std(y[idx]) ** 2
        else:
            my[i], vy[i] = mean_and_variance(y[idx], weights[idx])

    return my, vy


def sym_interval_around(x, xm, alpha):
    """
    In a distribution represented by a set of samples, find the interval
    that contains (1-alpha)/2 to each the left and right of xm.
    If xm is too marginal to allow both sides to contain (1-alpha)/2,
    add the remaining fraction to the other side.
    """
    xt = x.copy()
    xt.sort()
    i = xt.searchsorted(xm)  # index of central value
    n = len(x)  # number of samples
    ns = int((1 - alpha) * n)  # number of samples corresponding to 1-alpha

    i0 = i - ns / 2  # index of lower and upper bound of interval
    i1 = i + ns / 2

    # if central value doesn't allow for (1-alpha)/2 on left side, add to right
    if i0 < 0:
        i1 -= i0
        i0 = 0
    # if central value doesn't allow for (1-alpha)/2 on right side, add to left
    if i1 >= n:
        i0 -= i1 - n + 1
        i1 = n - 1

    return xt[int(i0)], xt[int(i1)]
