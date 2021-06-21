from scipy.signal import hilbert

import numpy as np
import sys

conversion_factor_integrated_signal = 2.65441729e-3 * 6.24150934e18  # to convert V**2/m**2 * s -> J/m**2 -> eV/m**2


def calculate_energy_fluence_vector(traces, times, signal_window=100., remove_noise=True):
    """ get energy fluence vector

    Parameters
    ----------
    traces: array
        time series in V / m
        array is expected to have 2 (time, polarisation) or 3 (antenna, time, polarisation) dimensions
    times: array
        corresponding time vector in seconds
        array is expected to have 1 (time) or 2 (antenna, time) dimensions
    signal_window (optional): float
        time window used to calculate the signal power in nano seconds

    remove_noise (optional): bool
        if true, subtract energy content in noise window (~signal_window) from content in signal window
        default: True

    Returns
    --------
    array:
        energy fluence per polarisation in eV / m**2 (not a real vector)
        array has the (3,) or (n_antenna, 3)
    """

    if traces.ndim != 2 and traces.ndim != 3 and traces.ndim != times.ndim:
        sys.exit("Error: traces does not fullfil reqiuerments")

    # if traces for only on antenna is given (dim = 2) a dummy dimension is added
    if traces.ndim == 2:
        traces = np.expand_dims(traces, axis=0)
        times = np.expand_dims(times, axis=0)

    # determine signal position with maximum of hilbert envelope
    hilbenv = np.abs(hilbert(traces, axis=1))
    hilbenv_sum_max_idx = np.argmax(np.sum(hilbenv ** 2, axis=-1) ** 0.5, axis=-1)  # produces FutureWarning
    peak_sum_time = times[range(len(hilbenv_sum_max_idx)), hilbenv_sum_max_idx]

    # choose signal window
    # conversion from ns in s
    signal_window *= 1e-9
    mask_signal = (times > (peak_sum_time[..., None] - signal_window / 2.)) & (times < (peak_sum_time[..., None] + signal_window / 2.))

    # get tstep
    tstep = times[0, 1] - times[0, 0]

    # calculate energy fluence in signal window
    u_signal = np.array([np.sum(traces[i][mask_signal[i]] ** 2, axis=0) for i in range(len(mask_signal))])
    u_signal *= conversion_factor_integrated_signal * tstep

    if remove_noise:
        mask_noise = ~mask_signal
        # calculate energy fluence in noise window
        u_noise = np.array([np.sum(traces[i][mask_noise[i]] ** 2, axis=0) for i in range(len(mask_noise))])
        u_noise *= conversion_factor_integrated_signal * tstep
        # account for unequal window sizes
        u_noise *= (np.sum(mask_signal, axis=-1) / np.sum(mask_noise, axis=-1))[..., None]
        power = u_signal - u_noise

    else:
        power = u_signal

    return np.squeeze(power)


def calculate_energy_fluence(traces, times, signal_window=100., remove_noise=True):
    """ get energy fluence

    Parameters
    ----------
    traces: array
        time series in V / m
        array is expected to have 2 (time, polarisation) or 3 (antenna, time, polarisation) dimensions
    times: array
        corresponding time vector in seconds
        array is expected to have 1 (time) or 2 (antenna, time) dimensions
    signal_window (optional): float
        time window used to calculate the signal power in nano seconds

    remove_noise (optional): bool
        if true, subtract energy content in noise window (~signal_window) from content in signal window
        default: True

    Returns
    --------
    float or array:
        energy fluence in eV / m**2
        for a singel station or array of stations
    """

    energy_fluence_vector = calculate_energy_fluence_vector(traces, times, signal_window, remove_noise)

    return np.sum(energy_fluence_vector, axis=-1)
