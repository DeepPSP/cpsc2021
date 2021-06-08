"""
denoise, mainly concerning the motion artefacts, using naive methods

some of the CPSC2020 records have segments of severe motion artefacts,
such segments should be eliminated from feature computation

Process:
--------
1. detect segments of nearly constant values, and slice these segments out
2. detect motion artefact (large variation of values in a short time), and further slice the record into segments without motion artefact
3. more?

References:
-----------
to add
"""
from typing import Union, Optional, List
from numbers import Real

import numpy as np
from easydict import EasyDict as ED

from utils.misc import mask_to_intervals


__all__ = [
    "remove_spikes_naive",
    "ecg_denoise_naive",
]


def remove_spikes_naive(sig:np.ndarray) -> np.ndarray:
    """ finished, checked,

    remove `spikes` from `sig` using a naive method proposed in entry 0416 of CPSC2019

    `spikes` here refers to abrupt large bumps with (abs) value larger than 20 mV,
    do NOT confuse with `spikes` in paced rhythm

    Parameters
    ----------
    sig: ndarray,
        single-lead ECG signal with potential spikes
    
    Returns
    -------
    filtered_sig: ndarray,
        ECG signal with `spikes` removed
    """
    b = list(filter(lambda k: k > 0, np.argwhere(np.abs(sig)>20).squeeze(-1)))
    filtered_sig = sig.copy()
    for k in b:
        filtered_sig[k] = filtered_sig[k-1]
    return filtered_sig


def ecg_denoise_naive(filtered_sig:np.ndarray, fs:Real, config:ED) -> List[List[int]]:
    """ finished, checked,

    a naive function removing non-ECG segments (flat and motion artefact)

    Parameters
    ----------
    filtered_sig: ndarray,
        single-lead filtered (typically bandpassed) ECG signal,
    fs: real number,
        sampling frequency of `filtered_sig`
    config: dict,
        configs of relavant parameters, like window, step, etc.

    Returns
    -------
    intervals: list of (length 2) list of int,
        list of intervals of non-noise segment of `filtered_sig`
    """
    _LABEL_VALID, _LABEL_INVALID = 1, 0
    # constants
    siglen = len(filtered_sig)
    window = int(config.get("window", 2000) * fs / 1000)  # 2000 ms
    step = int(config.get("step", window/5))
    ampl_min = config.get("ampl_min", 0.2)  # 0.2 mV
    ampl_max = config.get("ampl_max", 6.0)  # 6 mV

    mask = np.zeros_like(filtered_sig, dtype=int)

    if siglen < window:
        result = []
        return result

    # detect and remove flat part
    n_seg, residue = divmod(siglen-window+step, step)
    start_inds = [idx*step for idx in range(n_seg)]
    if residue != 0:
        start_inds.append(siglen-window)
        n_seg += 1

    for idx in start_inds:
        window_vals = filtered_sig[idx:idx+window]
        ampl = np.max(window_vals)-np.min(window_vals)
        if ampl > ampl_min:
            mask[idx:idx+window] = _LABEL_VALID

    # detect and remove motion artefact
    window = window // 2  # 1000 ms
    step = window // 5
    n_seg, residue = divmod(siglen-window+step, step)
    start_inds = [idx*step for idx in range(n_seg)]
    if residue != 0:
        start_inds.append(siglen-window)
        n_seg += 1

    for idx in start_inds:
        window_vals = filtered_sig[idx:idx+window]
        ampl = np.max(window_vals)-np.min(window_vals)
        if ampl > ampl_max:
            mask[idx:idx+window] = _LABEL_INVALID

    # mask to intervals
    interval_threshold = int(config.get("len_threshold", 5)*fs)  # 5s
    intervals = mask_to_intervals(mask, _LABEL_VALID)
    intervals = [item for item in intervals if item[1]-item[0]>interval_threshold]

    return intervals
