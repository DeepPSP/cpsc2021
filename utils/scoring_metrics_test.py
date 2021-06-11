"""
"""
from typing import Union, List
from numbers import Real

import numpy as np
import wfdb

from .scoring_metrics import (
    RefInfo, load_ans,
    score, ue_calculate, ur_calculate,
    compute_challenge_metric, gen_endpoint_score_mask,
)
from cfg import BaseCfg



def _load_af_episodes(fp:str, fmt:str="intervals") -> Union[List[List[int]], np.ndarray]:
    """ finished, checked,

    load the episodes of atrial fibrillation, in terms of intervals or mask,

    a simplified version of corresponding method in the data reader class in data_reader.py

    Parameters
    ----------
    fp: str,
        path of the record
    sampfrom: int, optional,
        start index of the data to be loaded,
        not used when `fmt` is "c_intervals"
    fmt: str, default "intervals",
        format of the episodes of atrial fibrillation, can be one of "intervals", "mask", "c_intervals"

    Returns
    -------
    af_episodes: list or ndarray,
        episodes of atrial fibrillation, in terms of intervals or mask
    """
    header = wfdb.rdheader(fp)
    ann = wfdb.rdann(fp, extension="atr")
    aux_note = np.array(ann.aux_note)
    critical_points = ann.sample
    af_start_inds = np.where((aux_note=="(AFIB") | (aux_note=="(AFL"))[0]
    af_end_inds = np.where(aux_note=="(N")[0]
    assert len(af_start_inds) == len(af_end_inds), \
        "unequal number of af period start indices and af period end indices"

    if fmt.lower() in ["c_intervals",]:
        af_episodes = [[start, end] for start, end in zip(af_start_inds, af_end_inds)]
        return af_episodes

    intervals = []
    for start, end in zip(af_start_inds, af_end_inds):
        itv = [critical_points[start], critical_points[end]]
        intervals.append(itv)
    af_episodes = intervals

    if fmt.lower() in ["mask",]:
        mask = np.zeros((header.sig_len,), dtype=int)
        for itv in intervals:
            mask[itv[0]:itv[1]] = 1
        af_episodes = mask

    return af_episodes
