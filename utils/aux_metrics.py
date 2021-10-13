"""
auxiliary metrics for the task of qrs detection

References
----------
[1] http://2019.icbeb.org/Challenge.html
"""
import math
import multiprocessing as mp
from typing import Union, Optional, Sequence
from numbers import Real

import numpy as np

from .misc import mask_to_intervals


__all__ = [
    "compute_rpeak_metric",
    "compute_rr_metric",
]


def compute_rpeak_metric(rpeaks_truths:Sequence[Union[np.ndarray,Sequence[int]]],
                         rpeaks_preds:Sequence[Union[np.ndarray,Sequence[int]]],
                         fs:Real,
                         thr:float=0.075,
                         verbose:int=0) -> float:
    """ finished, checked,

    Parameters
    ----------
    rpeaks_truths: sequence,
        sequence of ground truths of rpeaks locations (indices) from multiple records
    rpeaks_preds: sequence,
        predictions of ground truths of rpeaks locations (indices) for multiple records
    fs: real number,
        sampling frequency of ECG signal
    thr: float, default 0.075,
        threshold for a prediction to be truth positive,
        with units in seconds,
    verbose: int, default 0,
        print verbosity

    Returns
    -------
    rec_acc: float,
        accuracy of predictions
    """
    assert len(rpeaks_truths) == len(rpeaks_preds), \
        f"number of records does not match, truth indicates {len(rpeaks_truths)}, while pred indicates {len(rpeaks_preds)}"
    n_records = len(rpeaks_truths)
    record_flags = np.ones((len(rpeaks_truths),), dtype=float)
    thr_ = thr * fs
    if verbose >= 1:
        print(f"number of records = {n_records}")
        print(f"threshold in number of sample points = {thr_}")
    for idx, (truth_arr, pred_arr) in enumerate(zip(rpeaks_truths, rpeaks_preds)):
        false_negative = 0
        false_positive = 0
        true_positive = 0
        extended_truth_arr = np.concatenate((truth_arr.astype(int), [int(9.5*fs)]))
        for j, t_ind in enumerate(extended_truth_arr[:-1]):
            next_t_ind = extended_truth_arr[j+1]
            loc = np.where(np.abs(pred_arr - t_ind) <= thr_)[0]
            if j == 0:
                err = np.where((pred_arr >= 0.5*fs + thr_) & (pred_arr <= t_ind - thr_))[0]
            else:
                err = np.array([], dtype=int)
            err = np.append(
                err,
                np.where((pred_arr >= t_ind+thr_) & (pred_arr <= next_t_ind-thr_))[0]
            )

            false_positive += len(err)
            if len(loc) >= 1:
                true_positive += 1
                false_positive += len(loc) - 1
            elif len(loc) == 0:
                false_negative += 1

        if false_negative + false_positive > 1:
            record_flags[idx] = 0
        elif false_negative == 1 and false_positive == 0:
            record_flags[idx] = 0.3
        elif false_negative == 0 and false_positive == 1:
            record_flags[idx] = 0.7

        if verbose >= 2:
            print(f"for the {idx}-th record,\ntrue positive = {true_positive}\nfalse positive = {false_positive}\nfalse negative = {false_negative}")

    rec_acc = round(np.sum(record_flags) / n_records, 4)

    if verbose >= 1:
        print(f'QRS_acc: {rec_acc}')
        print('Scoring complete.')

    return rec_acc


def compute_rr_metric(rr_truths:Sequence[Union[np.ndarray,Sequence[int]]],
                      rr_preds:Sequence[Union[np.ndarray,Sequence[int]]],
                      verbose:int=0) -> float:
    """ finished, checked,

    this metric imitates the metric provided by the organizers of CPSC2021

    Parameters
    ----------
    rr_truths: array_like,
        sequences of AF labels on rr intervals, of shape (n_samples, seq_len)
    rr_truths: array_like,
        sequences of AF predictions on rr intervals, of shape (n_samples, seq_len)

    Returns
    -------
    score: float,
        the score, similar to CPSC2021 challenge metric
    """
    with mp.Pool(processes=max(1,mp.cpu_count())) as pool:
        af_episode_truths = pool.starmap(
            func=mask_to_intervals,
            iterable=[(row,1,True) for row in rr_truths]
        )
    with mp.Pool(processes=max(1,mp.cpu_count())) as pool:
        af_episode_preds = pool.starmap(
            func=mask_to_intervals,
            iterable=[(row,1,True) for row in rr_preds]
        )
    scoring_mask = np.zeros_like(np.array(rr_truths))
    n_samples, seq_len = scoring_mask.shape
    for idx, sample in enumerate(af_episode_truths):
        for itv in sample:
            scoring_mask[idx][max(0,itv[0]-2):min(seq_len,itv[0]+2)] = 0.5
            scoring_mask[idx][max(0,itv[1]-2):min(seq_len,itv[1]+3)] = 0.5
            scoring_mask[idx][max(0,itv[0]-1):min(seq_len,itv[0]+2)] = 1
            scoring_mask[idx][max(0,itv[1]-1):min(seq_len,itv[0]+2)] = 1
    score = sum([scoring_mask[idx][itv].sum() for idx in range(n_samples) for itv in af_episode_preds[idx]])
    return score
