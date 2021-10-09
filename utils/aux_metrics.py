"""
auxiliary metrics for the task of qrs detection

References
----------
[1] http://2019.icbeb.org/Challenge.html
"""
import math
from typing import Union, Optional, Sequence
from numbers import Real

import numpy as np


__all__ = [
    "compute_rpeak_metric",
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
