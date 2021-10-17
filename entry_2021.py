#!/usr/bin/env python3

import numpy as np
import os
import sys
import time

import wfdb
import numpy as np
import torch
import scipy.signal as SS

from utils.misc import save_dict
from model import (
    ECG_SEQ_LAB_NET_CPSC2021,
    ECG_UNET_CPSC2021,
    RR_LSTM_CPSC2021,
    _qrs_detection_post_process,
    _main_task_post_process,
)

"""
Written by:  Xingyao Wang, Chengyu Liu
             School of Instrument Science and Engineering
             Southeast University, China
             chengyu@seu.edu.cn

Save answers to '.json' files, the format is as {‘predict_endpoints’: [[s0, e0], [s1, e1], …, [sm-1, em-2]]}.
"""


_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_CUDA = torch.device("cuda")
_CPU = torch.device("cpu")

_VERBOSE = 1


def challenge_entry(sample_path):
    """
    This is a baseline method.
    """
    print("\n" + "*"*100)
    msg = "   CPSC2021 challenge entry starts   ".center("#", "100")
    print("*"*100)
    start_time = time.time()
    timer = time.time()

    # all models are loaded into cpu
    # when using, move to gpu
    rpeak_model, rpeak_cfg = ECG_SEQ_LAB_NET_CPSC2021.from_checkpoint(
        os.path.join(_PROJECT_DIR, "saved_models", "BestModel_qrs_detection.pth.tar"),
        device=_CPU,
    )
    rr_lstm_model, rr_cfg = RR_LSTM_CPSC2021.from_checkpoint(
        os.path.join(_PROJECT_DIR, "saved_models", "BestModel_rr_lstm.pth.tar"),
        device=_CPU,
    )
    main_task_model, main_task_cfg = ECG_SEQ_LAB_NET_CPSC2021.from_checkpoint(
        os.path.join(_PROJECT_DIR, "saved_models", "BestModel_main_seq_lab.pth.tar")
    )  # TODO: consider unets

    _sample_path = os.path.splitext(sample_path)[0]
    try:
        rec = wfdb.rdrecord(sample_path, physical=True)
    except:
        rec = wfdb.rdrecord(_sample_path, physical=True)
    sig = np.asarray(wfdb_rec.p_signal.T)

    # preprocessing, e.g. resample, bandpass, normalization, etc.
    # finished, not checked,
    if main_task_cfg.fs != rec.fs:
        sig = SS.resample_poly(sig, main_task_cfg.fs, rec.fs, axis=1)
    if "baseline" in main_task_cfg:
        bl_win = [main_task_cfg.baseline_window1, main_task_cfg.baseline_window2]
    else:
        bl_win = None
    if "bandpass" in main_task_cfg:
        band_fs = main_task_cfg.filter_band
    sig = preprocess_multi_lead_signal(
        sig,
        fs=main_task_cfg.fs,
        bl_win=bl_win,
        band_fs=band_fs,
        verbose=_VERBOSE,
    )["filtered_ecg"]

    # slice data into segments for rpeak detection and main task
    # not finished, not checked,
    seglen = main_task_cfg.input_len
    overlap_len = main_task_cfg.overlap_len
    forward_len = seglen - overlap_len
    dl_input = np.array([]).reshape((main_task_cfg.n_leads, 0))

    for idx in range((siglen-self.seglen)//forward_len + 1):
        seg_data = sig[seglen*idx:seglen*(idx+1)]
        if main_task_cfg.random_normalize:  # to keep consistency of data distribution
            seg_data = normalize(
                sig=seg_data,
                mean=list(repeat(np.mean(main_task_cfg.random_normalize_mean), main_task_cfg.n_leads)),
                std=list(repeat(np.mean(main_task_cfg.random_normalize_std), main_task_cfg.n_leads)),
                per_channel=True,
            )
        # TODO: add tail
    # detect rpeaks
    # not finished, not checked,
    raw_rpeaks = _detect_rpeaks(
        model=rpeak_model,
        sig=dl_input,
        config=rpeak_cfg,
    )

    # rr_lstm
    # not finished, not checked,

    # main_task
    # not finished, not checked,

    # merge results from rr_lstm and main_task
    # not finished, not checked,


def _detect_rpeaks(model, sig, config):
    """ not finished, not checked,
    NOTE: sig are sliced data with overlap,
    hence DO NOT directly use model's inference method
    """
    try:
        model = model.to(_CUDA)
    except:
        pass
    _device = next(model.parameters()).device
    _dtype = next(model.parameters()).dtype
    sig = torch.as_tensor(sig, device=_device, dtype=_dtype)
    if sig.ndim == 2:
            sig = sig.unsqueeze(0)  # add a batch dimension
    # batch_size, channels, seq_len = sig.shape
    pred = self.forward(sig)
    pred = self.sigmoid(sig)
    pred = pred.cpu().detach().numpy().squeeze(-1)
    # TODO: finish this function
    raise NotImplementedError


def _rr_lstm(model, rr, config):
    """ finished, not checked,
    """
    try:
        model = model.to(_CUDA)
    except:
        pass
    # just use the model's inference method
    pred, af_episodes = model.inference(
        input=rr,
        bin_pred_thr=0.5,
        episode_len_thr=5,
    )
    af_episodes = af_episodes[0]
    return af_episodes


def _main_task(model, sig, config):
    """ not finished, not checked,
    """
    try:
        model = model.to(_CUDA)
    except:
        pass
    _device = next(model.parameters()).device
    _dtype = next(model.parameters()).dtype
    sig = torch.as_tensor(sig, device=_device, dtype=_dtype)
    if sig.ndim == 2:
            sig = sig.unsqueeze(0)  # add a batch dimension
    # batch_size, channels, seq_len = sig.shape
    pred = self.forward(sig)
    pred = self.sigmoid(sig)
    pred = pred.cpu().detach().numpy().squeeze(-1)
        

if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        
    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        pred_dict = challenge_entry(sample_path)

        save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)

