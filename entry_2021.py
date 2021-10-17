#!/usr/bin/env python3

import numpy as np
import os
import sys
import time

import wfdb
import numpy as np
import torch
import scipy.signal as SS
from easydict import EasyDict as ED

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
    rpeak_cfg = ED(rpeak_cfg)
    rr_lstm_model, rr_cfg = RR_LSTM_CPSC2021.from_checkpoint(
        os.path.join(_PROJECT_DIR, "saved_models", "BestModel_rr_lstm.pth.tar"),
        device=_CPU,
    )
    rr_cfg = ED(rr_cfg)
    main_task_model, main_task_cfg = ECG_SEQ_LAB_NET_CPSC2021.from_checkpoint(
        os.path.join(_PROJECT_DIR, "saved_models", "BestModel_main_seq_lab.pth.tar")
    )  # TODO: consider unets
    main_task_cfg = ED(main_task_cfg)

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
    # finished, not checked,
    seglen = main_task_cfg.input_len
    overlap_len = 4 * main_task_cfg.fs
    forward_len = seglen - overlap_len
    dl_input = np.array([]).reshape((main_task_cfg.n_leads, 0))

    for idx in range((sig.shape[1]-seglen)//forward_len + 1):
        seg_data = sig[forward_len*idx: forward_len*idx+seglen]
        if main_task_cfg.random_normalize:  # to keep consistency of data distribution
            seg_data = normalize(
                sig=seg_data,
                mean=list(repeat(np.mean(main_task_cfg.random_normalize_mean), main_task_cfg.n_leads)),
                std=list(repeat(np.mean(main_task_cfg.random_normalize_std), main_task_cfg.n_leads)),
                per_channel=True,
            )
        dl_input = np.concatenate((dl_input, seg_data))
    # add tail
    if sig.shape[1] > seglen:
        seg_data = sig[max(0,sig.shape[1]-seglen):sig.shape[1]]
        if main_task_cfg.random_normalize:  # to keep consistency of data distribution
            seg_data = normalize(
                sig=seg_data,
                mean=list(repeat(np.mean(main_task_cfg.random_normalize_mean), main_task_cfg.n_leads)),
                std=list(repeat(np.mean(main_task_cfg.random_normalize_std), main_task_cfg.n_leads)),
                per_channel=True,
            )
        dl_input = np.concatenate((dl_input, seg_data))

    # detect rpeaks
    # finished, not checked,
    rpeaks = _detect_rpeaks(
        model=rpeak_model,
        sig=dl_input,
        siglen=sig.shape[1],
        overlap_len=overlap_len,
        config=rpeak_cfg,
    )

    # rr_lstm
    # not finished, not checked,
    rr_pred = _rr_lstm(
        model=rr,
        rpeaks=rpeaks,
        config=rr_cfg,
    )

    # main_task
    # not finished, not checked,
    main_pred = _main_task(
        model=main_task_model,
        sig=dl_input,
        siglen=sig.shape[1],
        overlap_len=overlap_len,
        config=main_task_cfg,
    )

    # merge results from rr_lstm and main_task
    # not finished, not checked,


def _detect_rpeaks(model, sig, siglen, overlap_len, config):
    """ finished, not checked,

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

    # merge the prob array
    seglen = config.input_len
    qua_overlap_len = overlap_len // 4
    forward_len = seglen - overlap_len
    merged_pred = np.zeros((siglen,))
    if pred.shape[0] > 1:
        merged_pred[:seglen-qua_overlap_len] = pred[0,:seglen-qua_overlap_len]
        merged_pred[siglen-(seglen-qua_overlap_len):] = pred[-1,qua_overlap_len:]
        for idx in range(1,pred.shape[0]-1):
            to_compare = np.zeros((siglen,))
            start_idx = forward_len*idx-qua_overlap_len
            end_idx = forward_len*idx+seglen-qua_overlap_len
            to_compare[start_idx: end_idx] = pred[idx,qua_overlap_len:seglen-qua_overlap_len]
            merged_pred = np.maximum(merged_pred, to_compare,)
        # tail
        to_compare = np.zeros((siglen,))
        start_idx = forward_len*(pred.shape[0]-2) + seglen - qua_overlap_len
        to_compare[start_idx:] = pred[-1, siglen-start_idx:]
        merged_pred = np.maximum(merged_pred, to_compare,)
    else:
        merged_pred = pred[0,...]
    
    rpeaks = _qrs_detection_post_process(
        pred=merged_pred,
        fs=config.fs, 
        reduction=config.reduction,
        bin_pred_thr=0.5
    )

    return rpeaks


def _rr_lstm(model, rpeaks, config):
    """ finished, not checked,
    """
    try:
        model = model.to(_CUDA)
    except:
        pass
    rr = np.diff(rpeaks) / config.fs
    # just use the model's inference method
    pred, af_episodes = model.inference(
        input=rr,
        bin_pred_thr=0.5,
        episode_len_thr=5,
    )
    af_episodes = af_episodes[0]
    return af_episodes


def _main_task(model, sig, siglen, overlap_len, config):
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

