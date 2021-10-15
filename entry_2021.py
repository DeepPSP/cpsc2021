#!/usr/bin/env python3

import numpy as np
import os
import sys

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


def challenge_entry(sample_path):
    """
    This is a baseline method.
    """
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
    if main_task_cfg.fs != rec.fs:
        sig = SS.resample_poly(sig, main_task_cfg.fs, rec.fs, axis=1)

    # slice data into segments for rpeak detection and main task

    # detect rpeaks

    # rr_lstm

    # main_task

    # merge results from rr_lstm and main_task


def _detect_rpeaks(model, sig, config):
    """
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

    raise NotImplementedError


def _rr_lstm(model, rr, config):
    """
    """
    raise NotImplementedError


def _main_task(model, sig, config):
    """
    """
    raise NotImplementedError
        

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

