"""
"""
import os
from copy import deepcopy
from itertools import repeat

import numpy as np
from easydict import EasyDict as ED


__all__ = [
    "BaseCfg",
    "TrainCfg",
    "ModelCfg",
    "PlotCfg",
]


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


BaseCfg = ED()
BaseCfg.db_dir = "/home/taozi/Data/CPSC2021/"
BaseCfg.log_dir = os.path.join(_BASE_DIR, "log")
BaseCfg.model_dir = os.path.join(_BASE_DIR, "saved_models")
os.makedirs(BaseCfg.log_dir, exist_ok=True)
os.makedirs(BaseCfg.model_dir, exist_ok=True)
BaseCfg.test_data_dir = os.path.join(_BASE_DIR, "test_data")
BaseCfg.fs = 200
BaseCfg.torch_dtype = "float"  # "double"

BaseCfg.class_fn2abbr = { # fullname to abbreviation
    "non atrial fibrillation": "N",
    "paroxysmal atrial fibrillation": "AFp",
    "persistent atrial fibrillation": "AFf",
}
BaseCfg.class_abbr2fn = {v:k for k,v in BaseCfg.class_fn2abbr.items()}
BaseCfg.class_fn_map = { # fullname to number
    "non atrial fibrillation": 0,
    "paroxysmal atrial fibrillation": 2,
    "persistent atrial fibrillation": 1,
}
BaseCfg.class_abbr_map = {k: BaseCfg.class_fn_map[v] for k,v in BaseCfg.class_abbr2fn.items()}


TrainCfg = ED()


ModelCfg = ED()


# configurations for visualization
PlotCfg = ED()
# default const for the plot function in dataset.py
# used only when corr. values are absent
# all values are time bias w.r.t. corr. peaks, with units in ms
PlotCfg.p_onset = -40
PlotCfg.p_offset = 40
PlotCfg.q_onset = -20
PlotCfg.s_offset = 40
PlotCfg.qrs_radius = 60
PlotCfg.t_onset = -100
PlotCfg.t_offset = 60
