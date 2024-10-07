"""

References:
-----------
wfdb
biosppy

"""

from .ecg_denoise import ecg_denoise_naive, remove_spikes_naive
from .ecg_preproc import merge_rpeaks, preprocess_multi_lead_signal, preprocess_single_lead_signal, rpeaks_detect_multi_leads
from .ecg_rpeaks import (
    christov_detect,
    engzee_detect,
    gamboa_detect,
    gqrs_detect,
    hamilton_detect,
    pantompkins_detect,
    ssf_detect,
    xqrs_detect,
)
from .ecg_rpeaks_dl import seq_lab_net_detect

# __all__ = [s for s in dir() if not s.startswith('_')]

__all__ = [
    # preprocessing pipelines
    "preprocess_multi_lead_signal",
    "preprocess_single_lead_signal",
    "rpeaks_detect_multi_leads",
    "merge_rpeaks",
    # traditional rpeak detection methods
    "xqrs_detect",
    "gqrs_detect",
    "pantompkins_detect",
    "hamilton_detect",
    "ssf_detect",
    "christov_detect",
    "engzee_detect",
    "gamboa_detect",
    # deep learning rpeak detection
    "seq_lab_net_detect",
    # denoise methods
    "remove_spikes_naive",
    "ecg_denoise_naive",
]
