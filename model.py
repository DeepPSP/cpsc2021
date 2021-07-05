"""

Possible Solutions
------------------
1. segmentation (AF, non-AF) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
2. sequence labelling (AF, non-AF) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
3. per-beat (R peak detection first) classification (CNN, etc. + RR LSTM) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
4. object detection (? onsets and offsets)
"""
from copy import deepcopy
from typing import Union, Optional, Sequence, Tuple, NoReturn, Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from easydict import EasyDict as ED

# models from torch_ecg
from torch_ecg.torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.torch_ecg.models.ecg_seq_lab_net import ECG_SEQ_LAB_NET
from torch_ecg.torch_ecg.models.unets import ECG_UNET, ECG_SUBTRACT_UNET
from torch_ecg.torch_ecg.models.rr_lstm import RR_LSTM
from cfg import ModelCfg


__all__ = [
    "ECG_SEQ_LAB_NET_CPSC2021",
    "ECG_UNET_CPSC2021",
    "ECG_SUBTRACT_UNET_CPSC2021",
    "RR_LSTM_CPSC2021",
]


class ECG_SEQ_LAB_NET_CPSC2021(ECG_SEQ_LAB_NET):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_SEQ_LAB_NET_CPSC2021"

    def __init__(self, classes:Sequence[str], n_leads:int, config:Optional[ED]=None) -> NoReturn:
        """ NOT finished, NOT checked,

        Parameters
        ----------
        classes: list,
            list of the classes for sequence labeling
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        super().__init__()
        raise NotImplementedError

    @torch.no_grad()
    def inference(self,
                  input:Union[np.ndarray,Tensor],
                  class_names:bool=False,
                  bin_pred_thr:float=0.5) -> Any:
        """ NOT finished, NOT checked,

        auxiliary function to `forward`, for CPSC2021,
        """
        raise NotImplementedError

    @torch.no_grad()
    def inference_CPSC2021(self,
                           input:Union[np.ndarray,Tensor],
                           class_names:bool=False,
                           bin_pred_thr:float=0.5) -> Any:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr)


class ECG_UNET_CPSC2021(ECG_UNET):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_UNET_CPSC2021"
    
    def __init__(self, classes:Sequence[str], n_leads:int, config:dict) -> NoReturn:
        """ NOT finished, NOT checked,

        Parameters
        ----------
        classes: sequence of int,
            name of the classes
        n_leads: int,
            number of input leads (number of input channels)
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        super().__init__()
        raise NotImplementedError

    @torch.no_grad()
    def inference(self,
                  input:Union[np.ndarray,Tensor],
                  class_names:bool=False,
                  bin_pred_thr:float=0.5) -> Any:
        """ NOT finished, NOT checked,

        auxiliary function to `forward`, for CPSC2021,
        """
        raise NotImplementedError

    @torch.no_grad()
    def inference_CPSC2021(self,
                           input:Union[np.ndarray,Tensor],
                           class_names:bool=False,
                           bin_pred_thr:float=0.5) -> Any:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr)


class ECG_SUBTRACT_UNET_CPSC2021(ECG_SUBTRACT_UNET):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_SUBTRACT_UNET_CPSC2021"

    def __init__(self, classes:Sequence[str], n_leads:int, config:dict) -> NoReturn:
        """ NOT finished, NOT checked,

        Parameters
        ----------
        classes: sequence of int,
            name of the classes
        n_leads: int,
            number of input leads
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        super().__init__()
        raise NotImplementedError

    @torch.no_grad()
    def inference(self,
                  input:Union[np.ndarray,Tensor],
                  class_names:bool=False,
                  bin_pred_thr:float=0.5) -> Any:
        """ NOT finished, NOT checked,

        auxiliary function to `forward`, for CPSC2021,
        """
        raise NotImplementedError

    @torch.no_grad()
    def inference_CPSC2021(self,
                           input:Union[np.ndarray,Tensor],
                           class_names:bool=False,
                           bin_pred_thr:float=0.5) -> Any:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr)


class RR_LSTM_CPSC2021(RR_LSTM):
    """
    """
    __DEBUG__ = True
    __name__ = "RR_LSTM_CPSC2021"

    def __init__(self, classes:Sequence[str], n_leads:int, config:Optional[ED]=None) -> NoReturn:
        """ NOT finished, NOT checked,

        Parameters
        ----------
        classes: list,
            list of the classes for classification
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        super().__init__()
        raise NotImplementedError

    @torch.no_grad()
    def inference(self,
                  input:Union[np.ndarray,Tensor],
                  class_names:bool=False,
                  bin_pred_thr:float=0.5) -> Any:
        """ NOT finished, NOT checked,

        auxiliary function to `forward`, for CPSC2021,
        """
        raise NotImplementedError

    @torch.no_grad()
    def inference_CPSC2021(self,
                           input:Union[np.ndarray,Tensor],
                           class_names:bool=False,
                           bin_pred_thr:float=0.5) -> Any:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr)
