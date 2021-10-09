"""

Possible Solutions
------------------
1. segmentation (AF, non-AF) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
2. sequence labelling (AF, non-AF) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
3. per-beat (R peak detection first) classification (CNN, etc. + RR LSTM) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
4. object detection (? onsets and offsets)
"""
from copy import deepcopy
from numbers import Real
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
from signal_processing.ecg_preproc import merge_rpeaks


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
        super().__init__(classes, n_leads, config)
        self.task = config.task

    @torch.no_grad()
    def inference(self,
                  input:Union[np.ndarray,Tensor],
                  bin_pred_thr:float=0.5,
                  **kwargs:Any) -> Any:
        """ NOT finished, NOT checked,
        """
        if self.task == "qrs_detection":
            return self._inference_qrs_detection(input, bin_pred_thr, **kwargs)
        elif self.task == "main":
            return self._inference_main_task(input, bin_pred_thr, **kwargs)

    @torch.no_grad()
    def inference_CPSC2021(self,
                           input:Union[np.ndarray,Tensor],
                           bin_pred_thr:float=0.5,
                           **kwargs:Any) -> Any:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr, **kwargs)

    @torch.no_grad()
    def _inference_qrs_detection(self,
                                 input:Union[np.ndarray,Tensor],
                                 bin_pred_thr:float=0.5,
                                 duration_thr:int=4*16,
                                 dist_thr:Union[int,Sequence[int]]=200,) -> Tuple[np.ndarray, List[np.ndarray]]:
        """ NOT finished, NOT checked,
        auxiliary function to `forward`, for CPSC2021,

        NOTE: each segment of input be better filtered using `_remove_spikes_naive`,
        and normalized to a suitable mean and std

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        duration_thr: int, default 4*16,
            minimum duration for a "true" qrs complex, units in ms
        dist_thr: int or sequence of int, default 200,
            if is sequence of int,
            (0-th element). minimum distance for two consecutive qrs complexes, units in ms;
            (1st element).(optional) maximum distance for checking missing qrs complexes, units in ms,
            e.g. [200, 1200]
            if is int, then is the case of (0-th element).

        Returns
        -------
        pred: ndarray,
            the array of scalar predictions
        rpeaks: list of ndarray,
            list of rpeak indices for each batch element
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.to(device)
        batch_size, channels, seq_len = input.shape
        if isinstance(input, np.ndarray):
            _input = torch.from_numpy(input).to(device)
        else:
            _input = input.to(device)
        pred = self.forward(_input)
        pred = self.sigmoid(pred)
        pred = pred.cpu().detach().numpy().squeeze(-1)

        # prob --> qrs mask --> qrs intervals --> rpeaks
        rpeaks = _qrs_detection_post_process(
            pred=pred,
            fs=self.config.fs,
            reduction=self.config.reduction,
            bin_pred_thr=bin_pred_thr,
            duration_thr=duration_thr,
            dist_thr=dist_thr
        )

        return pred, rpeaks

    @torch.no_grad()
    def _inference_main_task(self,
                             input:Union[np.ndarray,Tensor],
                             bin_pred_thr:float=0.5) -> Any:
        """
        """
        raise NotImplementedError


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
        super().__init__(classes, n_leads, config)
        self.task = config.task

    @torch.no_grad()
    def inference(self,
                  input:Union[np.ndarray,Tensor],
                  bin_pred_thr:float=0.5,
                  **kwargs:Any) -> Any:
        """ NOT finished, NOT checked,
        """
        if self.task == "qrs_detection":
            return self._inference_qrs_detection(input, bin_pred_thr, **kwargs)
        elif self.task == "main":
            return self._inference_main_task(input, bin_pred_thr, **kwargs)

    @torch.no_grad()
    def inference_CPSC2021(self,
                           input:Union[np.ndarray,Tensor],
                           bin_pred_thr:float=0.5,
                           **kwargs:Any) -> Any:
        """
        alias for `self.inference`
        """
        return self.inference(input, bin_pred_thr, **kwargs)

    @torch.no_grad()
    def _inference_qrs_detection(self,
                                 input:Union[np.ndarray,Tensor],
                                 bin_pred_thr:float=0.5,
                                 duration_thr:int=4*16,
                                 dist_thr:Union[int,Sequence[int]]=200,) -> Tuple[np.ndarray, List[np.ndarray]]:
        """ NOT finished, NOT checked,
        auxiliary function to `forward`, for CPSC2021,

        NOTE: each segment of input be better filtered using `_remove_spikes_naive`,
        and normalized to a suitable mean and std

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        duration_thr: int, default 4*16,
            minimum duration for a "true" qrs complex, units in ms
        dist_thr: int or sequence of int, default 200,
            if is sequence of int,
            (0-th element). minimum distance for two consecutive qrs complexes, units in ms;
            (1st element).(optional) maximum distance for checking missing qrs complexes, units in ms,
            e.g. [200, 1200]
            if is int, then is the case of (0-th element).

        Returns
        -------
        pred: ndarray,
            the array of scalar predictions
        rpeaks: list of ndarray,
            list of rpeak indices for each batch element
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.to(device)
        batch_size, channels, seq_len = input.shape
        if isinstance(input, np.ndarray):
            _input = torch.from_numpy(input).to(device)
        else:
            _input = input.to(device)
        pred = self.forward(_input)
        pred = self.sigmoid(pred)
        pred = pred.cpu().detach().numpy().squeeze(-1)

        # prob --> qrs mask --> qrs intervals --> rpeaks
        rpeaks = _qrs_detection_post_process(
            pred=pred,
            fs=self.config.fs,
            reduction=1,
            bin_pred_thr=bin_pred_thr,
            duration_thr=duration_thr,
            dist_thr=dist_thr
        )

        return pred, rpeaks

    @torch.no_grad()
    def _inference_main_task(self,
                             input:Union[np.ndarray,Tensor],
                             bin_pred_thr:float=0.5,) -> Any:
        """
        """
        raise NotImplementedError


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
        super().__init__(classes, n_leads, config)
        self.task = config.task

    @torch.no_grad()
    def inference(self,
                  input:Union[np.ndarray,Tensor],
                  bin_pred_thr:float=0.5,
                  **kwargs:Any) -> Any:
        """ NOT finished, NOT checked,
        """
        if self.task == "qrs_detection":
            return self._inference_qrs_detection(input, bin_pred_thr, **kwargs)
        elif self.task == "main":
            return self._inference_main_task(input, bin_pred_thr, **kwargs)

    @torch.no_grad()
    def inference_CPSC2021(self,
                           input:Union[np.ndarray,Tensor],
                           bin_pred_thr:float=0.5,
                           **kwargs:Any,) -> Any:
        """
        alias for `self.inference`
        """
        return self.inference(input, bin_pred_thr, **kwargs)

    @torch.no_grad()
    def _inference_qrs_detection(self,
                                 input:Union[np.ndarray,Tensor],
                                 bin_pred_thr:float=0.5,
                                 duration_thr:int=4*16,
                                 dist_thr:Union[int,Sequence[int]]=200,) -> Tuple[np.ndarray, List[np.ndarray]]:
        """ NOT finished, NOT checked,
        auxiliary function to `forward`, for CPSC2021,

        NOTE: each segment of input be better filtered using `_remove_spikes_naive`,
        and normalized to a suitable mean and std

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        duration_thr: int, default 4*16,
            minimum duration for a "true" qrs complex, units in ms
        dist_thr: int or sequence of int, default 200,
            if is sequence of int,
            (0-th element). minimum distance for two consecutive qrs complexes, units in ms;
            (1st element).(optional) maximum distance for checking missing qrs complexes, units in ms,
            e.g. [200, 1200]
            if is int, then is the case of (0-th element).

        Returns
        -------
        pred: ndarray,
            the array of scalar predictions
        rpeaks: list of ndarray,
            list of rpeak indices for each batch element
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.to(device)
        batch_size, channels, seq_len = input.shape
        if isinstance(input, np.ndarray):
            _input = torch.from_numpy(input).to(device)
        else:
            _input = input.to(device)
        pred = self.forward(_input)
        pred = self.sigmoid(pred)
        pred = pred.cpu().detach().numpy().squeeze(-1)

        # prob --> qrs mask --> qrs intervals --> rpeaks
        rpeaks = _qrs_detection_post_process(
            pred=pred,
            fs=self.config.fs,
            reduction=1,
            bin_pred_thr=bin_pred_thr,
            duration_thr=duration_thr,
            dist_thr=dist_thr
        )

        return pred, rpeaks

    @torch.no_grad()
    def _inference_main_task(self,
                             input:Union[np.ndarray,Tensor],
                             bin_pred_thr:float=0.5,) -> Any:
        """
        """
        raise NotImplementedError


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
        super().__init__(classes, n_leads, config)

    @torch.no_grad()
    def inference(self,
                  input:Union[np.ndarray,Tensor],
                  bin_pred_thr:float=0.5,) -> Any:
        """ NOT finished, NOT checked,
        """
        raise NotImplementedError

    @torch.no_grad()
    def inference_CPSC2021(self,
                           input:Union[np.ndarray,Tensor],
                           bin_pred_thr:float=0.5,) -> Any:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr)


def _qrs_detection_post_process(pred:np.ndarray,
                                fs:Real,
                                reduction:int,
                                bin_pred_thr:float=0.5,
                                skip_dist:
                                duration_thr:int=4*16,
                                dist_thr:Union[int,Sequence[int]]=200,) -> List[np.ndarray]:
    """ finished, NOT checked,
    """
    batch_size, prob_arr_len = pred.shape
    model_spacing = 1000 / fs  # units in ms
    model_granularity = reduction
    input_len = model_granularity * prob_arr_len
    _duration_thr = duration_thr / model_spacing / model_granularity
    _dist_thr = [dist_thr] if isinstance(dist_thr, int) else dist_thr
    assert len(_dist_thr) <= 2

    # mask = (pred > bin_pred_thr).astype(int)
    rpeaks = []
    for b_idx in range(batch_size):
        b_prob = pred[b_idx,...]
        b_mask = (b_prob > bin_pred_thr).astype(int)
        b_qrs_intervals = mask_to_intervals(b_mask, 1)
        b_rpeaks = np.array([itv[0]+itv[1] for itv in b_qrs_intervals if itv[1]-itv[0] >= _duration_thr])
        b_rpeaks = (model_granularity//2) * b_rpeaks
        # print(f"before post-process, b_qrs_intervals = {b_qrs_intervals}")
        # print(f"before post-process, b_rpeaks = {b_rpeaks}")

        check = True
        dist_thr_inds = _dist_thr[0] / model_spacing
        while check:
            check = False
            b_rpeaks_diff = np.diff(b_rpeaks)
            for r in range(len(b_rpeaks_diff)):
                if b_rpeaks_diff[r] < dist_thr_inds:  # 200 ms
                    prev_r_ind = int(b_rpeaks[r]/model_granularity)  # ind in _prob
                    next_r_ind = int(b_rpeaks[r+1]/model_granularity)  # ind in _prob
                    if b_prob[prev_r_ind] > b_prob[next_r_ind]:
                        del_ind = r+1
                    else:
                        del_ind = r
                    b_rpeaks = np.delete(b_rpeaks, del_ind)
                    check = True
                    break
        if len(_dist_thr) == 1:
            b_rpeaks = b_rpeaks[np.where((b_rpeaks>=skip_dist) & (b_rpeaks<input_len-skip_dist))[0]]
            rpeaks.append(b_rpeaks)
            continue
        check = True
        # TODO: parallel the following block
        # CAUTION !!! 
        # this part is extremely slow in some cases (long duration and low SNR)
        dist_thr_inds = _dist_thr[1] / model_spacing
        while check:
            check = False
            b_rpeaks_diff = np.diff(b_rpeaks)
            for r in range(len(b_rpeaks_diff)):
                if b_rpeaks_diff[r] >= dist_thr_inds:  # 1200 ms
                    prev_r_ind = int(b_rpeaks[r]/model_granularity)  # ind in _prob
                    next_r_ind = int(b_rpeaks[r+1]/model_granularity)  # ind in _prob
                    prev_qrs = [itv for itv in b_qrs_intervals if itv[0]<=prev_r_ind<=itv[1]][0]
                    next_qrs = [itv for itv in b_qrs_intervals if itv[0]<=next_r_ind<=itv[1]][0]
                    check_itv = [prev_qrs[1], next_qrs[0]]
                    l_new_itv = mask_to_intervals(b_mask[check_itv[0]: check_itv[1]], 1)
                    if len(l_new_itv) == 0:
                        continue
                    l_new_itv = [[itv[0]+check_itv[0], itv[1]+check_itv[0]] for itv in l_new_itv]
                    new_itv = max(l_new_itv, key=lambda itv: itv[1]-itv[0])
                    new_max_prob = (b_prob[new_itv[0]:new_itv[1]]).max()
                    for itv in l_new_itv:
                        itv_prob = (b_prob[itv[0]:itv[1]]).max()
                        if itv[1] - itv[0] == new_itv[1] - new_itv[0] and itv_prob > new_max_prob:
                            new_itv = itv
                            new_max_prob = itv_prob
                    b_rpeaks = np.insert(b_rpeaks, r+1, 4*(new_itv[0]+new_itv[1]))
                    check = True
                    break
        b_rpeaks = b_rpeaks[np.where((b_rpeaks>=skip_dist) & (b_rpeaks<input_len-skip_dist))[0]]
        rpeaks.append(b_rpeaks)
    return rpeaks
