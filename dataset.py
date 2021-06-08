"""
data generator for feeding data into pytorch models
"""
import os, sys
import json
import time
import textwrap
from random import shuffle, randint
from copy import deepcopy
from typing import Union, Optional, List, Tuple, Dict, Sequence, Set, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
from easydict import EasyDict as ED
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler

from cfg import (
    TrainCfg, ModelCfg,
)
from data_reader import CPSC2021Reader as CR
from utils.utils_signal import ensure_siglen, butter_bandpass_filter
from utils.misc import dict_to_str, list_sum


if ModelCfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "CPSC2021",
]


class CPSC2021(Dataset):
    """
    """
    __DEBUG__ = False
    __name__ = "CPSC2021"
    
    def __init__(self, config:ED, training:bool=True) -> NoReturn:
        """ NOT finished, NOT checked,

        Parameters
        ----------
        config: dict,
            configurations for the Dataset,
            ref. `cfg.TrainCfg`
        training: bool, default True,
            if True, the training set will be loaded, otherwise the test set
        """
        super().__init__()
        raise NotImplementedError


    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray]:
        """ finished, checked,
        """
        raise NotImplementedError

        # return values, labels


    def __len__(self) -> int:
        """
        """
        raise NotImplementedError


    def disable_data_augmentation(self) -> NoReturn:
        """
        """
        self.__data_aug = False

    
    def _train_test_split(self,
                          train_ratio:float=0.8,
                          force_recompute:bool=False) -> List[str]:
        """ finished, checked,

        do train test split,
        it is ensured that both the train and the test set contain all classes

        Parameters
        ----------
        train_ratio: float, default 0.8,
            ratio of the train set in the whole dataset
        force_recompute: bool, default False,
            if True, force redo the train-test split,
            regardless of the existing ones stored in json files

        Returns
        -------
        records: list of str,
            list of the records split for training or validation
        """
        raise NotImplementedError
