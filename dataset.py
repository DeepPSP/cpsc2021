"""
data generator for feeding data into pytorch models
"""
import os, sys
import json
import time
from random import shuffle, randint, sample, uniform
from copy import deepcopy
from typing import Union, Optional, List, Tuple, Dict, Sequence, Set, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
from scipy import signal as SS
from easydict import EasyDict as ED
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat, savemat

from cfg import (
    TrainCfg, ModelCfg,
)
from data_reader import CPSC2021Reader as CR
from signal_processing.ecg_preproc import preprocess_multi_lead_signal
from signal_processing.ecg_denoise import ecg_denoise_naive
from utils.utils_signal import ensure_siglen, butter_bandpass_filter, get_ampl
from utils.utils_interval import mask_to_intervals
from utils.misc import (
    dict_to_str, list_sum, nildent,
    get_record_list_recursive3,
)


if ModelCfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "CPSC2021",
]


_BASE_DIR = os.path.dirname(__file__)


class CPSC2021(Dataset):
    """
    1. ECGs are preprocessed and stored in one folder
    2. preprocessed ECGs are sliced with overlap to generate data and label for different tasks:
        the data files stores segments of fixed length of preprocessed ECGs,
        the annotation files contain "qrs_mask", and "af_mask"
    """
    __DEBUG__ = False
    __name__ = "CPSC2021"
    
    def __init__(self, config:ED, task:str, training:bool=True) -> NoReturn:
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
        self.config = deepcopy(config)
        self.reader = CR(db_dir=config.db_dir)
        if ModelCfg.torch_dtype.lower() == "double":
            self.dtype = np.float64
        else:
            self.dtype = np.float32
        self.allowed_preproc = self.config.preproc

        self.training = training
        self.__data_aug = self.training

        # create directories if needed
        # preprocess_dir stores pre-processed signals
        self.preprocess_dir = os.path.join(config.db_dir, "preprocessed")
        os.makedirs(self.preprocess_dir, exist_ok=True)
        # segments_dir for sliced segments
        self.base_segments_dir = os.path.join(config.db_dir, "segments")
        os.makedirs(self.base_segments_dir, exist_ok=True)
        self.segment_ext = "mat"

        self.__set_task(task)

    def __set_task(self, task:str) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        task: str,
            name of the task, can be one of `TrainCfg.tasks`
        """
        assert task.lower() in TrainCfg.tasks, f"illegal task \042{task}\042"
        if hasattr(self, "task") and self.task == task.lower():
            return
        self.task = task.lower()
        self.all_classes = self.config[task].classes
        self.n_classes = len(self.config[task].classes)

        self.seglen = self.config[task].input_len  # alias, for simplicity
        split_res = self._train_test_split(
            train_ratio=self.config.train_ratio,
            force_recompute=False,
        )

        if self.task in ["qrs_detection", "main",]:
            # for qrs detection, or for the main task
            self.segments_dirs = ED()
            self.__all_segments = ED()
            self.segments_json = os.path.join(self.base_segments_dir, "segments.json")
            self._ls_segments()

            if self.training:
                self.segments = list_sum([self.__all_segments[subject] for subject in split_res.train])
                shuffle(self.segments)
            else:
                self.segments = list_sum([self.__all_segments[subject] for subject in split_res.test])
        elif self.task.lower() in ["rr_lstm",]:
            pass
        else:
            raise NotImplementedError(f"data generator for task \042{self.task}\042 not implemented")

    def reset_task(self, task:str) -> NoReturn:
        """ finished, checked,
        """
        self.__set_task(task)

    def _ls_segments(self) -> NoReturn:
        """ finished, checked,

        list all the segments
        """
        for item in ["data", "ann"]:
            self.segments_dirs[item] = ED()
            for s in self.reader.all_subjects:
                self.segments_dirs[item][s] = os.path.join(self.base_segments_dir, item, s)
                os.makedirs(self.segments_dirs[item][s], exist_ok=True)
        if os.path.isfile(self.segments_json):
            with open(self.segments_json, "r") as f:
                self.__all_segments = json.load(f)
            return
        print(f"please allow the reader a few minutes to collect the segments from {self.base_segments_dir}...")
        seg_filename_pattern = f"S_\d{{1,3}}_\d{{1,2}}_\d{{7}}{self.reader.rec_ext}"
        self.__all_segments = ED({
            s: get_record_list_recursive3(self.segments_dirs.data[s], seg_filename_pattern) \
                for s in self.reader.all_subjects
        })
        if all([len(self.__all_segments[s])>0 for s in self.reader.all_subjects]):
            with open(self.segments_json, "w") as f:
                json.dump(self.__all_segments, f)

    @property
    def all_segments(self) -> ED:
        return self.__all_segments

    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray]:
        """ NOT finished, NOT checked,
        """
        seg_name = self.segments[index]
        seg_data = self._load_seg_data(seg_name)
        # TODO:
        raise NotImplementedError

    def __len__(self) -> int:
        """
        """
        return len(self.segments)

    def _get_seg_data_path(self, seg:str) -> str:
        """ finished, checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_1_1_0000193"

        Returns
        -------
        fp: str,
            path of the data file of the segment
        """
        subject = seg.split("_")[1]
        fp = os.path.join(self.segments_dirs.data[subject], f"{seg}.{self.segment_ext}")
        return fp

    def _get_seg_ann_path(self, seg:str) -> str:
        """ finished, checked,
        Parameters:
        -----------
        seg: str,
            name of the segment, of pattern like "S_1_1_0000193"
        Returns:
        --------
        fp: str,
            path of the annotation file of the segment
        """
        subject = seg.split("_")[1]
        fp = os.path.join(self.segments_dirs.ann[subject], f"{seg}.{self.segment_ext}")
        return fp

    def _load_seg_data(self, seg:str) -> np.ndarray:
        """ finished, checked,
        Parameters:
        -----------
        seg: str,
            name of the segment, of pattern like "S_1_1_0000193"
        Returns:
        --------
        seg_data: ndarray,
            data of the segment, of shape (2, `self.seglen`)
        """
        seg_data_fp = self._get_seg_data_path(seg)
        seg_data = loadmat(seg_data_fp)["ecg"]
        return seg_data

    def _load_seg_mask(self, seg:str) -> np.ndarray:
        """ finished, NOT checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_1_1_0000193"

        Returns
        -------
        seg_mask: np.ndarray,
            label of the sequence,
            of shape (self.seglen, self.n_classes)
        """
        seg_mask_fp = self._get_seg_ann_path(seg)
        if self.task in ["qrs_detection",]:
            seg_mask = loadmat(seg_mask_fp)["qrs_mask"]
        elif self.task in ["main",]:
            seg_mask = loadmat(seg_mask_fp)["af_mask"]
        return seg_mask

    def _load_seg_seq_lab(self, seg:str, reduction:int=8) -> np.ndarray:
        """ finished, NOT checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_1_1_0000193"
        reduction: int, default 8,
            reduction (granularity) of length of the model output,
            compared to the original signal length

        Returns
        -------
        seq_lab: np.ndarray,
            label of the sequence,
            of shape (self.seglen//reduction, self.n_classes)
        """
        seg_mask = self._load_seg_mask(seg)
        seg_len, n_classes = seg_mask.shape
        seq_lab = np.array([
            np.mean(seg_mask[reduction*idx:reduction*(idx+1)],axis=0,keepdims=True).astype(int) \
                for idx in range(seg_len//reduction)
        ])
        return seq_lab

    def disable_data_augmentation(self) -> NoReturn:
        """
        """
        self.__data_aug = False

    def enable_data_augmentation(self) -> NoReturn:
        """
        """
        self.__data_aug = True

    def persistence(self, force_recompute:bool=False, verbose:int=0) -> NoReturn:
        """ NOT finished, NOT checked,

        make the dataset persistent w.r.t. the ratios in `self.config`

        Parameters
        ----------
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        verbose: int, default 0,
            print verbosity
        """
        self._preprocess_data(
            self.allowed_preproc,
            force_recompute=force_recompute,
            verbose=verbose,
        )
        self._slice_data(
            force_recompute=force_recompute,
            verbose=verbose,
        )

    def _preprocess_data(self, preproc:List[str], force_recompute:bool=False, verbose:int=0) -> NoReturn:
        """ finished, checked,

        preprocesses the ecg data in advance for further use,
        offline for `self.persistence`

        Parameters
        ----------
        preproc: list of str,
            type of preprocesses to perform,
            should be sublist of `self.allowed_preproc`
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        verbose: int, default 0,
            print verbosity
        """
        preproc = self._normalize_preprocess_names(preproc, True)
        for idx, rec in enumerate(self.reader.all_records):
            self._preprocess_one_record(
                rec=rec,
                preproc=preproc,
                force_recompute=force_recompute,
                verbose=verbose,
            )
            if verbose >= 1:
                print(f"{idx+1}/{len(self.reader.all_records)} records", end="\r")

    def _preprocess_one_record(self, rec:str, preproc:List[str], force_recompute:bool=False, verbose:int=0) -> NoReturn:
        """ finished, checked,

        preprocesses the ecg data in advance for further use,
        offline for `self.persistence`

        Parameters
        ----------
        rec: str,
            filename of the record
        preproc: list of str,
            type of preprocesses to perform,
            should be sublist of `self.allowed_preproc`
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        verbose: int, default 0,
            print verbosity
        """
        suffix = self._get_rec_suffix(preproc)
        save_fp = os.path.join(self.preprocess_dir, f"{rec}-{suffix}.{self.segment_ext}")
        if (not force_recompute) and os.path.isfile(save_fp):
            return
        # perform pre-process
        if "baseline" in preproc:
            bl_win = [self.config.baseline_window1, self.config.baseline_window2]
        else:
            bl_win = None
        if "bandpass" in preproc:
            band_fs = self.config.filter_band
        pps = preprocess_multi_lead_signal(
            self.reader.load_data(rec),
            fs=self.reader.fs,
            bl_win=bl_win,
            band_fs=band_fs,
            verbose=verbose,
        )
        savemat(save_fp, {"ecg": pps["filtered_ecg"]}, format="5")

    def load_preprocessed_data(self, rec:str, preproc:Optional[List[str]]=None) -> np.ndarray:
        """ finished, checked,

        Parameters
        ----------
        rec: str,
            filename of the record
        preproc: list of str, optional
            type of preprocesses to perform,
            should be sublist of `self.allowed_preproc`,
            defaults to `self.allowed_preproc`

        Returns
        -------
        p_sig: ndarray,
            the pre-computed processed ECG
        """
        if preproc is None:
            preproc = self.allowed_preproc
        suffix = self._get_rec_suffix(preproc)
        fp = os.path.join(self.preprocess_dir, f"{rec}-{suffix}.{self.segment_ext}")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"preprocess(es) \042{preproc}\042 not done for {rec} yet")
        p_sig = loadmat(fp)["ecg"]
        if p_sig.shape[0] != 2:
            p_sig = p_sig.T
        return p_sig

    def _normalize_preprocess_names(self, preproc:List[str], ensure_nonempty:bool) -> List[str]:
        """ finished, checked,

        to transform all preproc into lower case,
        and keep them in a specific ordering 
        
        Parameters
        ----------
        preproc: list of str,
            list of preprocesses types,
            should be sublist of `self.allowd_features`
        ensure_nonempty: bool,
            if True, when the passed `preproc` is empty,
            `self.allowed_preproc` will be returned

        Returns
        -------
        _p: list of str,
            "normalized" list of preprocess types
        """
        _p = [item.lower() for item in preproc] if preproc else []
        if ensure_nonempty:
            _p = _p or self.allowed_preproc
        # ensure ordering
        _p = [item for item in self.allowed_preproc if item in _p]
        # assert all([item in self.allowed_preproc for item in _p])
        return _p

    def _get_rec_suffix(self, operations:List[str]) -> str:
        """ finished, checked,

        Parameters
        ----------
        operations: list of str,
            names of operations to perform (or has performed),
            should be sublist of `self.allowed_preproc`

        Returns
        -------
        suffix: str,
            suffix of the filename of the preprocessed ecg signal
        """
        suffix = "-".join(sorted([item.lower() for item in operations]))
        return suffix


    def _slice_data(self, force_recompute:bool=False, verbose:int=0) -> NoReturn:
        """ NOT finished, NOT checked,

        slice all records into segments of length `self.seglen`,
        and perform data augmentations specified in `self.config`
        
        Parameters
        ----------
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        verbose: int, default 0,
            print verbosity
        """
        if force_recompute:
            self._clear_cached_segments()
        for idx,rec in enumerate(self.reader.all_records):
            self._slice_one_record(
                rec=rec,
                force_recompute=force_recompute,
                update_segments_json=False,
                verbose=verbose,
            )
            if verbose >= 1:
                print(f"{idx+1}/{len(self.reader.all_records)} records", end="\r")
        if force_recompute:
            with open(self.segments_json, "w") as f:
                json.dump(self.__all_segments, f)

    def _slice_one_record(self, rec:str, force_recompute:bool=False, update_segments_json:bool=False, verbose:int=0) -> NoReturn:
        """ finished, NOT checked,

        slice one record into segments of length `self.seglen`,
        and perform data augmentations specified in `self.config`
        
        Parameters
        ----------
        rec: str,
            filename of the record
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        update_segments_json: bool, default False,
            if both `force_recompute` and `update_segments_json` are True,
            the file `self.segments_json` will be updated,
            useful when slicing not all records
        verbose: int, default 0,
            print verbosity
        """
        if (not force_recompute) and len(self.__all_segments[rec_name]) > 0:
            return
        elif force_recompute:
            self.__all_segments[rec_name] = []

        # data = self.reader.load_data(rec, units="mV")
        data = self.load_preprocessed_data(rec)
        siglen = data.shape[1]
        rpeaks = self.reader.load_rpeaks(rec)
        af_mask = self.reader.load_af_episodes(rec, fmt="mask")
        forward_len = self.seglen - self.config.overlap_len
        critical_forward_len = self.seglen - self.config.critical_overlap_len
        critical_forward_len = [critical_forward_len//2, critical_forward_len]

        # find critical points
        critical_points, = np.where(np.diff(af_mask)!=0)
        critical_points = [p for p in critical_points if critical_forward_len<=p<siglen-critical_forward_len]

        segments = []

        # offline augmentations are done, including strech-or-compress, ...
        stretch_compress_choices = [-1,1] + [0] * int(2/self.config.stretch_compress_prob - 2)
        # ordinary segments with constant forward_len
        for idx in range(siglen//forward_len):
            start_idx = idx * forward_len
            new_seg = self.__generate_segment(
                rec=rec, data=data, start_idx=start_idx,
                stretch_compress_choices=stretch_compress_choices,
            )
            segments.append(new_seg)
        # the tail segment
        if self.config.stretch_compress != 0:
            sign = sample(stretch_compress_choices, 1)[0]
            if sign != 0:
                sc_ratio = self.config.stretch_compress
                sc_ratio = 1 + (uniform(sc_ratio/4, sc_ratio) * sign) / 100
                sc_len = int(round(sc_ratio * self.seglen))
                end_idx = siglen
                start_idx = end_idx - sc_len
                aug_seg = data[start_idx: end_idx]
                aug_seg = SS.resample(x=aug_seg, num=self.seglen).reshape((1,-1))
            else:
                end_idx = siglen
                start_idx = end_idx - sc_len
                # the segment of original signal, with no augmentation
                aug_seg = data[start_idx: end_idx]
                sc_ratio = 1
        else:
            end_idx = siglen
            start_idx = end_idx - sc_len
            # the segment of original signal, with no augmentation
            aug_seg = data[start_idx: end_idx]
            sc_ratio = 1
        # adjust af_mask and rpeaks
        seg_rpeaks = self.reader.load_rpeaks(
            rec=rec, sampfrom=start_idx, sampto=end_idx, zero_start=True,
        )
        seg_rpeaks = [
            int(round(r*sc_ratio)) for r in seg_rpeaks if rpeak_border_dist<=r<self.seglen-rpeak_border_dist
        ]
        # generate qrs_mask from rpeaks
        seg_qrs_mask = np.zeros((self.seglen,), dtype=int)
        for r in seg_rpeaks:
            seg_qrs_mask[r-self.config.qrs_mask_bias:r+self.config.qrs_mask_bias] = 1
        # generate af_mask
        seg_af_intervals = self.reader.load_af_episodes(
            rec=rec, sampfrom=start_idx, sampto=end_idx, zero_start=True, fmt="intervals",
        )
        seg_af_mask = np.zeros((self.seglen,), dtype=int)
        for itv in seg_af_intervals:
            seg_af_mask[int(round(itv[0]*sc_ratio)): int(round(itv[1]*sc_ratio))] = 1

        new_seg = ED(
            data=aug_seg,
            rpeaks=seg_rpeaks,
            qrs_mask=seg_qrs_mask,
            af_mask=seg_af_mask,
            interval=[start_idx,end_idx],
        )
        segments.append(new_seg)

        # special segments around critical_points with random forward_len in critical_forward_len
        for cp in critical_points:
            start_idx = max(0, cp - self.seglen + randint(critical_forward_len//4, critical_forward_len))
            while start_idx <= min(cp - critical_forward_len, siglen - self.seglen):
                new_seg = self.__generate_segment(
                rec=rec, data=data, start_idx=start_idx,
                stretch_compress_choices=stretch_compress_choices,
                )
                segments.append(new_seg)
                start_idx += randint(critical_forward_len//4, critical_forward_len)
        
        self.__save_segments(segments)

    def __generate_segment(self, rec:str, data:np.ndarray, start_idx:int, stretch_compress_choices:List[int]) -> ED:
        """ finished, NOT checked,

        generate segment, with possible data augmentation

        Parameter
        ---------
        rec: str,
            filename of the record
        data: ndarray,
            the whole of (preprocessed) ECG record
        start_idx: int,
            start index of the signal of `rec` for generating the segment
        stretch_compress_choices: list of int,
            choices list for the stretch_or_compress augmentation

        Returns
        -------
        new_seg: dict,
            segments (meta-)data, containing:
            - data: values of the segment, with units in mV
            - rpeaks: indices of rpeaks of the segment
            - qrs_mask: mask of qrs complexes of the segment
            - af_mask: mask of af episodes of the segment
            - interval: interval ([start_idx, end_idx]) in the original ECG record of the segment
        """
        if self.config.stretch_compress != 0:
            sign = sample(stretch_compress_choices, 1)[0]
            if sign != 0:
                sc_ratio = self.config.stretch_compress
                sc_ratio = 1 + (uniform(sc_ratio/4, sc_ratio) * sign) / 100
                sc_len = int(round(sc_ratio * self.seglen))
                end_idx = start_idx + sc_len
                if end_idx > siglen:
                    end_idx = siglen
                    start_idx = end_idx - sc_len
                aug_seg = data[start_idx: end_idx]
                aug_seg = SS.resample(x=aug_seg, num=self.seglen).reshape((1,-1))
            else:
                end_idx = start_idx + self.seglen
                # the segment of original signal, with no augmentation
                aug_seg = data[start_idx: end_idx]
                sc_ratio = 1
        else:
            end_idx = start_idx + self.seglen
            aug_seg = data[start_idx: end_idx]
            sc_ratio = 1
        # adjust af_mask and rpeaks
        seg_rpeaks = self.reader.load_rpeaks(
            rec=rec, sampfrom=start_idx, sampto=end_idx, zero_start=True,
        )
        seg_rpeaks = [
            int(round(r*sc_ratio)) for r in seg_rpeaks \
                if self.config.rpeaks_dist2border <= r < self.seglen-self.config.rpeaks_dist2border
        ]
        # generate qrs_mask from rpeaks
        seg_qrs_mask = np.zeros((self.seglen,), dtype=int)
        for r in seg_rpeaks:
            seg_qrs_mask[r-self.config.qrs_mask_bias:r+self.config.qrs_mask_bias] = 1
        # generate af_mask
        seg_af_intervals = self.reader.load_af_episodes(
            rec=rec, sampfrom=start_idx, sampto=end_idx, zero_start=True, fmt="intervals",
        )
        seg_af_mask = np.zeros((self.seglen,), dtype=int)
        for itv in seg_af_intervals:
            seg_af_mask[int(round(itv[0]*sc_ratio)): int(round(itv[1]*sc_ratio))] = 1

        new_seg = ED(
            data=aug_seg,
            rpeaks=seg_rpeaks,
            qrs_mask=seg_qrs_mask,
            af_mask=seg_af_mask,
            interval=[start_idx, end_idx],
        )
        return new_seg

    def __save_segments(self, rec:str, segments:List[ED]) -> NoReturn:
        """ finished, NOT checked,

        Parameters
        ----------
        rec: str,
            filename of the record
        segments: list of dict,
            list of the segments (meta-)data
        """
        ordering = list(range(n_seg))
        shuffle(ordering)
        for i, idx in enumerate(ordering):
            seg = segments[idx]
            data_path = os.path.join(self.segments_dirs.data, f"{rec}_{i:07d}.{self.segment_ext}".replace("data", "S"))
            savemat(data_path, {"ecg": seg.data})
            ann_path = os.path.join(self.segments_dirs.ann, f"{rec}_{i:07d}.{self.segment_ext}".replace("data", "S"))
            savemat(ann_path, {k:v for k,v in seg.items() if k not in ["data",]})

    def _clear_cached_segments(self, recs:Optional[Sequence[str]]=None) -> NoReturn:
        """ finished, NOT checked,

        Parameters
        ----------
        recs: sequence of str, optional
            sequence of the records whose segments are to be cleared,
            defaults to all records
        """
        if recs is not None:
            for rec in recs:
                subject = self.reader.get_subject_id(rec)
                for item in ["data", "ann",]:
                    path = self.segments_dirs[item][subject]
                    for f in [n for n in os.listdir(path) if n.endswith(self.segment_ext)]:
                        os.remove(os.path.join(path, f))
        for subject in self.reader.all_subjects:
            for item in ["data", "ann",]:
                path = self.segments_dirs[item][subject]
                for f in [n for n in os.listdir(path) if n.endswith(self.segment_ext)]:
                    os.remove(os.path.join(path, f))

    def _train_test_split(self,
                          train_ratio:float=0.8,
                          force_recompute:bool=False) -> Dict[str, List[str]]:
        """ finished, checked,

        do train test split,
        it is ensured that both the train and the test set contain all classes

        Parameters
        ----------
        train_ratio: float, default 0.8,
            ratio of the train set in the whole dataset (or the whole tranche(s))
        force_recompute: bool, default False,
            if True, force redo the train-test split,
            regardless of the existing ones stored in json files

        Returns
        -------
        split_res: dict,
            keys are "train" and "test",
            values are list of the subjects split for training or validation
        """
        start = time.time()
        print("\nstart performing train test split...\n")
        _train_ratio = int(train_ratio*100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0

        train_file = os.path.join(self.reader.db_dir_base, f"train_ratio_{_train_ratio}.json")
        test_file = os.path.join(self.reader.db_dir_base, f"test_ratio_{_test_ratio}.json")

        if not all([os.path.isfile(train_file), os.path.isfile(test_file)]):
            train_file = os.path.join(_BASE_DIR, "utils", os.path.basename(train_file))
            test_file = os.path.join(_BASE_DIR, "utils", os.path.basename(test_file))

        if force_recompute or not all([os.path.isfile(train_file), os.path.isfile(test_file)]):
            all_subjects = set(self.reader.df_stats.subject_id.tolist())
            afp_subjects = set(self.reader.df_stats[self.reader.df_stats.label=="AFp"].subject_id.tolist())
            aff_subjects = set(self.reader.df_stats[self.reader.df_stats.label=="AFf"].subject_id.tolist()) - afp_subjects
            normal_subjects = all_subjects - afp_subjects - aff_subjects

            test_set = sample(afp_subjects, int(round(len(afp_subjects)*_test_ratio/100))) + \
                sample(aff_subjects, int(round(len(aff_subjects)*_test_ratio/100))) + \
                sample(normal_subjects, int(round(len(normal_subjects)*_test_ratio/100)))
            train_set = list(all_subjects - set(test_set))
            
            shuffle(test_set)
            shuffle(train_set)

            train_file_1 = os.path.join(self.reader.db_dir_base, f"train_ratio_{_train_ratio}.json")
            train_file_2 = os.path.join(_BASE_DIR, "utils", f"train_ratio_{_train_ratio}.json")
            with open(train_file_1, "w") as f1, open(train_file_2, "w") as f2:
                json.dump(train_set, f1, ensure_ascii=False)
                json.dump(train_set, f2, ensure_ascii=False)
            test_file_1 = os.path.join(self.reader.db_dir_base, f"test_ratio_{_test_ratio}.json")
            test_file_2 = os.path.join(_BASE_DIR, "utils", f"test_ratio_{_test_ratio}.json")
            with open(test_file_1, "w") as f1, open(test_file_2, "w") as f2:
                json.dump(test_set, f1, ensure_ascii=False)
                json.dump(test_set, f2, ensure_ascii=False)
            print(nildent(f"""
                train set saved to \n\042{train_file_1}\042and\n\042{train_file_2}\042
                test set saved to \n\042{test_file_1}\042and\n\042{test_file_2}\042
                """
            ))
        else:
            with open(train_file, "r") as f:
                train_set = json.load(f)
            with open(test_file, "r") as f:
                test_set = json.load(f)

        print(f"train test split finished in {(time.time()-start)/60:.2f} minutes")

        split_res = ED({
            "train": train_set,
            "test": test_set,
        })
        return split_res

    def plot_seg(self, seg:str, ticks_granularity:int=0, rpeak_inds:Optional[Union[Sequence[int],np.ndarray]]=None) -> NoReturn:
        """ finished, NOT checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_1_1_0000193"
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        rpeak_inds: array_like, optional,
            indices of R peaks,
        """
        seg_data = self._load_seg_data(seg)
        seg_mask = self._load_seg_mask(seg)
        intervals = mask_to_intervals(seg_mask.flatten(), vals=1)
        seg_ann = "rpeaks" if self.task in ["qrs_detection",] else "af_episodes"
        seg_ann = {seg_ann: intervals}
        rec_name = seg.replace("S", "data")[:-8]
        self.reader.plot(
            rec=rec_name,  # unnecessary indeed
            data=seg_data,
            ann=seg_ann,
            ticks_granularity=ticks_granularity,
            rpeak_inds=rpeak_inds,
        )
