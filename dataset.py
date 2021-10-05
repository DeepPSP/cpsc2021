"""
data generator for feeding data into pytorch models
"""
import os, sys
import json
import time
import textwrap
from random import shuffle, randint, sample
from copy import deepcopy
from typing import Union, Optional, List, Tuple, Dict, Sequence, Set, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
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
from utils.misc import dict_to_str, list_sum, get_record_list_recursive3


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
        """ NOT finished, NOT checked,

        Parameters
        ----------
        task: str,
            name of the task, can be one of `TrainCfg.tasks`
        """
        assert task.lower() in TrainCfg.tasks, "illegal task"
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
        elif self.config.task.lower() in ["rr_lstm",]:
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
        subject = int(seg.split("_")[1])
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
        subject = int(seg.split("_")[1])
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
        """ finished, NOT checked,

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
        """ finished, NOT checked,

        preprocesses the ecg data in advance for further use,
        offline for `self.persistence`

        Parameters
        ----------
        rec: int or str,
            filename of the record
        preproc: list of str,
            type of preprocesses to perform,
            should be sublist of `self.allowed_preproc`
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        verbose: int, default 0,
            print verbosity
        """
        rec_name = self.reader._get_rec_name(rec)
        suffix = self._get_rec_suffix(preproc)
        save_fp = os.path.join(self.preprocess_dir, f"{rec_name}-{suffix}.{self.segment_ext}")
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

    def _normalize_preprocess_names(self, preproc:List[str], ensure_nonempty:bool) -> List[str]:
        """ finished, NOT checked,

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
        """ finished, NOT checked,

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
        """ NOT finished, NOT checked,

        slice one record into segments of length `self.seglen`,
        and perform data augmentations specified in `self.config`
        
        Parameters
        ----------
        rec: int or str,
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

        data = self.reader.load_data(rec, units="mV")
        rpeaks = self.reader.load_rpeaks(rec)
        mask = self.reader.load_af_episodes(rec, fmt="mask")
        border_dist = int(0.5 * self.config.fs)
        forward_len = self.seglen - self.config.overlap_len
        # TODO: not finished
        raise NotImplementedError

    def _clear_cached_segments(self, recs:Optional[Sequence[str]]=None) -> NoReturn:
        """
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
            
            # test_set = self.reader.df_stats[self.reader.df_stats.subject_id.isin(test_set)].record.tolist()
            # train_set = self.reader.df_stats[self.reader.df_stats.subject_id.isin(train_set)].record.tolist()
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
            print(textwrap.dedent(f"""
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
