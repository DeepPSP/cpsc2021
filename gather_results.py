"""
"""

import os
import re
import time
import glob
import zipfile
import json
from typing import Sequence, NoReturn, Optional, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from data_reader import CPSC2021Reader as DR
from cfg import BaseCfg


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORK_DIR = os.path.join(_BASE_DIR, "working_dir")
_VAL_RES_DIR = os.path.join(_WORK_DIR, "val_res")
_RR_LSTM_RES_DIR = os.path.join(_VAL_RES_DIR, "sample_results_rr_lstm")
_SEQ_LAB_RES_DIR = os.path.join(_VAL_RES_DIR, "sample_results_main_seq_lab")
_UNION_RES_DIR = os.path.join(_VAL_RES_DIR, "sample_results")
_NEW_UNION_RES_DIR = os.path.join(_VAL_RES_DIR, "sample_results_new")


# extract if needed
def extract_val_res_if_needed() -> NoReturn:
    print(_VAL_RES_DIR)
    if not os.path.exists(_VAL_RES_DIR):
        os.makedirs(_VAL_RES_DIR, exist_ok=True)
        zf = zipfile.ZipFile(os.path.join(_BASE_DIR, "results", "val_res.zip"))
        zf.extractall(_VAL_RES_DIR)
        zf.close()


def gather_val_res(
    dataset_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """ """
    extract_val_res_if_needed()
    val_set = [
        os.path.splitext(os.path.basename(item))[0]
        for item in glob.glob(os.path.join(_UNION_RES_DIR, "*.json"))
    ]
    dr = DR(dataset_dir or BaseCfg.db_dir)

    agg_res = []
    for item in val_set:
        siglen = dr.df_stats[dr.df_stats.record == item].sig_len.values[0]
        fp = os.path.join(_UNION_RES_DIR, item + ".json")
        with open(fp, "r") as f:
            val_res_final = json.load(f)["predict_endpoints"]
        if len(val_res_final) == 0:
            val_res_final_cls = "N"
        elif len(val_res_final) == 1 and np.diff(val_res_final)[-1] == siglen - 1:
            val_res_final_cls = "AFf"
        else:
            val_res_final_cls = "AFp"
        fp = os.path.join(_NEW_UNION_RES_DIR, item + ".json")
        with open(fp, "r") as f:
            val_res_final_new = json.load(f)["predict_endpoints"]
        if len(val_res_final_new) == 0:
            val_res_final_new_cls = "N"
        elif (
            len(val_res_final_new) == 1 and np.diff(val_res_final_new)[-1] == siglen - 1
        ):
            val_res_final_new_cls = "AFf"
        else:
            val_res_final_new_cls = "AFp"
        fp = os.path.join(_RR_LSTM_RES_DIR, item + ".json")
        with open(fp, "r") as f:
            val_res_rr = json.load(f)["predict_endpoints"]
        if len(val_res_rr) == 0:
            val_res_rr_cls = "N"
        elif len(val_res_rr) == 1 and np.diff(val_res_rr)[-1] == siglen - 1:
            val_res_rr_cls = "AFf"
        else:
            val_res_rr_cls = "AFp"
        fp = os.path.join(_SEQ_LAB_RES_DIR, item + ".json")
        with open(fp, "r") as f:
            val_res_main_seq = json.load(f)["predict_endpoints"]
        if len(val_res_main_seq) == 0:
            val_res_main_seq_cls = "N"
        elif len(val_res_main_seq) == 1 and np.diff(val_res_main_seq)[-1] == siglen - 1:
            val_res_main_seq_cls = "AFf"
        else:
            val_res_main_seq_cls = "AFp"
        truth = dr.load_af_episodes(item)
        truth_cls = dr.load_label(item)
        agg_res.append(
            {
                "record": item,
                "final_pred": val_res_final,
                "final_pred_cls": val_res_final_cls,
                "final_pred_new": val_res_final_new,
                "final_pred_new_cls": val_res_final_new_cls,
                "rr_pred": val_res_rr,
                "rr_pred_cls": val_res_rr_cls,
                "seq_pred": val_res_main_seq,
                "seq_pred_cls": val_res_main_seq_cls,
                "truth": truth,
                "truth_cls": truth_cls,
            }
        )
    df_agg_res = pd.DataFrame(agg_res)

    cols = [
        "record",
        "truth",
        "final_pred",
        "final_pred_new",
        "rr_pred",
        "seq_pred",
        "truth_cls",
        "final_pred_cls",
        "final_pred_new_cls",
        "rr_pred_cls",
        "seq_pred_cls",
    ]
    df_agg_res = df_agg_res[cols]

    classes = ["N", "AFp", "AFf"]
    cm_rr_lstm = np.zeros((3, 3), dtype=int)
    cm_seq = np.zeros((3, 3), dtype=int)
    for idx, row in df_agg_res.iterrows():
        cm_rr_lstm[
            classes.index(row["rr_pred_cls"]), classes.index(row["truth_cls"])
        ] += 1
        cm_seq[classes.index(row["seq_pred_cls"]), classes.index(row["truth_cls"])] += 1

    return df_agg_res, cm_rr_lstm, cm_seq, classes


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: Sequence[str],
    normalize: bool = False,
    title: Optional[str] = None,
    cmap: mpl.colors.Colormap = plt.cm.Blues,
    fmt: str = "pdf",
) -> Any:
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized Confusion Matrix"
            save_name = f"normalized_cm_{int(time.time())}.{fmt}"
        else:
            title = "Confusion Matrix"
            save_name = f"not_normalized_cm_{int(time.time())}.{fmt}"
    else:
        save_name = re.sub("[\\s_-]+", "-", title.lower().replace(" ", "-")) + f".{fmt}"

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
    )
    ax.set_title(title, fontsize=24)
    ax.set_xlabel("Label", fontsize=20)
    ax.set_ylabel("Predicted", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=16)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    text_fmt = ".2f" if (normalize or cm.dtype == "float") else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], text_fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=18,
            )
    fig.tight_layout()
    # plt.show()
    plt.savefig(
        os.path.join(_VAL_RES_DIR, save_name), format=fmt, dpi=1200, bbox_inches="tight"
    )

    return ax
