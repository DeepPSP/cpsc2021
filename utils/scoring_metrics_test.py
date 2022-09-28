"""
"""

import os
import glob
import textwrap
import argparse
from typing import Union, Optional, List


def import_parents(level: int = 1) -> None:
    # https://gist.github.com/vaultah/d63cb4c86be2774377aa674b009f759a
    import sys
    import importlib
    from pathlib import Path

    global __package__
    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]

    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError:  # already removed
        pass
    __package__ = ".".join(parent.parts[len(top.parts) :])
    importlib.import_module(__package__)  # won't be needed after that


if __name__ == "__main__" and __package__ is None:
    import_parents(level=1)

import numpy as np
import wfdb

from .scoring_metrics import (  # noqa: F401
    RefInfo,
    load_ans,
    score,
    ue_calculate,
    ur_calculate,
    compute_challenge_metric,
    gen_endpoint_score_mask,
)
from cfg import BaseCfg
from sample_data import extract_sample_data_if_needed


extract_sample_data_if_needed()
# _l_test_records = list(set([os.path.splitext(item)[0] for item in os.listdir(BaseCfg.test_data_dir)]))
_l_test_records = [
    os.path.splitext(os.path.basename(item))[0]
    for item in glob.glob(os.path.join(BaseCfg.test_data_dir, "*.dat"))
]


def get_parser() -> dict:
    """ """
    description = "test for the custom scoring metrics, adjusted from the official socring metrics."
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--db-dir",
        type=str,
        help="full database directory",
        dest="db_dir",
    )
    parser.add_argument(
        "-c",
        "--classes",
        type=str,
        help="""classes to check, separated by ",", subset of {"n", "aff", "afp"}, case insensitive""",
        dest="classes",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log verbosity",
        dest="verbose",
    )

    args = vars(parser.parse_args())

    return args


def _load_af_episodes(
    fp: str, fmt: str = "c_intervals"
) -> Union[List[List[int]], np.ndarray]:
    """finished, checked,

    load the episodes of atrial fibrillation, in terms of intervals or mask,

    a simplified version of corresponding method in the data reader class in data_reader.py

    Parameters
    ----------
    fp: str,
        path of the record
    fmt: str, default "c_intervals",
        format of the episodes of atrial fibrillation, can be one of "intervals", "mask", "c_intervals"

    Returns
    -------
    af_episodes: list or ndarray,
        episodes of atrial fibrillation, in terms of intervals or mask
    """
    header = wfdb.rdheader(fp)
    ann = wfdb.rdann(fp, extension="atr")
    aux_note = np.array(ann.aux_note)
    critical_points = ann.sample
    af_start_inds = np.where((aux_note == "(AFIB") | (aux_note == "(AFL"))[0]
    af_end_inds = np.where(aux_note == "(N")[0]
    assert len(af_start_inds) == len(
        af_end_inds
    ), "unequal number of af period start indices and af period end indices"

    if fmt.lower() in [
        "c_intervals",
    ]:
        af_episodes = [[start, end] for start, end in zip(af_start_inds, af_end_inds)]
        return af_episodes

    intervals = []
    for start, end in zip(af_start_inds, af_end_inds):
        itv = [critical_points[start], critical_points[end]]
        intervals.append(itv)
    af_episodes = intervals

    if fmt.lower() in [
        "mask",
    ]:
        mask = np.zeros((header.sig_len,), dtype=int)
        for itv in intervals:
            mask[itv[0] : itv[1]] = 1
        af_episodes = mask

    return af_episodes


def run_single_test(
    rec: str, classes: Optional[List[str]] = None, verbose: bool = False
) -> bool:
    """finished, checked,

    Parameters
    ----------
    rec: str,
        path of the record (without extension)
    classes: list of str, optional,
        classes (subset of {"n", "aff", "afp"}) to run test, case insensitive,
        if not specified, all 3 classes will be tested
    verbose: bool, default False,
        log verbosity

    Returns
    -------
    masks_agree: bool,
        True if official and custom onset and offset scoring masks agree,
        otherwise False
    """
    header = wfdb.rdheader(rec)
    ann = wfdb.rdann(rec, extension="atr")
    official_ref_info = RefInfo(rec)

    if classes and BaseCfg.class_fn2abbr[header.comments[0]].lower() not in [
        c.lower() for c in classes
    ]:
        print(
            f"class of {os.path.basename(rec)} is {BaseCfg.class_fn2abbr[header.comments[0]]}, hence skipped"
        )
        return

    print(f"  {os.path.basename(rec)} starts ".center(30, "-"))
    print(
        textwrap.dedent(
            f"""
        record = {os.path.basename(rec)},
        class = {header.comments[0]},
        """
        )
    )

    custom_onset_scoring_mask, custom_offset_scoring_mask = gen_endpoint_score_mask(
        siglen=header.sig_len,
        critical_points=ann.sample,
        af_intervals=_load_af_episodes(rec, fmt="c_intervals"),
        verbose=verbose,
    )
    (
        official_onset_scoring_mask,
        official_offset_scoring_mask,
    ) = official_ref_info._gen_endpoint_score_range(verbose=verbose)
    # print(textwrap.dedent(f"""
    #     custom_onset_scoring_mask.shape = {custom_onset_scoring_mask.shape},
    #     custom_offset_scoring_mask.shape = {custom_offset_scoring_mask.shape},
    #     official_onset_scoring_mask.shape = {official_onset_scoring_mask.shape},
    #     official_offset_scoring_mask.shape = {official_offset_scoring_mask.shape}
    #     """))
    onsets = (official_onset_scoring_mask == custom_onset_scoring_mask).all()
    print(f"onset masks agree: {onsets}")
    if not onsets:
        print(f"{np.where(official_onset_scoring_mask!=custom_onset_scoring_mask)[0]}")
    offsets = (official_offset_scoring_mask == custom_offset_scoring_mask).all()
    print(f"offset masks agree: {offsets}")
    if not offsets:
        print(
            f"{np.where(official_offset_scoring_mask!=custom_offset_scoring_mask)[0]}"
        )
    print("\n" + f"  {os.path.basename(rec)} finishes ".center(30, "-") + "\n")

    masks_agree = onsets and offsets
    return masks_agree


def run_test(
    l_rec: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    verbose: bool = False,
) -> None:
    """finished, checked,

    Parameters
    ----------
    l_rec: list of str, optional,
        list of records (path, excluding file extension) to run test,
        if not specified, records in the `test_data` folder will be used
    classes: list of str, optional,
        classes (subset of {"n", "aff", "afp"}) to run test, case insensitive
        if not specified, all 3 classes will be tested
    verbose: bool, default False,
        log verbosity
    """
    err_list = []
    if not l_rec:
        l_rec = [os.path.join(BaseCfg.test_data_dir, rec) for rec in _l_test_records]
    for rec in l_rec:
        masks_agree = run_single_test(rec, classes, verbose)
        if not masks_agree:
            err_list.append(os.path.basename(rec))
    print("execution finished!")
    print(f"""err_list = {",".join(err_list) or "empty"}""")


if __name__ == "__main__":
    from data_reader import CPSC2021Reader

    args = get_parser()
    db_dir = args.get("db_dir", None)
    if db_dir:
        dr = CPSC2021Reader(db_dir)
        l_rec = [dr._get_path(rec) for rec in dr.all_records]
    else:
        # if the location of full database is not given
        # use the test data instead
        l_rec = [os.path.join(BaseCfg.test_data_dir, rec) for rec in _l_test_records]
    classes = args.get("classes", None)
    if classes:
        classes = classes.split(",")
    verbose = args.get("verbose", False)
    run_test(l_rec, classes, verbose)
