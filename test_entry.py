"""
"""

import os, zipfile, glob

from entry_2021 import challenge_entry
from utils.misc import save_dict
from sample_data import extract_sample_data_if_needed


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORK_DIR = os.path.join(_BASE_DIR, "working_dir")
_SAMPLE_DATA_DIR = os.path.join(_WORK_DIR, "sample_data")
_SAMPLE_RESULTS_DIR = os.path.join(_WORK_DIR, "sample_results")

os.makedirs(_SAMPLE_DATA_DIR, exist_ok=True)
os.makedirs(_SAMPLE_RESULTS_DIR, exist_ok=True)


def run_test():
    """
    """
    extract_sample_data_if_needed()
    sample_set = [
        os.path.splitext(os.path.basename(item))[0] \
            for item in glob.glob(os.path.join(_SAMPLE_DATA_DIR, "*.dat"))
    ]

    for i, sample in enumerate(sample_set):
        print(sample)
        sample_path = os.path.join(_SAMPLE_DATA_DIR, sample)
        pred_dict = challenge_entry(sample_path)

        save_dict(os.path.join(_SAMPLE_RESULTS_DIR, sample+'.json'), pred_dict)


if __name__ == "__main__":
    run_test()
