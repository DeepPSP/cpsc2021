"""
"""

import os, zipfile, glob


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORK_DIR = os.path.join(_BASE_DIR, "working_dir")
_SAMPLE_DATA_DIR = os.path.join(_WORK_DIR, "sample_data")


__all__ = ["extract_sample_data_if_needed",]


def extract_sample_data_if_needed():
    """
    """
    if os.path.exists(_SAMPLE_DATA_DIR) and len(glob.glob(os.path.join(_SAMPLE_DATA_DIR, "*.dat"))) > 0:
        return
    os.makedirs(_SAMPLE_DATA_DIR, exist_ok=True)
    zf = zipfile.ZipFile(os.path.join(_SAMPLE_DATA_DIR, "sample_data.zip"))
    zf.extractall(_WORK_DIR)
    zf.close()
