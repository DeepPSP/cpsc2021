"""

Possible Solutions
------------------
1. segmentation (AF, non-AF) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
2. sequence labelling (AF, non-AF) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
3. per-beat (R peak detection first) classification (CNN, etc. + RR LSTM) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
4. object detection (? onsets and offsets)
"""
