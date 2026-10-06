#!/usr/bin/env bash
# Full reproduction: data -> segmenter -> plate readouts -> OoC sample-efficiency -> figures -> demo.
# Runtime on one RTX 4080 SUPER: ~15 min training + ~5 min analysis. Set OOC_DATA to reuse a data folder.
set -euo pipefail
PY=${PY:-python}
$PY scripts/download_data.py
[ -f models/unet3_bbbc039.pt ] || $PY scripts/train_segmentation.py --epochs 60
$PY scripts/evaluate_segmentation.py
$PY scripts/run_assay.py --plate BBBC013
$PY scripts/run_assay.py --plate BBBC014
$PY scripts/sample_efficiency.py --reps 60
$PY scripts/make_figures.py
$PY scripts/demo.py --plate BBBC013 --well A08
