#!/bin/sh
# Paper Sec. 4.1-4.3, untuned regime: the same 35 datasets with the fixed 2 x 512 MLP
# (no tuning), IMP (global) + controls, dense L1 / strong-reg baselines. Clean data.
cd "$(dirname "$0")/.."   # run from the repository root
G="--gpus 0,1"
C="--config configs/tabarena35_fixed512.yaml"
python3 run_grid.py $C --stage imp --workers 10 $G
python3 run_grid.py $C --stage baselines --workers 8 $G
