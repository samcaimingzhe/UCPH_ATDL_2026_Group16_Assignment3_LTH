#!/bin/sh
# Stage 3 experiment queue. Each step skips finished runs, so it can be re-run safely.
G="--gpus 0,1"
python3 run_grid.py --stage baselines --workers 8 $G
python3 run_grid.py --stage noise --workers 10 $G
for c in mode_uniform mode_single_input mode_single_middle mode_single_last rewind0; do
  python3 run_grid.py --config configs/$c.yaml --stage imp --workers 10 $G
done
python3 analysis/run_all.py
