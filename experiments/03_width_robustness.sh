#!/bin/sh
# Paper Appendix C: untuned MLPs of width 256 / 512 / 1024 (depth 2) on the 8 pilot datasets,
# IMP + controls and dense baselines, clean data.
cd "$(dirname "$0")/.."   # run from the repository root
G="--gpus 0,1"
for w in 256 512 1024; do
  python3 run_grid.py --config configs/fixed_w$w.yaml --stage imp --workers 10 $G
  python3 run_grid.py --config configs/fixed_w$w.yaml --stage baselines --workers 8 $G
done
