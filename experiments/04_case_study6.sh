#!/bin/sh
# Paper Sec. 4.3 / Appendix D: 6 case-study datasets (tuned regime). Noise (label flip, Gaussian
# feature noise, uninformative columns), baselines under noise, pruning modes (uniform,
# input/middle/last layer only) and rewinding to k = 0. Needs 01 to have finished.
cd "$(dirname "$0")/.."   # run from the repository root
G="--gpus 0,1"
python3 -m lth.run_grid --config configs/complex6.yaml --stage noise --workers 10 $G
python3 -m lth.run_grid --config configs/complex6.yaml --stage baselines --workers 8 $G
for c in c6_mode_uniform c6_mode_single_input c6_mode_single_middle c6_mode_single_last c6_rewind0; do
  python3 -m lth.run_grid --config configs/$c.yaml --stage imp --workers 10 $G
done
