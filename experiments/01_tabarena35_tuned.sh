#!/bin/sh
# Paper Sec. 4.1-4.3, tuned regime: 35 TabArena datasets, per-dataset Optuna tuning,
# IMP (global) + controls, dense L1 / strong-reg baselines. Clean data. Finished runs are skipped.
cd "$(dirname "$0")/.."   # run from the repository root
G="--gpus 0,1"
C="--config configs/tabarena35.yaml"
python3 -m lth.run_grid $C --stage tune --workers 6 $G
python3 -m lth.run_grid $C --stage imp --workers 10 $G
python3 -m lth.run_grid $C --stage baselines --workers 8 $G
