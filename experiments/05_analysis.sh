#!/bin/sh
# All analyses behind the paper's figures and tables (CPU only, no training).
cd "$(dirname "$0")/.."   # run from the repository root
python3 analysis/fixed_width.py                    # results/analysis/tables/fixed_width*.csv
python3 analysis/select_datasets_v2.py             # case-study selection table (Appendix D)
python3 analysis/run_all.py --config configs/complex6.yaml --out analysis_complex6
python3 analysis/complex6_summary.py               # case-study noise / mode / hp tables
python3 analysis/paper_assets.py                   # Fig. 1, Table 1, width table -> paper_assets/
python3 analysis/mechanism35.py                    # Fig. 2, impurity importance
python3 analysis/mechanism35.py permutation        # Fig. 2 robustness, permutation importance
