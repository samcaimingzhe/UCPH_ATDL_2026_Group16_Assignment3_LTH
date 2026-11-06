# Shang Xi: archived IMP search-budget analysis

This additive contribution contains the code, manuscript snippets and derived results for a retrospective search-budget audit. It does not retrain the group's original models. The main analyses are fixed-prefix selection and a sensitivity check restricting candidates to <=20% of recorded prunable Linear weights. The patience-based stopping experiment is exploratory and is not the primary manuscript result.

## Data and provenance

- Original training repository: https://github.com/samcaimingzhe/UCPH_ATDL_2026_Group16_Assignment3_LTH
- Audited code commit: `452ecb414b2e82967d642e84d3b317f8ebd50ba4`.
- Original results: https://huggingface.co/datasets/samcaimingzhe/UCPH_ATDL_2026_Group16_Assignment3_LTH
- Archived dataset revision: `95216e75a0d2ae04fed0a7f68b958cc1fa3145c9`.
- `source_manifest.json` records SHA-256 hashes of the 350 original JSON files. Raw logs and model weights are not duplicated here. Use the fixed HF revision, not a changing main branch.

## Reproduce (CPU; no Torch or OpenML required)

From **this directory**, create/activate a Python>=3.9 environment and install:

```bash
python3 -m pip install -r requirements.txt
python3 run_all.py --inputs-root /path/to/hf_results
```

The input directory must directly contain `imp_global/<dataset>/seed42..46/result.json` and `imp_fixed_w512/<dataset>/seed42..46/result.json` for all 35 tasks. Existing users can point to `LTH_analysis/inputs/hf_results`. Others must obtain these files from the public dataset at the revision above; missing or hash-mismatched files cause failure. Data retrieval itself is not performed by this script.

Each execution creates a fresh `runs/review_<timestamp>_<id>/` folder containing both analyses, a process log, environment versions, and comparisons with the reference summary tables. The runner rejects numerical disagreement above 1e-9 rather than adjusting data. Bootstrap endpoints may depend on the software environment; preserve any differing run for inspection.

Recorded user Mac environment for the original fixed-prefix replay: Python 3.9.6, NumPy 2.0.2, pandas 2.3.3, matplotlib 3.9.4, PyYAML 6.0.3. The combined runner additionally records its actual environment. An x86/ARM/GPU environment is not a substitute for a record of the execution performed.

## Files and expected conclusions

| Directory/file | Purpose |
|---|---|
| `analysis/compute_tradeoff.py`, `lth/budget_protocol.py`, `configs/` | Fixed-prefix accounting, validation selection, audit and uncertainty |
| `sparse_budget_audit.py` | Standalone hash-verified <=20%-density replay, b=10 vs b=15 |
| `reference_outputs/fixed/` | 2,100 per-run budget rows, 420 dataset rows, 12 summary rows |
| `reference_outputs/sparse/` | 700 per-run rows, 70 paired dataset rows, 2 summary rows |
| `reference_outputs/adaptive/`, `exploratory/` | Validation-plateau stopping replay, secondary only |
| `evidence/mac_20261102/` | User-supplied fixed-budget execution record and environment |
| `paper_snippets/` | Exact contribution snippets for the group report |

At b=5 versus b=15, the unconstrained fixed-prefix analysis accounts for ~62.4% fewer ticket-stage updates, with mean accuracy differences of -0.2458 pp (tuned) and -0.0976 pp (fixed MLP). It may select dense round zero. Under the <=20% density ceiling, every run is feasible at b=10; b=10 versus b=15 accounts for ~31.2% fewer updates, with differences -0.2289 pp [-0.4237,-0.0684] and -0.1457 pp [-0.3041,0.0141]. All intervals are pointwise 95% dataset-cluster percentile intervals, with five seeds first averaged per dataset.

Accounting excludes controls, hyperparameter search, evaluation and I/O. It is not measured runtime or FLOPs. Different budgets select different densities, including under the ceiling. The analyses are post-hoc, use one data split per task, and do not establish noninferiority, equality, winning initialization, or a broadly novel pruning method. Raw training results belong to their original contributors. AI tools assisted implementation, analysis and drafting; see the group's final AI-use statement and course policy.

Exploratory replay, if desired, runs from this same directory:

```bash
python3 -m exploratory.adaptive_stop --results-root /path/to/hf_results --out runs/adaptive_new
```

Do not promote the patience rule as superior to fixed b=5: its paired intervals do not establish an advantage, and many runs select dense round zero.
