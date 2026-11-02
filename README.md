# Lottery Tickets for Tabular MLPs
[**NDAK24003U Advanced Topics in Deep Learning (ATDL)**](https://kurser.ku.dk/course/ndak24003u) --- Assignment 3

Paper: [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)

Group 16 contributors: Mingzhe Cai; Shang Xi; Yusheng Lu; Chenze Zhang;

Code for the course report *When do lottery tickets exist in tabular MLPs? Tuning decides,
feature selection may explain*.


- **Results:** [Hugging Face dataset](https://huggingface.co/datasets/samcaimingzhe/UCPH_ATDL_2026_Group16_Assignment3_LTH)

We test the Lottery Ticket Hypothesis (LTH) on MLPs for tabular classification. We run
iterative magnitude pruning (IMP) with weight rewinding on 35 TabArena classification datasets
in two training regimes, a per-dataset **tuned** MLP and an **untuned** fixed 2 x 512 MLP.
Tickets are compared with re-initialised and random-mask controls and with regularised dense
baselines.

## Main findings

| Comparison (datasets significantly better / worse) | Tuned MLP | Untuned 2 x 512 MLP |
|---|---|---|
| Ticket (round 15, 3.5% of weights) vs. random layer-wise mask | **18 / 3** | 3 / **17** |
| Ticket vs. random global mask | **19 / 3** | 10 / 16 (not significant across datasets) |
| Ticket vs. reinit | **13 / 3** | 3 / **17** |
| Val-best ticket vs. dense | 2 / 0 (+0.63 pp, p < .001) | 2 / 0 (+0.28 pp, p = .003) |
| Val-best ticket vs. dense + L1 / dense + strong reg | not different | worse on 9 datasets each (not significant across datasets) |

Per dataset: paired t-test over 5 seeds, Benjamini-Hochberg over the 35 datasets (q < 0.05).
Across datasets: Wilcoxon signed-rank test on seed means, Holm-corrected within each of the two
tables (ticket vs. sparse controls, ticket vs. dense baselines).

1. **Tickets exist only for tuned MLPs.** With tuning, the ticket beats reinit and both random
   masks of the same sparsity. Without tuning, it loses to reinit and to the random layer-wise
   mask, and it is mixed against the random global mask. The losses against reinit and the
   layer-wise mask also appear at widths 256, 512 and 1024 (8 datasets). A possible reason is
   that the untuned dense network overfits, so its largest weights are not the most useful ones.
2. **Tickets do not beat standard regularisation.** They improve slightly on the plain dense
   network, but they match (tuned) or trail (untuned) dense + L1 and dense + strong weight
   decay and dropout.
3. **Feature selection in the input layer is not settled.** In the tuned regime, the agreement
   between the input-layer mask and random-forest *impurity* importance predicts the ticket's
   advantage over a random mask (Spearman 0.59, p < 0.001, 35 datasets). With accuracy-based
   *permutation* importance the relation is weak and not significant (0.16, p = 0.36,
   34 datasets, because every permutation importance of Diabetes130US is 0). Both measures have
   known problems. Impurity importance favours variables with many categories. Permutation
   importance is zero or negative for almost all variables of some datasets (all variables of
   Diabetes130US, at least 98% of those of taiwanese_bankruptcy, coil2000 and MIC), because
   shuffling one variable rarely changes the forest's accuracy there. The amount of input pruning
   does not predict the benefit (Spearman 0.14, p = 0.41).

## Setup

Python 3.12 or newer (required by `scipy>=1.18`).

```bash
pip install -r requirements.txt
python -m pytest -q tests
```

Datasets are downloaded from OpenML on first use and cached in `cache/`. A GPU is used if
available. Runs are deterministic (`CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms`).

## Raw results

`results/` is not tracked by git. All runs behind the report are in the
[Hugging Face dataset](https://huggingface.co/datasets/samcaimingzhe/UCPH_ATDL_2026_Group16_Assignment3_LTH).
Its top-level folders are the experiment directories of `results/` (see [Layout](#layout)), so
the analyses can be rerun without any training:

```bash
pip install huggingface_hub
hf download samcaimingzhe/UCPH_ATDL_2026_Group16_Assignment3_LTH --repo-type dataset --local-dir results
sh experiments/05_analysis.sh      # also downloads the datasets from OpenML for the random forests
```

## Reproducing

Every step skips runs whose `result.json` already exists, so all scripts can be resumed.
The scripts assume two GPUs (`--gpus 0,1`). Edit `G=` at the top of a script to change this.

| Script | What it runs | Report |
|---|---|---|
| `experiments/01_tabarena35_tuned.sh` | 35 datasets, Optuna tuning, IMP + controls, dense baselines | Sec. 4.1-4.3 |
| `experiments/02_tabarena35_untuned.sh` | 35 datasets, fixed 2 x 512 MLP, IMP + controls, dense baselines | Sec. 4.1-4.3 |
| `experiments/03_width_robustness.sh` | untuned widths 256 / 512 / 1024 on 8 datasets | Appendix D |
| `experiments/04_case_study6.sh` | 6 datasets: noise, pruning modes, rewind k = 0 (needs 01) | Sec. 4.3, Appendix E |
| `experiments/05_analysis.sh` | all analyses and figures (CPU only) | all figures and tables, incl. Appendix F |

Each shell script calls the job runner `python -m lth.run_grid`, which lists the pending jobs
of one stage (`tune`, `imp`, `noise`, `baselines`) for a config and runs them in parallel, one
GPU per worker. A single run can also be started directly, for example

```bash
python -m lth.tune --dataset heloc --config configs/tabarena35.yaml
python -m lth.run_imp --dataset heloc --seed 42 --config configs/tabarena35.yaml
python -m lth.run_baselines --dataset heloc --config configs/tabarena35.yaml
```

All commands are run from the repository root.

## Outputs for the report (`paper_assets/`)

| File | Content | Report |
|---|---|---|
| `fig1_heatmap_compact.{pdf,png}` | ticket vs. each control / dense baseline, per dataset, tuned and untuned | Fig. 1 |
| `fig1_heatmap.{pdf,png}` | the same with datasets as rows | Fig. 4 |
| `table1_ticket_vs_controls.csv` | ticket (round 15) vs. reinit and random masks | Table 1, upper half |
| `table2_ticket_vs_dense.csv` | val-best ticket vs. dense, dense + L1, dense + strong reg | Table 1, lower half |
| `table_s1_width.csv` | width robustness (untuned, 8 datasets) | Table 3 |
| `per_dataset_{tuned,untuned}.csv` | seed-mean accuracies, differences, raw and BH p-values | Fig. 1, Fig. 4 |
| `mechanism35/` | input-layer mask vs. feature importance on 35 datasets; `*_perm` = permutation importance | Fig. 2, Table 7 |
| `complex6/` | case-study tables and figures (mask statistics, seed stability, noise, pruning modes) | Tables 4-6, Fig. 3 |

Generated by `analysis/paper_assets.py` and `analysis/mechanism35.py`. The case-study results
come from `analysis/run_all.py --config configs/complex6.yaml --out analysis_complex6` and
`analysis/complex6_summary.py`.

## Layout

| Path | Contents |
|---|---|
| `lth/` | core library (everything below is inside this package) |
| `lth/common.py` | seeds, device, config loading (`base:` inheritance), repository paths |
| `lth/data/` | OpenML loader, preprocessing and splits, noise injection |
| `lth/models/mlp.py` | MLP: `[Linear -> ReLU -> Dropout] x depth -> Linear`, no normalisation |
| `lth/train.py` | training loop, rewind checkpoint, mask re-applied after every step |
| `lth/tune.py` | Optuna (TPE) tuning; writes `results/tuned/<dataset>.json` |
| `lth/pruning/masks.py` | magnitude pruning (global / uniform / single layer), random masks |
| `lth/metrics.py` | accuracy, log loss, ECE (15 bins), Brier score |
| `lth/run_imp.py` | one IMP run with all controls |
| `lth/run_baselines.py` | dense + L1 and dense + strong regularisation baselines |
| `lth/run_grid.py` | parallel job runner over datasets, seeds, noise levels and GPUs |
| `configs/` | `default.yaml` (tuned protocol), `tabarena35*.yaml`, `fixed_w*.yaml`, `complex6.yaml`, `c6_*.yaml` |
| `experiments/` | launch scripts, one per experiment |
| `analysis/` | result loading, statistics, figures |
| `tests/` | unit tests |

Results are written to `results/<exp_name>/<dataset>/[<noise>_<level>/]seed<s>/`:
`result.json` (config, hyper-parameters, seeds, metrics per round and method),
`masks.npz` (ticket masks per round) and `rewind.pt` (rewind checkpoint).

| Experiment directory | Content |
|---|---|
| `tuned/` | tuned hyper-parameters per dataset |
| `imp_global/`, `baselines/` | tuned regime: clean runs (35 datasets) and noise runs (6 case-study datasets) |
| `imp_fixed_w{256,512,1024}/`, `baselines_fixed_w*/` | untuned regime (512: 35 datasets; 256 / 1024: 8) |
| `imp_uniform/`, `imp_single_input/`, `imp_global_rw0/` | case study: uniform pruning, input-only pruning, rewinding to k = 0 |
| `imp_single_middle/`, `imp_single_last/` | case study: middle-only and last-only pruning (run, but not reported) |
| `analysis/`, `analysis_complex6/` | outputs of `analysis/run_all.py` and `analysis/fixed_width.py` |

## Protocol

- **Data.** 35 of the 38 TabArena-v0.1 classification tasks (OpenML suite 457). Excluded are
  Amazon_employee_access (high-cardinality categoricals), kddcup09_appetency and hiva_agnostic
  (extreme class imbalance). Split with `split_seed=0`: 70% train (at most 10k rows), the rest
  30% val / 70% test. Numerical columns get median imputation and a quantile (normal) transform
  fitted on train. Categorical columns are one-hot encoded with at most 64 levels (rarer levels
  pooled). Unmarked string columns become days since 1970 if more than 99% parse as dates, and
  categorical otherwise. A `feature_map` from input unit to original variable is stored in every
  result.
- **Seeds.** split 0; model 42-46; noise `1000 + level index`; mask `2000 + model seed`.
- **Tuned regime.** 50 TPE trials per dataset on train/val (depth 1-8, width 16-1024,
  lr 1e-5 to 1e-2, weight decay 1e-8 to 1e-3, dropout 0-0.5, batch size 256/512/1024), at most
  300 epochs with early stopping on val loss (patience 30). The epoch count is then fixed to the
  best epoch, at least 50.
- **Untuned regime.** depth 2, width 512 (256 / 1024 for robustness), lr 1e-3, no weight
  decay, no dropout, batch size 256, 100 epochs.
- **IMP.** 15 rounds. Each round prunes 20% of the remaining weights of all layers except the
  output layer (input layer included), ranked globally by magnitude; the output layer is pruned
  at 10%. Rewind to step k = 2% of training. Kept counts are `round(N * 0.8^r)`, so rounding
  errors do not add up. AdamW; the mask is re-applied after every optimizer step.
- **Controls.** dense (round 0 of the same run), reinit, random layer-wise mask, random global
  mask (same rewind checkpoint and data order as the ticket), dense + L1, dense + strong
  dropout / weight decay (strength selected on val).
- **Feature importance (analysis).** One random forest (300 trees) per dataset on the training
  split. Impurity importance is summed over the one-hot units of a variable. Permutation
  importance is the drop in validation accuracy when all units of a variable are shuffled
  together (5 repeats, at most 2000 validation rows).
- **Noise (case study).** label flips, Gaussian noise on numerical features, added
  uninformative columns (row-permuted copies); applied to train and val only.
- **Metrics.** accuracy (primary), log loss, ECE (15 bins), Brier; per-layer sparsity.
