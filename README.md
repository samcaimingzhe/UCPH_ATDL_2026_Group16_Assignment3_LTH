# Lottery Tickets for Tabular MLPs

Code for testing the Lottery Ticket Hypothesis (LTH) on MLPs for tabular classification:
iterative magnitude pruning (IMP) with weight rewinding on 8 TabArena datasets, against
dense, regularised and random-mask controls, under clean and noisy data.

**Short answer.** Tickets exist and beat random masks of the same sparsity, but their
advantage over a tuned dense MLP is matched by tuned L1 / strong weight decay. Pruning
learns per-layer sparsity rather than a stable set of connections; it tracks feature
importance only weakly and does not reliably cut injected uninformative features.
Tickets found on clean data transfer well to label-noise data (better than running IMP on
the noisy data). Details in `paper/main.tex`.

Results (≈42k trained networks, 3.5 GB) are on Hugging Face:
[`samcaimingzhe/lth_tabular_results`](https://huggingface.co/datasets/samcaimingzhe/lth_tabular_results).

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q tests        # 24 tests
```

Datasets are downloaded from OpenML on first use and cached in `cache/`. A GPU is used if
available; runs are deterministic (`CUBLAS_WORKSPACE_CONFIG`, deterministic algorithms).

To use the published results instead of re-running:

```bash
hf download samcaimingzhe/lth_tabular_results --repo-type dataset --local-dir results
```

## Reproducing

Every script skips runs whose `result.json` already exists, so all steps can be resumed.

```bash
G="--gpus 0,1"
python run_grid.py --stage tune --workers 4 $G     # Optuna, 50 trials per dataset, train/val only
python run_grid.py --stage imp  --workers 8 $G     # IMP (global) + controls, 8 datasets x 5 seeds
sh run_stage3.sh                                   # baselines, noise, pruning modes, k=0, analysis
python paper/make_assets.py                        # paper figures and tables
```

`run_stage3.sh` runs, in order: `--stage baselines`, `--stage noise`, `--stage imp` for each of
`configs/mode_{uniform,single_input,single_middle,single_last}.yaml` and `configs/rewind0.yaml`,
then `analysis/run_all.py`.

## Layout

| Path | Contents |
|---|---|
| `common.py` | seeds, device, config loading (`base:` inheritance) |
| `data/` | OpenML loader, preprocessing and splits, noise injection |
| `models/mlp.py` | MLP: `[Linear → ReLU → Dropout] × depth → Linear`, no BatchNorm |
| `train.py` | training loop, rewind checkpoint, mask re-applied after every step |
| `tune.py` | Optuna (TPE) tuning; writes `results/tuned/<dataset>.json` |
| `pruning/masks.py` | magnitude pruning (global / uniform / single-layer), random masks |
| `run_imp.py` | one IMP run with all controls |
| `run_baselines.py` | dense + L1 and dense + strong weight decay baselines |
| `run_grid.py` | parallel job runner over datasets, seeds, noise levels and GPUs |
| `analysis/` | result loading, statistics and figures (`run_all.py`) |
| `configs/` | `default.yaml` (main), pruning-mode and rewind variants, old Grinsztajn setup |
| `paper/` | ISBI paper source (`main.tex`, `refs.bib`, figures, tables, `make_assets.py`) |

Results are written to `results/<exp_name>/<dataset>/[<noise>_<level>/]seed<s>/`:
`result.json` (config, hyper-parameters, seeds, metrics per round and method),
`masks.npz` (ticket masks per round) and `rewind.pt` (rewind checkpoint).
Aggregated tables are in `results/analysis/tables/`.

## Protocol

- **Data.** 8 TabArena-v0.1 binary/multiclass datasets (OpenML suite 457): credit-g,
  qsar-biodeg, students_dropout, churn, Bank_Customer_Churn, heloc, jm1, online_shoppers.
  Split with `split_seed=0`: 70% train (capped at 10k rows), the rest 30% val / 70% test.
  Numeric columns get a quantile (normal) transform fit on train, categoricals are one-hot
  encoded; a `feature_map` from input unit to original feature is kept in every result.
- **Seeds.** split 0; model 42–46; noise `1000 + level index`; mask `2000 + model seed`.
- **Tuning.** 50 TPE trials per dataset on train/val (depth 1–8, width 16–1024, lr, weight
  decay, dropout, batch size), early stopping on val loss. The hyper-parameters and the epoch
  count (best epoch, at least 50) are then frozen for every run on that dataset, including
  noisy ones.
- **IMP.** 15 rounds, 20% of the remaining weights per round (output layer 10%), rewind to
  step k = 2% of training (or k = 0). Kept counts are `round(N · (1 − p)^r)`, so masks are
  nested and exact. AdamW; the mask is re-applied after every optimizer step.
- **Controls.** dense (round 0 of the same run), reinit, random layer-wise mask, random global
  mask (both from the same rewind checkpoint and data order as the ticket), dense + L1,
  dense + strong weight decay (strength selected on val), and on noisy data the clean ticket.
- **Noise.** label flips, Gaussian noise on numeric features, added uninformative features
  (permuted or Gaussian); applied to train and val only, test stays clean.
- **Metrics.** accuracy, log loss, ECE (15 bins), Brier; per-layer sparsity.
- **Statistics.** Paired Wilcoxon tests across datasets (n = 8). With 5 seeds the smallest
  possible two-sided p within a dataset is 0.0625, so per-dataset tests are descriptive only.

