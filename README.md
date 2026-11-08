# When do lottery tickets exist in tabular MLPs?

Code for the report *When do lottery tickets exist in tabular MLPs? Tuning decides, feature selection may explain*,
written for [**NDAK24003U Advanced Topics in Deep Learning**](https://kurser.ku.dk/course/ndak24003u) (University of Copenhagen, 2026), Assignment 3.

Group 16: Mingzhe Cai, Shang Xi, Yusheng Lu, Chenze Zhang.
Original paper: [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635).
Raw results: [Hugging Face dataset](https://huggingface.co/datasets/samcaimingzhe/UCPH_ATDL_2026_Group16_Assignment3_LTH).

We run iterative magnitude pruning (IMP) with weight rewinding on 35 TabArena classification datasets, once with an MLP tuned for each dataset and once with an untuned 2 x 512 MLP. The tickets are compared with re-initialised and random-mask sub-networks and with regularised dense networks. We then ask whether tickets select features, what it costs to find them, and how fast they train.

## Experiments

| | Experiment | In the paper | Code |
|---|---|---|---|
| 1 | IMP, sparse controls and regularised dense baselines, tuned MLP, 35 datasets | Sec. 4.1, 4.2, Fig. 1, Table 1 | `experiments/01_tabarena35_tuned.sh` |
| 2 | The same for the untuned 2 x 512 MLP | Sec. 4.1, 4.2, Fig. 1, Table 1 | `experiments/02_tabarena35_untuned.sh` |
| 3 | Untuned MLPs of width 256, 512 and 1024 on 8 datasets | App. D | `experiments/03_width_robustness.sh` |
| 4 | Case study on 6 datasets (noise, pruning variants, rewinding to k = 0) | Sec. 4.3, App. E | `experiments/04_case_study6.sh` |
| 5 | Input-layer mask vs. random-forest feature importance on 35 datasets | Sec. 4.3, Fig. 3, App. F | `analysis/mechanism35.py` |
| 6 | Search budget of IMP (replay of the archived runs of 1 and 2) | Sec. 4.4 | `contributions/shang_xi/` |
| 7 | Training efficiency of tickets (iterations and GPU time) | App. H, Figs. 6 to 15 | `contributions/yusheng_lu/scripts/` |

<!-- TODO: add the Splice CPU experiment behind Fig. 2 once its script is in the repository. -->

Experiments 1 to 4 train models. All statistics, figures and tables for experiments 1 to 5 are made by `experiments/05_analysis.sh` and written to `paper_assets/`.

## Setup

Python 3.12 or newer.

```bash
pip install -r requirements.txt
```

Commands are run from the repository root unless a `cd` is shown. Datasets are downloaded from OpenML on first use.

## Reproduce without training (CPU)

Download the raw results into `results/` and rerun the analyses.

```bash
pip install huggingface_hub
hf download samcaimingzhe/UCPH_ATDL_2026_Group16_Assignment3_LTH --repo-type dataset --local-dir results

sh experiments/05_analysis.sh                       # experiments 1 to 5 -> paper_assets/

cd contributions/shang_xi                           # experiment 6
python3 run_all.py --inputs-root ../../results
```

`run_all.py` checks the hashes of the 350 input runs and compares its output with `reference_outputs/`.

## Reproduce with training (GPU)

The scripts for experiments 1 to 4 assume two GPUs (`G="--gpus 0,1"` at the top of each script). They skip runs that are already finished, so they can be resumed. Experiment 4 needs experiment 1.

```bash
sh experiments/01_tabarena35_tuned.sh
sh experiments/02_tabarena35_untuned.sh
sh experiments/03_width_robustness.sh
sh experiments/04_case_study6.sh
sh experiments/05_analysis.sh
```

Experiment 7 needs one CUDA GPU. It reruns IMP so that each mask and rewind checkpoint come from the same run that is timed, and it reuses the tuned hyper-parameters in `contributions/yusheng_lu/tuned/`. Run the two steps in this order.

```bash
cd contributions/yusheng_lu/scripts
python3 q5_fresh_imp.py --project ../../.. --tuned-source ../tuned --out ../q5_results6    # 6 case-study datasets, Figs. 6 and 11
python3 q5_extend35.py  --project ../../.. --tuned-source ../tuned --out ../q5_results35   # all 35 datasets, Figs. 7 to 10 and 12 to 15
```

A single run can also be started directly, for example

```bash
python -m lth.tune --dataset heloc --config configs/tabarena35.yaml
python -m lth.run_imp --dataset heloc --seed 42 --config configs/tabarena35.yaml
python -m lth.run_baselines --dataset heloc --config configs/tabarena35.yaml
```
