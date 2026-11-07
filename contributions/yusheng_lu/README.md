# LTH-Tabular-MLP-Training-Efficiency
Extension of UCPH ATDL 2026 Group 16 Assignment 3, evaluating the accuracy and training efficiency of winning tickets across 35 TabArena datasets.
# Training Efficiency of Winning Tickets in Tabular MLPs

This repository adds an independent training-efficiency analysis extending the [UCPH ATDL 2026 Group 16 LTH project](https://github.com/samcaimingzhe/UCPH_ATDL_2026_Group16_Assignment3_LTH). It reruns IMP across 35 TabArena datasets and five random seeds, ensuring that pruning masks and rewinding checkpoints originate from the same training trajectories. The experiments compare sparse subnetworks and dense MLPs in terms of classification accuracy, training iterations, GPU wall-clock time, and time-to-target accuracy.

The results show that **159/175 (90.9%)** sparse subnetworks achieve comparable accuracy (within 1 percentage point below dense), while **175/175 (100%)** exhibit comparable full-budget training time (within ±20%), with a median training-time ratio of 1.030. The repository provides experiment scripts, training histories, paired summaries, and visualization outputs.

Reproduce on a CUDA-enabled GPU using `q5_fresh_imp.py` and `q5_extend35.py`, followed by `plot_efficiency_optimized.py`, after obtaining the original Group 16 code and dependencies.

Training time excludes IMP discovery costs. The reported comparability thresholds are descriptive rather than formal statistical equivalence tests. The original model implementations, preprocessing, and hyperparameter tuning belong to the Group 16 project; this extension focuses specifically on training efficiency.
