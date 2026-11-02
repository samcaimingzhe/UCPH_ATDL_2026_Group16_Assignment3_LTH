"""Figures and tables for the paper, written to paper_assets/.

Settings: tuned (per-dataset Optuna MLP, exp imp_global / baselines) and untuned (fixed 2 x 512,
exp imp_fixed_w512 / baselines_fixed_w512), 35 TabArena classification datasets, clean data,
test accuracy, 5 model seeds.

Per dataset: paired t-test over seeds; p-values Benjamini-Hochberg corrected across the 35
datasets within each (setting, comparison). Across datasets: Wilcoxon signed-rank on seed means,
Holm corrected within each table.

  fig1_heatmap_compact.{pdf,png} the ticket vs each control / dense baseline, per dataset (main text)
  fig1_heatmap.{pdf,png}        the same with datasets as rows (appendix)
  table1_ticket_vs_controls.csv ticket (round 15) vs reinit / random masks (round 15)
  table2_ticket_vs_dense.csv    val-selected ticket vs dense / dense+L1 / dense+strong reg
  table_s1_width.csv            untuned widths 256 / 512 / 1024 on the 8 pilot datasets (Holm over the table)
  per_dataset_{tuned,untuned}.csv  seed-mean accuracy; mean diff, raw and BH p per comparison
  complex6/                     copies of the 6-dataset mechanism / noise / mode results
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.stats import false_discovery_control, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import style
from analysis.fixed_width import per_seed
from analysis.load import load_results
from lth.common import ROOT

ALPHA = 0.05
R = ROOT / "results"
OUT = ROOT / "paper_assets"
SETTINGS = {"tuned": ("imp_global", "baselines"), "untuned": ("imp_fixed_w512", "baselines_fixed_w512")}
# (ticket variant, reference, label). + in the heatmap = the ticket is significantly better.
# Controls are compared at equal sparsity (round 15); dense baselines against the val-best ticket.
COMPARISONS = [
    ("ticket_r15", "reinit_r15", "Ticket r15 vs reinit"),
    ("ticket_r15", "random_mask_layerwise_r15", "Ticket r15 vs rand. layerwise"),
    ("ticket_r15", "random_mask_global_r15", "Ticket r15 vs rand. global"),
    ("ticket_valsel", "dense", "Val-best ticket vs dense"),
    ("ticket_valsel", "dense_l1", "Val-best ticket vs dense + L1"),
    ("ticket_valsel", "dense_strong_reg", "Val-best ticket vs dense + strong reg"),
]
N_CONTROL_ROWS = 3
CONTROLS = ["reinit_r15", "random_mask_layerwise_r15", "random_mask_global_r15"]
DENSE = ["dense", "dense_l1", "dense_strong_reg"]


def compare(ps, a, b):
    """Per dataset: mean diff a - b and paired t-test p over seeds; BH across datasets."""
    rows = []
    for ds, g in ps.groupby("dataset"):
        d = g[a] - g[b]
        p = 1.0 if np.allclose(d, 0) else ttest_rel(g[a], g[b]).pvalue
        rows.append(dict(dataset=ds, diff=d.mean(), p=1.0 if np.isnan(p) else p))
    df = pd.DataFrame(rows).set_index("dataset")
    df["p_bh"] = false_discovery_control(df.p)
    df["score"] = np.where(df.p_bh < ALPHA, np.sign(df["diff"]), 0).astype(int)
    df["score_raw"] = np.where(df.p < ALPHA, np.sign(df["diff"]), 0).astype(int)
    return df


def summary_row(setting, a, b, c, means):
    d = means[a] - means[b]
    return dict(setting=setting, method=a, vs=b, n=len(d), mean_diff_pp=100 * d.mean(),
                sig_better_bh=int((c.score > 0).sum()), sig_worse_bh=int((c.score < 0).sum()),
                sig_better_raw=int((c.score_raw > 0).sum()), sig_worse_raw=int((c.score_raw < 0).sum()),
                mean_wins=int((d > 0).sum()), wilcoxon_p=wilcoxon(d).pvalue)


def holm(df):
    df["wilcoxon_p_holm"] = multipletests(df.wilcoxon_p, method="holm")[1]
    return df


def heatmap(scores, compact):
    """Ticket vs each reference, per dataset; tuned and untuned panels. compact: comparisons as
    rows and datasets as columns (main text); otherwise datasets as rows (appendix)."""
    style.apply()
    labels = [lab for _, _, lab in COMPARISONS]
    order = scores["tuned"].iloc[:, :N_CONTROL_ROWS].sum(axis=1).sort_values(ascending=False).index
    cmap = ListedColormap(["#e34948", "#f0efec", "#2a78d6"])
    if compact:
        fig, axes = plt.subplots(2, 1, figsize=(7.16, 3.0), sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 7.0), sharey=True)
    for ax, (name, s) in zip(axes, scores.items()):
        s = s.loc[order, labels]
        m = s.T if compact else s
        ax.imshow(m.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i, j in zip(*np.nonzero(m.values)):
            ax.text(j, i, "+" if m.values[i, j] > 0 else "−", ha="center", va="center",
                    color="white", fontsize=5.5 if compact else 7, fontweight="bold")
        counts = [f"{lab} ({(s[lab] > 0).sum()}/{(s[lab] < 0).sum()})" for lab in labels]
        if compact:
            ax.set_yticks(range(len(labels)), counts, fontsize=6)
            ax.set_xticks(range(len(order)), order, rotation=90, fontsize=5.5)
            ax.axhline(N_CONTROL_ROWS - 0.5, color="#333", lw=0.8)
        else:
            ax.set_xticks(range(len(labels)), counts, rotation=55, ha="right", fontsize=7)
            ax.set_yticks(range(len(order)), order, fontsize=6.5)
            ax.axvline(N_CONTROL_ROWS - 0.5, color="#333", lw=0.8)
        ax.set_xticks(np.arange(-.5, m.shape[1]), minor=True)
        ax.set_yticks(np.arange(-.5, m.shape[0]), minor=True)
        ax.grid(which="minor", color="white", linewidth=1 if compact else 1.5)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title({"tuned": "(a) Tuned MLP", "untuned": "(b) Untuned 2×512 MLP"}[name],
                     fontsize=7.5 if compact else 9, pad=3)
    if not compact:
        fig.suptitle("Ticket vs reference, test accuracy (paired t-test over 5 seeds, BH-corrected, "
                     "q<0.05): + ticket better, − ticket worse, gray n.s.", fontsize=8)
    fig.tight_layout(h_pad=0.4)
    name = "fig1_heatmap_compact" if compact else "fig1_heatmap"
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)


def main():
    OUT.mkdir(exist_ok=True)
    t1, t2, scores = [], [], {}
    for setting, (e, b) in SETTINGS.items():
        ps = per_seed(load_results(R / e), load_results(R / b))
        means = ps.groupby("dataset").mean()
        per_ds = means.add_prefix("acc_")
        sc = {}
        for a, b_, lab in COMPARISONS:
            c = compare(ps, a, b_)
            sc[lab] = c.score
            per_ds[[f"{a}_vs_{b_}_{k}" for k in ("diff", "p", "p_bh")]] = c[["diff", "p", "p_bh"]].values
            (t1 if b_ in CONTROLS else t2).append(summary_row(setting, a, b_, c, means))
        scores[setting] = pd.DataFrame(sc)
        per_ds.round(5).to_csv(OUT / f"per_dataset_{setting}.csv")
    t1, t2 = holm(pd.DataFrame(t1)), holm(pd.DataFrame(t2))
    t1.round(5).to_csv(OUT / "table1_ticket_vs_controls.csv", index=False)
    t2.round(5).to_csv(OUT / "table2_ticket_vs_dense.csv", index=False)
    heatmap(scores, compact=True)
    heatmap(scores, compact=False)

    # width robustness: untuned 2 x {256, 512, 1024} on the 8 pilot datasets (the only ones run
    # at every width), ticket r15 vs controls and val-selected ticket vs dense methods
    pilot = sorted(p.name for p in (R / "imp_fixed_w256").iterdir())
    ws = []
    for wd in (256, 512, 1024):
        imp, base = load_results(R / f"imp_fixed_w{wd}"), load_results(R / f"baselines_fixed_w{wd}")
        ps = per_seed(imp[imp.dataset.isin(pilot)], base[base.dataset.isin(pilot)])
        means = ps.groupby("dataset").mean()
        for a, bb in [("ticket_r15", c) for c in CONTROLS] + [("ticket_valsel", d) for d in DENSE]:
            ws.append(summary_row(f"untuned_w{wd}", a, bb, compare(ps, a, bb), means))
    holm(pd.DataFrame(ws)).round(5).to_csv(OUT / "table_s1_width.csv", index=False)

    c6 = OUT / "complex6"
    c6.mkdir(exist_ok=True)
    src = R / "analysis_complex6"
    for f in ["tables/mask_feature_stats.csv", "tables/summary_noise.csv", "tables/summary_modes.csv",
              "tables/summary_hp.csv", "tables/stability_iou.csv", "fig3_rf_spearman.pdf",
              "fig3_rf_spearman.png", "fig5_stability.pdf", "fig5_stability.png"]:
        shutil.copy2(src / f, c6 / Path(f).name)
    shutil.copytree(src / "fig4_uninformative", c6 / "fig4_uninformative", dirs_exist_ok=True)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t1.round(4).to_string(index=False))
        print(t2.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
