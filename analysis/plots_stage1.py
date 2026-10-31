"""Figure 1: test metric vs remaining weights (mean +- std over seeds, log x), all methods.
Figure 2: per-layer sparsity of the global-pruning ticket across rounds."""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.load import load_results
from common import ROOT
from metrics import primary_metric

STYLE = {
    "ticket": dict(color="#1f77b4", marker="o"),
    "reinit": dict(color="#d62728", marker="s"),
    "random_mask_layerwise": dict(color="#2ca02c", marker="^"),
    "random_mask_global": dict(color="#9467bd", marker="v"),
}


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_curves(df, out, split="test"):
    for ds, d in df[df.split == split].groupby("dataset"):
        metric = primary_metric(d.task_type.iloc[0])
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        dense = d[d.method == "dense"][metric]
        ax.axhline(dense.mean(), color="k", ls="--", lw=1, label=f"dense ({dense.mean():.4f})")
        ax.axhspan(dense.mean() - dense.std(), dense.mean() + dense.std(), color="k", alpha=0.08)
        for method, st in STYLE.items():
            g = d[d.method == method].groupby("round").agg(x=("remaining_frac", "mean"), m=(metric, "mean"), s=(metric, "std"))
            if g.empty:
                continue
            ax.plot(g.x, g.m, label=method, ms=3, lw=1.2, **st)
            ax.fill_between(g.x, g.m - g.s, g.m + g.s, color=st["color"], alpha=0.15)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("fraction of weights remaining")
        ax.set_ylabel(f"{split} {metric}")
        ax.set_title(f"{ds} (n_seeds={d.model_seed.nunique()})")
        ax.legend(fontsize=7)
        save(fig, out / f"fig1_curve_{ds}")


def fig_layer_sparsity(df, out):
    d = df[(df.method == "ticket") & (df.split == "test")]
    cols = [c for c in d.columns if c.startswith("sparsity_")]
    for ds, g in d.groupby("dataset"):
        layers = [c for c in cols if g[c].notna().any()]
        agg = g.groupby("round")[layers].agg(["mean", "std"])
        x = g.groupby("round").remaining_frac.mean()
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for c in layers:
            ax.errorbar(x, agg[(c, "mean")], yerr=agg[(c, "std")], label=c.removeprefix("sparsity_"), ms=3, marker="o", lw=1, capsize=2)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("fraction of weights remaining (global)")
        ax.set_ylabel("layer sparsity")
        ax.set_title(f"{ds}: per-layer sparsity, global IMP")
        ax.legend(fontsize=7)
        save(fig, out / f"fig2_layer_sparsity_{ds}")


def summary_table(df, split="test"):
    rows = []
    for ds, d in df[df.split == split].groupby("dataset"):
        metric = primary_metric(d.task_type.iloc[0])
        t = d.groupby(["round", "method"])[metric].agg(["mean", "std"]).unstack("method")
        t.insert(0, "remaining", d.groupby("round").remaining_frac.mean())
        rows.append((ds, metric, t))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="imp_global_rw2")
    args = ap.parse_args()
    exp_dir = ROOT / "results" / args.exp
    df = load_results(exp_dir)
    out = exp_dir / "figures"
    fig_curves(df, out)
    fig_layer_sparsity(df, out)
    df.to_parquet(exp_dir / "all_results.parquet")
    for ds, metric, t in summary_table(df):
        print(f"\n== {ds} ({metric}) ==")
        print(t.round(4).to_string())
