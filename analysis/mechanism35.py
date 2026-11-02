"""Input-layer mechanism on all 35 TabArena datasets, clean data, both settings.

Per (setting, dataset, seed) at pruning round ROUND, from the saved masks (no training):
  input_pruned   fraction of input-layer (fc0) weights pruned by the ticket
  input_share    input-layer share of all pruned hidden weights (where the budget went)
  rho_ticket     Spearman(kept input connections per original feature, RF importance)
  rho_random     the same for the layerwise random mask (same per-layer counts)
RF importance: impurity importance of a 300-tree random forest on the training split, summed
over the one-hot units of each original feature (one forest per dataset). With the argument
`permutation`: mean val-accuracy drop of the same forest when all units of a feature are
permuted together (5 repeats, at most 2000 val rows, cached per dataset); outputs get the suffix _perm.
Ticket benefit per dataset (seed-mean test accuracy, points):
  gain_vs_random  ticket - layerwise random mask, round ROUND
  gain_vs_reinit  ticket - reinit, round ROUND
  gain_vs_dense   val-selected ticket - dense
Then Spearman across the 35 datasets between each mechanism measure and each benefit.
Writes paper_assets/mechanism35/{per_dataset,correlations}.csv and fig_mechanism35.{pdf,png}.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import style
from analysis.load import feature_kept
from lth.common import ROOT, load_config
from lth.data.preprocess import load_dataset

ROUND = 10
R = ROOT / "results"
OUT = ROOT / "paper_assets" / "mechanism35"
SUFFIX = ""
SETTINGS = {"tuned": "imp_global", "untuned": "imp_fixed_w512"}
MECH = ["input_pruned", "input_share", "rho_ticket", "rho_minus_random"]
GAIN = ["gain_vs_random", "gain_vs_reinit", "gain_vs_dense"]


IMPORTANCE = "impurity"  # or "permutation" (set from the command line)


def rf_importance(cfg, ds, fm):
    data = load_dataset(cfg["datasets"][ds], cfg["split_seed"], cfg["max_train"])
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=32).fit(data.X_train, data.y_train)
    if IMPORTANCE == "impurity":
        return pd.Series(rf.feature_importances_[fm.unit.to_numpy()],
                         index=fm.original_feature.to_numpy()).groupby(level=0).sum()
    # permutation importance on val: all one-hot units of an original feature are permuted
    # together (same row order), mean accuracy drop over 5 repeats
    cache = OUT / "importance_perm" / f"{ds}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0).iloc[:, 0]
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(data.y_val))[:2000]  # at most 2000 val rows
    X, y = data.X_val[idx], data.y_val[idx]
    base = (rf.predict(X) == y).mean()
    out = {}
    for feat, units in fm.groupby("original_feature").unit:
        drops = []
        for _ in range(5):
            Xp = X.copy()
            Xp[:, units.to_numpy()] = X[rng.permutation(len(X))][:, units.to_numpy()]
            drops.append(base - (rf.predict(Xp) == y).mean())
        out[feat] = np.mean(drops)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(out, name="importance").to_csv(cache)
    return pd.Series(out)


def acc(records, rnd, method):
    return next(x["metrics"]["test"]["accuracy"] for x in records if x["round"] == rnd and x["method"] == method)


def per_seed_rows(cfg, setting, exp, rf_cache):
    rows = []
    for f in sorted((R / exp).glob("*/seed*/result.json")):  # clean runs only (noise runs are one level deeper)
        r = json.loads(f.read_text())
        ds = r["dataset"]
        if ds not in cfg["datasets"]:
            continue
        fm = pd.DataFrame(r["feature_map"])
        if ds not in rf_cache:
            rf_cache[ds] = rf_importance(cfg, ds, fm)
        imp = rf_cache[ds]
        masks = np.load(f.parent / "masks.npz")
        rec = r["records"]
        tick = next(x for x in rec if x["round"] == ROUND and x["method"] == "ticket")
        sizes = {k: masks[f"ticket/r{ROUND}/{k}"].size for k in tick["layer_sparsity"] if k != "out"}
        pruned = {k: tick["layer_sparsity"][k] * sizes[k] for k in sizes}
        kt = feature_kept(masks[f"ticket/r{ROUND}/fc0"], r["feature_map"])
        kr = feature_kept(masks[f"random_mask_layerwise/r{ROUND}/fc0"], r["feature_map"])
        val = {x["round"]: x["metrics"]["val"]["accuracy"] for x in rec if x["method"] == "ticket"}
        best = max(val, key=val.get)
        rows.append(dict(
            setting=setting, dataset=ds, seed=r["seeds"]["model_seed"], n_features=len(imp),
            input_pruned=tick["layer_sparsity"]["fc0"],
            input_share=pruned["fc0"] / sum(pruned.values()),
            rho_ticket=spearmanr(kt.loc[imp.index], imp).statistic,
            rho_random=spearmanr(kr.loc[imp.index], imp).statistic,
            gain_vs_random=100 * (acc(rec, ROUND, "ticket") - acc(rec, ROUND, "random_mask_layerwise")),
            gain_vs_reinit=100 * (acc(rec, ROUND, "ticket") - acc(rec, ROUND, "reinit")),
            gain_vs_dense=100 * (acc(rec, best, "ticket") - acc(rec, 0, "dense")),
        ))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, rf_cache = [], {}
    for setting, exp in SETTINGS.items():
        cfg = load_config(ROOT / "configs" / ("tabarena35.yaml" if setting == "tuned" else "tabarena35_fixed512.yaml"))
        rows += per_seed_rows(cfg, setting, exp, rf_cache)
    ps = pd.DataFrame(rows)
    ps["rho_minus_random"] = ps.rho_ticket - ps.rho_random
    pd_ = ps.groupby(["setting", "dataset"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    pd_.round(4).to_csv(OUT / f"per_dataset{SUFFIX}.csv", index=False)

    corr = []
    for setting, g in pd_.groupby("setting"):
        for m in MECH:
            for y in GAIN:
                res = spearmanr(g[m], g[y], nan_policy="omit")
                corr.append(dict(setting=setting, mechanism=m, benefit=y, n=g[[m, y]].dropna().shape[0],
                                 rho=res.statistic, p=res.pvalue))
    corr = pd.DataFrame(corr)
    corr.round(4).to_csv(OUT / f"correlations{SUFFIX}.csv", index=False)

    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    pairs = [("input_pruned", "gain_vs_random", "input-layer weights pruned", "ticket − random mask (pp)"),
             ("rho_ticket", "gain_vs_random", "Spearman(mask, RF importance)", "ticket − random mask (pp)"),
             ("rho_ticket", "gain_vs_dense", "Spearman(mask, RF importance)", "ticket − dense (pp)")]
    colors = {"tuned": "#2a78d6", "untuned": "#eb6834"}
    for ax, (x, y, xl, yl) in zip(axes, pairs):
        for setting, g in pd_.groupby("setting"):
            c = corr[(corr.setting == setting) & (corr.mechanism == x) & (corr.benefit == y)].iloc[0]
            ax.scatter(g[x], g[y], s=12, color=colors[setting], alpha=0.85, edgecolor="white", linewidth=0.5,
                       label=f"{setting} (ρ={c.rho:.2f}, " + ("p<0.001" if c.p < 0.001 else f"p={c.p:.3f}") + ")")
        ax.axhline(0, color="#888", lw=0.6)
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, frameon=False, loc="best")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"Input-layer mechanism vs ticket benefit across 35 datasets (round {ROUND}, {IMPORTANCE} importance)", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_mechanism35{SUFFIX}.{ext}", dpi=300)

    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(pd_.groupby("setting")[MECH + ["rho_random"] + GAIN].mean().round(3).to_string())
        print(corr.round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "permutation":
        IMPORTANCE, SUFFIX = "permutation", "_perm"
    main()
