"""All analysis figures and tables (items 1-7 of the plan).

  fig1  accuracy vs remaining weights, all methods + dense baselines, per dataset x noise
  fig2  per-layer sparsity across rounds (global IMP)
  fig3  Spearman(input-mask kept per feature, RF importance) across rounds
  fig4  kept connections per feature: uninformative vs real
  fig5  cross-seed stability: input-mask IoU and per-feature kept correlation; layer sparsity std
  fig6  correlation of layer-sparsity summaries with dataset meta-features
  tables/wilcoxon_*.csv  paired Wilcoxon signed-rank tests (seeds; datasets)
Outputs go to results/analysis/.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.load import feature_kept, iter_runs, load_results
from common import ROOT, load_config
from metrics import primary_metric
from run_imp import load_data

STYLE = {
    "ticket": dict(color="#1f77b4", marker="o"),
    "reinit": dict(color="#d62728", marker="s"),
    "random_mask_layerwise": dict(color="#2ca02c", marker="^"),
    "random_mask_global": dict(color="#9467bd", marker="v"),
    "clean_ticket": dict(color="#ff7f0e", marker="D"),
}
BASELINES = {"dense": ("k", "--"), "dense_l1": ("#8c564b", ":"), "dense_strong_reg": ("#e377c2", "-.")}
CONTROLS = ["reinit", "random_mask_layerwise", "random_mask_global", "clean_ticket",
            "dense", "dense_l1", "dense_strong_reg"]


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def noise_label(kind, level):
    return "clean" if kind == "none" else f"{kind}_{level:g}"


# ---------------------------------------------------------------- fig 1
def fig_curves(df, base, out, split="test"):
    d_all = pd.concat([df, base]) if base is not None else df
    d_all = d_all[d_all.split == split]
    for (ds, nk, nl), d in d_all.groupby(["dataset", "noise_kind", "noise_level"]):
        metric = primary_metric(d.task_type.iloc[0])
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for method, (c, ls) in BASELINES.items():
            v = d[d.method == method][metric]
            if len(v):
                ax.axhline(v.mean(), color=c, ls=ls, lw=1, label=f"{method} ({v.mean():.4f})")
                ax.axhspan(v.mean() - v.std(), v.mean() + v.std(), color=c, alpha=0.06)
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
        ax.set_title(f"{ds} [{noise_label(nk, nl)}] (n_seeds={d.model_seed.nunique()})", fontsize=9)
        ax.legend(fontsize=6)
        save(fig, out / "fig1_curves" / f"{ds}__{noise_label(nk, nl)}")


# ---------------------------------------------------------------- fig 2
def fig_layer_sparsity(df, out, tag="global"):
    d = df[(df.method == "ticket") & (df.split == "test") & (df.noise_kind == "none")]
    cols = [c for c in d.columns if c.startswith("sparsity_")]
    for ds, g in d.groupby("dataset"):
        layers = [c for c in cols if g[c].notna().any()]
        agg = g.groupby("round")[layers].agg(["mean", "std"])
        x = g.groupby("round").remaining_frac.mean()
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for c in layers:
            ax.errorbar(x, agg[(c, "mean")], yerr=agg[(c, "std")], label=c.removeprefix("sparsity_"),
                        ms=3, marker="o", lw=1, capsize=2)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("fraction of weights remaining")
        ax.set_ylabel("layer sparsity")
        ax.set_title(f"{ds}: per-layer sparsity, {tag} IMP")
        ax.legend(fontsize=7)
        save(fig, out / f"fig2_layer_sparsity_{tag}" / ds)


# ---------------------------------------------------------------- mask statistics (3, 4, 5)
def mask_tables(cfg, exp_dir, n_trees=300):
    """One row per (dataset, noise, seed, round, method in {ticket, random_mask_layerwise})
    with Spearman vs RF importance and per-feature kept counts."""
    rows, kept_rows = [], []
    rf_cache = {}
    for r, masks, _ in iter_runs(exp_dir):
        ds, seed = r["dataset"], r["seeds"]["model_seed"]
        nk, nl = r["noise"]["kind"], r["noise"]["level"]
        key = (ds, nk, nl, seed)
        if key not in rf_cache:
            levels = cfg["noise"]["levels"].get(nk, [0])
            data, _ = load_data(cfg, ds, None if nk == "none" else nk, levels.index(nl) if nk != "none" else 0)
            RF = RandomForestClassifier if data.task_type == "classification" else RandomForestRegressor
            rf = RF(n_estimators=n_trees, random_state=seed, n_jobs=16).fit(data.X_train, data.y_train)
            fm = pd.DataFrame(r["feature_map"])
            rf_cache[key] = pd.Series(rf.feature_importances_[fm.unit.to_numpy()],
                                      index=fm.original_feature.to_numpy()).groupby(level=0).sum()
        imp = rf_cache[key]
        fm = pd.DataFrame(r["feature_map"])
        uninf = fm.groupby("original_feature").is_uninformative.first()
        for k in masks.files:
            method, rr, layer = k.split("/")
            if layer != "fc0" or method not in ("ticket", "random_mask_layerwise"):
                continue
            rnd = int(rr[1:])
            kept = feature_kept(masks[k], r["feature_map"])
            rho = spearmanr(kept.loc[imp.index], imp).statistic
            base = dict(dataset=ds, noise_kind=nk, noise_level=nl, model_seed=seed, round=rnd, method=method)
            rows.append({**base, "spearman_rf": rho,
                         "kept_real_mean": kept[~uninf.loc[kept.index].to_numpy()].mean(),
                         "kept_uninf_mean": kept[uninf.loc[kept.index].to_numpy()].mean() if uninf.any() else np.nan})
            kept_rows.append({**base, "kept": kept.to_numpy(), "mask": masks[k]})
    return pd.DataFrame(rows), kept_rows


def fig_rf_corr(mt, out):
    d = mt[mt.noise_kind == "none"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for (ds, method), g in d.groupby(["dataset", "method"]):
        a = g.groupby("round").spearman_rf.agg(["mean", "std"])
        ax.errorbar(a.index, a["mean"], yerr=a["std"], ls="-" if method == "ticket" else ":", marker="o", ms=3,
                    capsize=2, lw=1, label=f"{ds} ({'ticket' if method == 'ticket' else 'random'})",
                    alpha=1 if method == "ticket" else 0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("pruning round")
    ax.set_ylabel("Spearman(kept per feature, RF importance)")
    ax.set_title("input-layer mask vs RF importance (clean)")
    ax.legend(fontsize=5, ncol=2)
    save(fig, out / "fig3_rf_spearman")


def fig_uninformative(mt, out):
    d = mt[(mt.noise_kind == "uninformative") & (mt.method == "ticket")]
    if d.empty:
        return
    for ds, g in d.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for i, (lv, gl) in enumerate(g.groupby("noise_level")):
            a = gl.groupby("round")[["kept_real_mean", "kept_uninf_mean"]].mean()
            c = plt.cm.viridis(i / max(1, g.noise_level.nunique() - 1))
            ax.plot(a.index, a.kept_real_mean, color=c, marker="o", ms=3, label=f"real (x{lv:g})")
            ax.plot(a.index, a.kept_uninf_mean, color=c, marker="x", ms=3, ls="--", label=f"uninformative (x{lv:g})")
        ax.set_yscale("log")
        ax.set_xlabel("pruning round")
        ax.set_ylabel("mean kept input connections per feature")
        ax.set_title(f"{ds}: uninformative vs real features")
        ax.legend(fontsize=6)
        save(fig, out / "fig4_uninformative" / ds)


def stability(kept_rows, df):
    """Cross-seed IoU of the input-layer mask and Spearman of per-feature kept counts."""
    rows = []
    kr = pd.DataFrame(kept_rows)
    kr = kr[kr.method == "ticket"]
    for (ds, nk, nl, rnd), g in kr.groupby(["dataset", "noise_kind", "noise_level", "round"]):
        recs = g.to_dict("records")
        for a, b in itertools.combinations(recs, 2):
            inter = (a["mask"] & b["mask"]).sum()
            union = (a["mask"] | b["mask"]).sum()
            dens = (a["mask"].mean() + b["mask"].mean()) / 2
            rows.append(dict(dataset=ds, noise_kind=nk, noise_level=nl, round=rnd,
                             iou=inter / union, iou_random=dens / (2 - dens),  # expected for independent masks
                             feature_rank_corr=spearmanr(a["kept"], b["kept"]).statistic))
    st = pd.DataFrame(rows)
    d = df[(df.method == "ticket") & (df.split == "test")]
    cols = [c for c in d.columns if c.startswith("sparsity_")]
    sp_std = d.groupby(["dataset", "noise_kind", "noise_level", "round"])[cols].std()
    return st, sp_std


def fig_stability(st, sp_std, out):
    d = st[st.noise_kind == "none"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ds, g in d.groupby("dataset"):
        a = g.groupby("round")[["iou", "iou_random", "feature_rank_corr"]].mean()
        l, = axes[0].plot(a.index, a.iou, marker="o", ms=3, label=ds)
        axes[0].plot(a.index, a.iou_random, ls=":", color=l.get_color())
        axes[1].plot(a.index, a.feature_rank_corr, marker="o", ms=3, label=ds)
    axes[0].set_title("input-mask IoU across seeds (dotted: random)")
    axes[1].set_title("Spearman of kept-per-feature across seeds")
    s = sp_std.xs("none", level="noise_kind").groupby(["dataset", "round"]).mean()
    for ds, g in s.groupby(level="dataset"):
        axes[2].plot(g.index.get_level_values("round"), g.mean(1), marker="o", ms=3, label=ds)
    axes[2].set_title("mean over layers of cross-seed std of layer sparsity")
    for ax in axes:
        ax.set_xlabel("pruning round")
        ax.legend(fontsize=6)
    save(fig, out / "fig5_stability")


# ---------------------------------------------------------------- fig 6
def meta_correlation(cfg, df, out, round_idx):
    d = df[(df.method == "ticket") & (df.split == "test") & (df["round"] == round_idx)
           & df.noise_kind.isin(["none", "uninformative"])]
    rows = []
    for (ds, nk, nl), g in d.groupby(["dataset", "noise_kind", "noise_level"]):
        cols = sorted([c for c in g.columns if c.startswith("sparsity_fc") and g[c].notna().all()],
                      key=lambda c: int(c.removeprefix("sparsity_fc")))
        base, _ = load_data(cfg, ds)
        fm = base.feature_map
        n_feat = base.n_original_features
        n_uninf = round(nl * n_feat) if nk == "uninformative" else 0
        rows.append(dict(
            dataset=ds, noise=noise_label(nk, nl),
            sp_input=g[cols[0]].mean(),
            sp_hidden_mean=g[cols[1:]].mean(axis=None) if len(cols) > 1 else np.nan,
            sp_last_hidden=g[cols[-1]].mean(),
            sp_out=g["sparsity_out"].mean(),
            n_samples=len(base.X_train), n_features=n_feat + n_uninf,
            cat_frac=fm.groupby("original_feature").is_categorical.first().mean() * n_feat / (n_feat + n_uninf),
            uninf_frac=n_uninf / (n_feat + n_uninf),
        ))
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "meta_features_sparsity.csv", index=False)
    sp = ["sp_input", "sp_hidden_mean", "sp_last_hidden", "sp_out"]
    meta = ["n_samples", "n_features", "cat_frac", "uninf_frac"]
    corr = pd.DataFrame([[spearmanr(t[a], t[b], nan_policy="omit").statistic for b in meta] for a in sp], index=sp, columns=meta)
    corr.to_csv(out / "tables" / "meta_correlation.csv")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(corr.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(meta)), meta, rotation=30)
    ax.set_yticks(range(len(sp)), sp)
    for i, j in np.ndindex(corr.shape):
        ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    ax.set_title(f"Spearman, round {round_idx}, n={len(t)} (dataset x uninf. level)", fontsize=8)
    save(fig, out / "fig6_meta_correlation")
    return corr


# ---------------------------------------------------------------- 7
def wilcoxon_tables(df, base, out, split="test"):
    d = pd.concat([df, base]) if base is not None else df
    d = d[d.split == split]
    seed_rows, ds_rows = [], []
    for (ds, nk, nl), g in d.groupby(["dataset", "noise_kind", "noise_level"]):
        metric = primary_metric(g.task_type.iloc[0])
        tk = g[g.method == "ticket"].set_index(["round", "model_seed"])[metric]
        for ctrl in CONTROLS:
            c = g[g.method == ctrl]
            if c.empty:
                continue
            for rnd in sorted(tk.index.get_level_values("round").unique()):
                if rnd == 0:
                    continue
                a = tk.loc[rnd]
                # dense-type baselines have no rounds: compare every round's ticket to them
                b = c.set_index("model_seed")[metric] if ctrl.startswith("dense") else c[c["round"] == rnd].set_index("model_seed")[metric]
                common = a.index.intersection(b.index)
                if len(common) < 2:
                    continue
                diff = a.loc[common] - b.loc[common]
                p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
                seed_rows.append(dict(dataset=ds, noise=noise_label(nk, nl), round=rnd, control=ctrl, metric=metric,
                                      n=len(common), mean_diff=diff.mean(), wins=(diff > 0).sum(), p=p))
    st = pd.DataFrame(seed_rows)
    st.to_csv(out / "tables" / "wilcoxon_seeds.csv", index=False)
    # across datasets: seed-mean difference per dataset, paired over datasets
    if not st.empty:
        for (noise, rnd, ctrl), g in st.groupby(["noise", "round", "control"]):
            if len(g) >= 3:
                p = wilcoxon(g.mean_diff).pvalue if (g.mean_diff != 0).any() else 1.0
                ds_rows.append(dict(noise=noise, round=rnd, control=ctrl, n_datasets=len(g),
                                    median_diff=g.mean_diff.median(), wins=(g.mean_diff > 0).sum(), p=p))
    pd.DataFrame(ds_rows).to_csv(out / "tables" / "wilcoxon_datasets.csv", index=False)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--meta-round", type=int, default=10)
    ap.add_argument("--skip-masks", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = ROOT / cfg["results_dir"]
    out = root / "analysis"
    (out / "tables").mkdir(parents=True, exist_ok=True)

    df = load_results(root / cfg["exp_name"])
    bdir = root / cfg["baselines"]["exp_name"]
    base = load_results(bdir) if bdir.exists() else None
    df.to_parquet(out / "tables" / "imp_results.parquet")
    print(f"{len(df)} rows, datasets={sorted(df.dataset.unique())}")

    fig_curves(df, base, out)
    fig_layer_sparsity(df, out)
    for mode in ("uniform", "single_input", "single_middle", "single_last"):
        md = root / f"imp_{mode}"
        if md.exists():
            dm = load_results(md)
            fig_curves(dm, base[base.noise_kind == "none"] if base is not None else None, out / f"mode_{mode}")
            fig_layer_sparsity(dm, out, mode)
    wilcoxon_tables(df, base, out)
    meta_correlation(cfg, df, out, args.meta_round)

    if not args.skip_masks:
        mt, kept_rows = mask_tables(cfg, root / cfg["exp_name"])
        mt.to_csv(out / "tables" / "mask_feature_stats.csv", index=False)
        fig_rf_corr(mt, out)
        fig_uninformative(mt, out)
        st, sp_std = stability(kept_rows, df)
        st.to_csv(out / "tables" / "stability_iou.csv", index=False)
        sp_std.to_csv(out / "tables" / "layer_sparsity_seed_std.csv")
        fig_stability(st, sp_std, out)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
