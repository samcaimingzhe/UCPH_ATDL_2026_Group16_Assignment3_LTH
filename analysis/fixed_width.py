"""Tuned vs untuned fixed-width MLPs (2 x {256, 512, 1024}) on clean data.

Per dataset and architecture: seed-mean test accuracy of dense, the val-selected ticket
(per seed, the round with the best val accuracy), the ticket at rounds 10 / 15, the
controls at round 15 and the dense L1 / strong-regularisation baselines. Then a paired
Wilcoxon test across datasets of the val-selected ticket against each dense method.
Writes results/analysis/tables/fixed_width{,_wilcoxon}.csv.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.load import load_results
from lth.common import ROOT

R = ROOT / "results"
ARCHS = {"tuned": ("imp_global", "baselines"),
         **{f"fixed_w{w}": (f"imp_fixed_w{w}", f"baselines_fixed_w{w}") for w in (256, 512, 1024)}}


def per_seed(imp, base, rounds=(15,)):
    imp = imp[imp.noise_kind == "none"]
    acc = imp[imp.split == "test"].set_index(["dataset", "model_seed", "round", "method"]).accuracy
    val = imp[(imp.split == "val") & (imp.method == "ticket")].set_index(["dataset", "model_seed", "round"]).accuracy
    best = val.groupby(["dataset", "model_seed"]).idxmax()
    rows = {"dense": acc.xs((0, "dense"), level=("round", "method")),
            "ticket_valsel": pd.Series({k[:2]: acc[(*k, "ticket")] for k in best}),
            "ticket_r10": acc.xs((10, "ticket"), level=("round", "method")),
            "ticket_r15": acc.xs((15, "ticket"), level=("round", "method"))}
    for r in rounds:
        for c in ("reinit", "random_mask_layerwise", "random_mask_global"):
            rows[f"{c}_r{r}"] = acc.xs((r, c), level=("round", "method"))
    if base is not None:
        b = base[(base.noise_kind == "none") & (base.split == "test")]
        for m in ("dense_l1", "dense_strong_reg"):
            rows[m] = b[b.method == m].set_index(["dataset", "model_seed"]).accuracy
    df = pd.DataFrame(rows)
    df.index.names = ["dataset", "model_seed"]
    return df


def main():
    tabs, tests = [], []
    for arch, (e, b) in ARCHS.items():
        if not (R / e).exists():
            continue
        imp = load_results(R / e)
        if imp.empty:
            continue
        base = load_results(R / b) if (R / b).exists() else None
        ps = per_seed(imp, base if base is not None and not base.empty else None)
        m = ps.groupby("dataset").mean()
        m.insert(0, "arch", arch)
        tabs.append(m.reset_index())
        for ctrl in ("dense", "dense_l1", "dense_strong_reg"):
            if ctrl not in m or m[ctrl].isna().any():
                continue
            d = m.ticket_valsel - m[ctrl]
            tests.append(dict(arch=arch, vs=ctrl, n=len(d), mean_diff=d.mean(), wins=int((d > 0).sum()),
                              p=wilcoxon(d).pvalue if (d != 0).any() else 1.0))
    out = R / "analysis" / "tables"
    t = pd.concat(tabs)
    t.to_csv(out / "fixed_width.csv", index=False)
    w = pd.DataFrame(tests)
    w.to_csv(out / "fixed_width_wilcoxon.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print((t.set_index(["arch", "dataset"]) * 100).round(1).to_string())
        print(t.groupby("arch").mean(numeric_only=True).mul(100).round(2).to_string())
        print(w.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
