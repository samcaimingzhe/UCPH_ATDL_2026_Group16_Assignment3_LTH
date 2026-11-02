"""Group summary of the detailed experiments on the 6 selected datasets (configs/complex6.yaml):
ticket helps (splice, SDSS17, in_vehicle_coupon) vs does not help (heloc, MIC, churn).

Seed-mean test accuracy, in points:
  noise.csv      per dataset x noise setting: val-selected ticket minus dense, dense_l1,
                 dense_strong_reg and the val-chosen best of the three; clean ticket minus the
                 noisy-IMP ticket at rounds 5 / 10 / 15
  modes.csv      per dataset x pruning mode (and k = 0): val-selected ticket, minus global
  hp.csv         tuned hyper-parameters, dense train-val gap (over/under-fitting)
Writes results/analysis_complex6/tables/summary_*.csv and prints group means.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.load import load_results
from lth.common import ROOT, load_config

GROUPS = {"splice": "helps", "SDSS17": "helps", "in_vehicle_coupon": "helps",
          "heloc": "no_help", "MIC": "no_help", "churn": "no_help"}
R = ROOT / "results"
OUT = R / "analysis_complex6" / "tables"
DENSE = ["dense", "dense_l1", "dense_strong_reg"]


def load(exp):
    d = load_results(R / exp)
    return d[d.dataset.isin(GROUPS)]


def val_selected(tk):
    """Per seed, the ticket round with the best val accuracy -> its test accuracy."""
    v = tk[tk.split == "val"].set_index(["model_seed", "round"]).accuracy
    t = tk[tk.split == "test"].set_index(["model_seed", "round"]).accuracy
    return t.loc[list(v.groupby("model_seed").idxmax())].groupby("model_seed").mean()


def noise_table(imp, base):
    rows = []
    for (ds, nk, nl), g in imp.groupby(["dataset", "noise_kind", "noise_level"]):
        b = base[(base.dataset == ds) & (base.noise_kind == nk) & (base.noise_level == nl)]
        row = dict(dataset=ds, group=GROUPS[ds], noise=nk, level=nl)
        tk = val_selected(g[g.method == "ticket"])
        dense = {"dense": g[(g.method == "dense")]}
        dense.update({m: b[b.method == m] for m in DENSE[1:]})
        val = {}
        for m, x in dense.items():
            if x.empty:
                continue
            te = x[x.split == "test"].set_index("model_seed").accuracy
            val[m] = x[x.split == "val"].accuracy.mean()
            row[f"ticket_minus_{m}"] = 100 * (tk - te).mean()
        if val:
            row["ticket_minus_best_dense"] = row[f"ticket_minus_{max(val, key=val.get)}"]
        acc = g[g.split == "test"].set_index(["round", "method", "model_seed"]).accuracy
        if "clean_ticket" in g.method.values:
            for r in (5, 10, 15):
                row[f"clean_minus_noisy_r{r}"] = 100 * (acc.loc[(r, "clean_ticket")] - acc.loc[(r, "ticket")]).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def mode_table():
    ref = load("imp_global")
    ref = ref[ref.noise_kind == "none"]
    rows = []
    for exp in ("imp_global", "imp_uniform", "imp_single_input", "imp_single_middle", "imp_single_last", "imp_global_rw0"):
        d = ref if exp == "imp_global" else load(exp)
        for ds, g in d.groupby("dataset"):
            tk = val_selected(g[g.method == "ticket"])
            rg = val_selected(ref[(ref.dataset == ds) & (ref.method == "ticket")])
            acc = g[g.split == "test"].set_index(["round", "method"]).accuracy.sort_index()
            rows.append(dict(dataset=ds, group=GROUPS[ds], exp=exp, ticket_valsel=100 * tk.mean(),
                             minus_global=100 * (tk - rg).mean(),
                             ticket_r15=100 * acc.loc[(15, "ticket")].mean()))
    return pd.DataFrame(rows)


def hp_table():
    cfg = load_config(ROOT / "configs/complex6.yaml")
    d = load("imp_global")
    d = d[(d.noise_kind == "none") & (d.method == "dense")]
    rows = []
    for ds in GROUPS:
        hp = json.loads((R / "tuned" / f"{ds}.json").read_text())["hp"]
        a = d[d.dataset == ds].groupby("split").accuracy.mean()
        rows.append(dict(dataset=ds, group=GROUPS[ds], task=cfg["datasets"][ds], depth=hp["depth"], width=hp["width"],
                         lr=hp["lr"], weight_decay=hp["weight_decay"], dropout=hp["dropout"], epochs=hp["epochs"],
                         dense_train=100 * a["train"], dense_val=100 * a["val"], dense_test=100 * a["test"],
                         train_minus_test=100 * (a["train"] - a["test"])))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nt = noise_table(load("imp_global"), load("baselines"))
    mt = mode_table()
    ht = hp_table()
    for name, t in (("noise", nt), ("modes", mt), ("hp", ht)):
        t.to_csv(OUT / f"summary_{name}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(ht.round(4).to_string(index=False))
    num = [c for c in nt.columns if c.startswith(("ticket_minus", "clean_minus"))]
    print(nt.groupby(["group", "noise", "level"])[num].mean().round(2).to_string())
    print(mt.pivot_table(index="exp", columns="group", values=["minus_global", "ticket_valsel"]).round(2).to_string())
    print(mt.pivot_table(index="dataset", columns="exp", values="minus_global").round(2).to_string())


if __name__ == "__main__":
    main()
