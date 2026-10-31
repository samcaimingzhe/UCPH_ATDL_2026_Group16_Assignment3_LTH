"""Grokking timing per run of run_grok.py.

t_fit:   first epoch with train acc >= 0.99 (memorisation)
t_grok:  first epoch after t_fit where val acc reaches val_at_fit + 0.9 * gain, gain being
         the best post-fit val acc minus val_at_fit
delay:   t_grok - t_fit (grokking means a large delay *and* a real gain)
gain is compared with one validation sample (1/n_val) to flag noise-level improvements.
Writes results/grok/grok_stats.csv.
"""
import json

import numpy as np
import pandas as pd

from common import ROOT

OUT = ROOT / "results/grok"


def stats(r, fit_acc=0.99, frac=0.9):
    c = r["curve"]
    ep = np.array([x["epoch"] for x in c])
    tr = np.array([x["train_acc"] for x in c])
    va = np.array([x["val_acc"] for x in c])
    n_val = round(1 / min(np.diff(np.unique(va)))) if len(np.unique(va)) > 1 else None
    row = {"dataset": r["dataset"], "wd": r["weight_decay"], "init": r["init_scale"],
           "best_val": va.max(), "best_epoch": int(ep[va.argmax()]), "final_val": va[-1]}
    hit = np.flatnonzero(tr >= fit_acc)
    if not len(hit):
        return {**row, "t_fit": None, "t_grok": None, "delay": None, "gain": None, "gain_in_samples": None}
    i = hit[0]
    gain = va[i:].max() - va[i]
    j = i + np.flatnonzero(va[i:] >= va[i] + frac * gain)[0]
    return {**row, "t_fit": int(ep[i]), "val_at_fit": va[i], "t_grok": int(ep[j]), "delay": int(ep[j] - ep[i]),
            "gain": gain, "gain_in_samples": gain * n_val if n_val else None}


if __name__ == "__main__":
    rows = [stats(json.loads(f.read_text())) for f in sorted(OUT.glob("*/*.json"))]
    df = pd.DataFrame(rows).sort_values(["dataset", "wd", "init"])
    df.to_csv(OUT / "grok_stats.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(df.round(3).to_string(index=False))
