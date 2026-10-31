"""Load result.json files into a long dataframe: one row per
(exp, dataset, noise, seed, round, method, split)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def result_files(exp_dir):
    exp_dir = Path(exp_dir)
    return sorted(set(exp_dir.glob("*/seed*/result.json")) | set(exp_dir.glob("*/*/seed*/result.json")))


def load_results(exp_dir) -> pd.DataFrame:
    rows = []
    for f in result_files(exp_dir):
        r = json.loads(f.read_text())
        noise = r.get("noise", {"kind": "none", "level": 0})
        for rec in r["records"]:
            for split, m in rec["metrics"].items():
                row = {
                    "exp": r.get("exp_name", Path(exp_dir).name),
                    "dataset": r["dataset"],
                    "task_type": r["task_type"],
                    "noise_kind": noise["kind"],
                    "noise_level": noise["level"],
                    "model_seed": r["seeds"]["model_seed"],
                    "round": rec["round"],
                    "method": rec["method"],
                    "remaining_frac": rec["remaining_frac"],
                    "split": split,
                    **m,
                }
                row.update({f"sparsity_{k}": v for k, v in rec["layer_sparsity"].items()})
                rows.append(row)
    return pd.DataFrame(rows)


def iter_runs(exp_dir):
    """Yield (result dict, masks npz, run dir) for every finished IMP run."""
    for f in result_files(exp_dir):
        r = json.loads(f.read_text())
        if (f.parent / "masks.npz").exists():
            yield r, np.load(f.parent / "masks.npz"), f.parent


def feature_kept(mask_fc0, feature_map):
    """Kept input-layer connections per original feature, averaged over its units (a
    one-hot feature with many categories would otherwise look important by unit count).
    mask_fc0 has shape (width, n_inputs)."""
    per_unit = mask_fc0.sum(0)
    fm = pd.DataFrame(feature_map)
    return pd.Series(per_unit[fm.unit.to_numpy()], index=fm.original_feature.to_numpy()).groupby(level=0).mean()
