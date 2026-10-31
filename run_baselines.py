"""Dense regularized baselines `dense_l1` and `dense_strong_reg` for one (dataset[, noise]).

The regularization strength is selected on val (seed = tune.seed, fixed epochs, tuned hp),
then each model_seed is trained with the selected value. Output:
  results/<baselines.exp_name>/<dataset>/[<noise_tag>/]selection.json
  results/<baselines.exp_name>/<dataset>/[<noise_tag>/]seed<s>/result.json
"""
import argparse
import itertools
import json
import time

from common import ROOT, get_device, load_config, set_seed, write_json_atomic
from metrics import primary_metric
from models.mlp import build_mlp
from run_imp import load_data, noise_tag, run_dir
from train import train
from tune import load_tuned


def candidates(bc, hp):
    """Hyper-parameter overrides searched for each baseline."""
    return {
        "dense_l1": [{"l1": c} for c in bc["l1_grid"]],
        "dense_strong_reg": [
            {"dropout": min(hp["dropout"] + dd, bc["max_dropout"]), "weight_decay": wd}
            for dd, wd in itertools.product(bc["dropout_add"], bc["weight_decay_grid"])
        ],
    }


def run_baselines(cfg, dataset, noise_kind=None, level_idx=0):
    bc = cfg["baselines"]
    data, level = load_data(cfg, dataset, noise_kind, level_idx)
    tag = noise_tag(noise_kind, level)
    exp = bc["exp_name"]
    base_dir = run_dir(cfg, dataset, 0, tag, exp_name=exp).parent
    if all((run_dir(cfg, dataset, s, tag, exp_name=exp) / "result.json").exists() for s in cfg["model_seeds"]):
        print(f"[baselines] {base_dir}: done, skip")
        return
    device = get_device(cfg["device"])
    hp0 = load_tuned(cfg, dataset)["hp"]
    metric = primary_metric(data.task_type)

    def fit(hp, seed, splits):
        set_seed(seed)
        model = build_mlp(data, hp)
        return train(model, data, hp, n_epochs=hp["epochs"], model_seed=seed, device=device, eval_splits=splits)

    sel_path = base_dir / "selection.json"
    if sel_path.exists():
        selection = json.loads(sel_path.read_text())
    else:
        selection = {}
        for method, cands in candidates(bc, hp0).items():
            scores = [fit({**hp0, **c}, cfg["tune"]["seed"], ("val",))["metrics"]["val"][metric] for c in cands]
            best = max(range(len(cands)), key=lambda i: scores[i])
            selection[method] = {"best": cands[best], "candidates": cands, f"val_{metric}": scores}
        write_json_atomic(selection, sel_path)

    for s in cfg["model_seeds"]:
        out = run_dir(cfg, dataset, s, tag, exp_name=exp)
        if (out / "result.json").exists():
            continue
        t0 = time.time()
        records = []
        for method, sel in selection.items():
            hp = {**hp0, **sel["best"]}
            res = fit(hp, s, ("train", "val", "test"))
            records.append({"round": 0, "method": method, "remaining_frac": 1.0, "hp": hp,
                            "layer_sparsity": {}, "metrics": res["metrics"]})
        write_json_atomic(
            {
                "dataset": dataset, "task_id": cfg["datasets"][dataset], "task_type": data.task_type,
                "exp_name": exp, "config": cfg, "hp": hp0,
                "seeds": {"split_seed": cfg["split_seed"], "model_seed": s, "mask_seed": None, "noise_seed": None},
                "noise": {"kind": noise_kind if tag else "none", "level": level},
                "records": records, "runtime_s": time.time() - t0,
            },
            out / "result.json",
        )
        print(f"[baselines] {dataset} {tag or 'clean'} seed{s} done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--noise", default=None)
    ap.add_argument("--level-idx", type=int, default=0)
    args = ap.parse_args()
    run_baselines(load_config(args.config), args.dataset, args.noise, args.level_idx)
