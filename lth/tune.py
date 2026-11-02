"""Optuna search for the dense MLP on train/val only. Writes results/tuned/<dataset>.json
with the best hyper-parameters and the fixed epoch count used by all pruning runs."""
import argparse
import json

import optuna

from lth.common import ROOT, get_device, load_config, set_seed, write_json_atomic
from lth.data.preprocess import load_dataset
from lth.metrics import primary_metric
from lth.models.mlp import build_mlp
from lth.train import train


def tuned_path(cfg, dataset):
    return ROOT / cfg["results_dir"] / "tuned" / f"{dataset}.json"


def load_tuned(cfg, dataset):
    """Tuned hyper-parameters, or the untuned `fixed_hp` of the config if it sets one."""
    if cfg.get("fixed_hp"):
        return {"hp": dict(cfg["fixed_hp"])}
    return json.loads(tuned_path(cfg, dataset).read_text())


def suggest(trial, space, fixed):
    hp = dict(fixed)
    hp["depth"] = trial.suggest_int("depth", *space["depth"])
    hp["width"] = trial.suggest_int("width", *space["width"], log=True)
    hp["lr"] = trial.suggest_float("lr", *space["lr"], log=True)
    hp["weight_decay"] = trial.suggest_float("weight_decay", *space["weight_decay"], log=True)
    hp["dropout"] = trial.suggest_float("dropout", *space["dropout"])
    hp["batch_size"] = trial.suggest_categorical("batch_size", space["batch_size"])
    return hp


def tune(cfg, dataset):
    out = tuned_path(cfg, dataset)
    if out.exists():
        print(f"[tune] {dataset}: exists, skip")
        return
    tc = cfg["tune"]
    data = load_dataset(cfg["datasets"][dataset], cfg["split_seed"], cfg["max_train"])
    device = get_device(cfg["device"])
    metric = primary_metric(data.task_type)

    def objective(trial):
        hp = suggest(trial, tc["space"], tc["fixed"])
        set_seed(tc["seed"])
        model = build_mlp(data, hp)
        res = train(model, data, hp, n_epochs=tc["max_epochs"], model_seed=tc["seed"], device=device,
                    patience=tc["patience"], eval_splits=("train", "val"))  # never touches test
        trial.set_user_attr("best_epoch", res["best_epoch"])
        trial.set_user_attr("val", res["metrics"]["val"])
        return res["metrics"]["val"][metric]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=tc["seed"]))
    study.optimize(objective, n_trials=tc["n_trials"])
    best = study.best_trial
    hp = suggest(optuna.trial.FixedTrial(best.params), tc["space"], tc["fixed"])
    hp["epochs"] = max(best.user_attrs["best_epoch"], tc["min_epochs"])
    write_json_atomic(
        {
            "dataset": dataset,
            "hp": hp,
            "best_val": best.user_attrs["val"],
            "trials": [
                {"params": t.params, "value": t.value, "best_epoch": t.user_attrs.get("best_epoch")}
                for t in study.trials
            ],
        },
        out,
    )
    print(f"[tune] {dataset}: val {metric}={best.value:.4f} hp={hp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    tune(load_config(args.config), args.dataset)
