"""Full IMP-with-rewinding run for one (dataset, config, model_seed[, noise]), plus controls.

Output in results/<exp_name>/<dataset>/[<noise_kind>_<level>/]seed<model_seed>/:
  result.json  config, hp, seeds, noise and one record per (round, method)
  masks.npz    every mask, keys "<method>/r<round>/<layer>"
  rewind.pt    the rewind checkpoint (model, optimizer, scheduler, RNG at step k)
A run whose result.json exists is skipped.

Noisy runs reuse the hyper-parameters tuned on clean data. The `clean_ticket` control takes
the ticket (mask + rewind checkpoint) found on clean data in experiment
cfg["noise"]["transfer_from"] and trains it on the noisy data.
"""
import argparse
import time

import numpy as np
import torch

from common import ROOT, get_device, load_config, mask_seed_for, noise_seed_for, set_seed, write_json_atomic
from data.noise import apply_noise
from data.preprocess import load_dataset
from models.mlp import build_mlp
from pruning.masks import Masker, magnitude_prune, random_mask_global, random_mask_layerwise, single_layer_name
from train import steps_per_epoch, train
from tune import load_tuned


def noise_tag(kind, level):
    return None if kind in (None, "none") or level == 0 else f"{kind}_{level:g}"


def run_dir(cfg, dataset, seed, tag=None, exp_name=None):
    d = ROOT / cfg["results_dir"] / (exp_name or cfg["exp_name"]) / dataset
    return (d / tag if tag else d) / f"seed{seed}"


def load_data(cfg, dataset, noise_kind=None, level_idx=0):
    """Clean data, or the noisy version for level index `level_idx` of `noise_kind`."""
    data = load_dataset(cfg["datasets"][dataset], cfg["split_seed"], cfg["max_train"])
    if not noise_kind:
        return data, 0
    nc = cfg["noise"]
    level = nc["levels"][noise_kind][level_idx]
    data = apply_noise(data, noise_kind, level, noise_seed_for(level_idx), nc["apply_to_test"],
                       nc["uninformative_method"])
    return data, level


def _record(round_idx, method, masker, res):
    return {
        "round": round_idx,
        "method": method,
        "remaining_frac": masker.remaining_frac(),
        "kept": masker.kept(),
        "layer_sparsity": masker.layer_sparsity(),
        "metrics": res["metrics"],
    }


def run_imp(cfg, dataset, model_seed, noise_kind=None, level_idx=0):
    data, level = load_data(cfg, dataset, noise_kind, level_idx)
    tag = noise_tag(noise_kind, level)
    out = run_dir(cfg, dataset, model_seed, tag)
    if (out / "result.json").exists():
        print(f"[imp] {out}: done, skip")
        return
    pc = cfg["pruning"]
    hp = load_tuned(cfg, dataset)["hp"]
    if pc["mode"].startswith("single_"):
        try:
            single_layer_name([f"fc{i}" for i in range(hp["depth"])], pc["mode"])
        except ValueError as e:
            print(f"[imp] {dataset}: skip {pc['mode']} ({e})")
            return
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = get_device(cfg["device"])
    n_epochs = hp["epochs"]
    total_steps = n_epochs * steps_per_epoch(len(data.X_train), hp["batch_size"])
    k = int(round(pc["rewind_frac"] * total_steps))
    mask_seed = mask_seed_for(model_seed)
    controls = list(cfg["controls"])

    clean = None
    if tag and "clean_ticket" in controls:
        if noise_kind == "uninformative":  # input dim differs from the clean ticket
            controls.remove("clean_ticket")
        else:
            cdir = run_dir(cfg, dataset, model_seed, exp_name=cfg["noise"]["transfer_from"])
            if not (cdir / "result.json").exists():
                raise FileNotFoundError(f"clean ticket missing: {cdir}")
            clean = {"ckpt": torch.load(cdir / "rewind.pt", weights_only=False),
                     "masks": np.load(cdir / "masks.npz")}
    elif "clean_ticket" in controls:
        controls.remove("clean_ticket")

    def fit(model, masker, **kw):
        return train(model, data, hp, n_epochs=n_epochs, model_seed=model_seed, device=device, masker=masker, **kw)

    records, all_masks = [], {}

    # round 0: dense training (this is also the `dense` baseline) and the rewind checkpoint
    set_seed(model_seed)
    model = build_mlp(data, hp).to(device)
    masker = Masker(model)
    res = fit(model, masker, save_ckpt_at=k)
    ckpt = res["ckpt"]
    records.append(_record(0, "dense", masker, res))
    records.append(_record(0, "ticket", masker, res))
    # full checkpoint (model + optimizer + scheduler + RNG at step k) only where another run
    # resumes from it (clean tickets, for the clean_ticket control); weights only elsewhere
    full = not tag and cfg.get("save_full_rewind", True)
    torch.save(ckpt if full else {"step": ckpt["step"], "model": ckpt["model"]}, out / "rewind.pt")

    for r in range(1, pc["rounds"] + 1):
        masks = magnitude_prune(model, masker.masks, r, pc["rate"], pc["out_factor"], pc["mode"],
                                pc.get("min_keep_frac", 0.01))
        masker = Masker(model, masks)
        set_seed(model_seed)
        res = fit(model, masker, from_ckpt=ckpt)
        records.append(_record(r, "ticket", masker, res))
        for n, m in masker.to_numpy().items():
            all_masks[f"ticket/r{r}/{n}"] = m

        for ctrl in controls:
            sub_seed = mask_seed * 100 + r
            m2 = build_mlp(data, hp).to(device)
            if ctrl == "reinit":
                set_seed(sub_seed)  # fresh init from mask_seed, trained from step 0
                m2 = build_mlp(data, hp).to(device)
                mk2 = Masker(m2, masks)
                set_seed(model_seed)
                res2 = fit(m2, mk2)
            elif ctrl in ("random_mask_layerwise", "random_mask_global"):
                fn = random_mask_layerwise if ctrl == "random_mask_layerwise" else random_mask_global
                mk2 = Masker(m2, fn(masks, sub_seed))
                set_seed(model_seed)
                res2 = fit(m2, mk2, from_ckpt=ckpt)  # same rewind weights and data as the ticket
            elif ctrl == "clean_ticket":
                cm = {n: torch.from_numpy(clean["masks"][f"ticket/r{r}/{n}"]) for n in masks}
                mk2 = Masker(m2, cm)
                set_seed(model_seed)
                res2 = fit(m2, mk2, from_ckpt=clean["ckpt"])
            else:
                raise NotImplementedError(ctrl)
            records.append(_record(r, ctrl, mk2, res2))
            # control masks are regenerable from (mask_seed, round); keep only the random
            # input layer, used as the analysis baseline, to save disk space
            if ctrl == "random_mask_layerwise":
                all_masks[f"{ctrl}/r{r}/fc0"] = mk2.to_numpy()["fc0"]
        print(f"[imp] {dataset} {tag or 'clean'} seed{model_seed} r{r} remain={masker.remaining_frac():.4f} "
              f"test={ {kk: round(v, 4) for kk, v in res['metrics']['test'].items()} }", flush=True)

    np.savez_compressed(out / "masks.npz", **all_masks)
    write_json_atomic(
        {
            "dataset": dataset,
            "task_id": cfg["datasets"][dataset],
            "task_type": data.task_type,
            "exp_name": cfg["exp_name"],
            "config": cfg,
            "hp": hp,
            "seeds": {"split_seed": cfg["split_seed"], "model_seed": model_seed, "mask_seed": mask_seed,
                      "noise_seed": noise_seed_for(level_idx) if tag else None},
            "noise": {"kind": noise_kind if tag else "none", "level": level},
            "n_train": len(data.X_train),
            "rewind_step": k,
            "total_steps": total_steps,
            "feature_map": data.feature_map.to_dict(orient="list"),
            "records": records,
            "runtime_s": time.time() - t0,
        },
        out / "result.json",  # written last: its existence marks the run complete
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--noise", default=None)
    ap.add_argument("--level-idx", type=int, default=0)
    args = ap.parse_args()
    run_imp(load_config(args.config), args.dataset, args.seed, args.noise, args.level_idx)
