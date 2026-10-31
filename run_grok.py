"""Grokking pilot: does a dense MLP on tabular data show delayed generalisation?

Long training (no early stopping) with strong weight decay and scaled-up initialisation
(Omnigrok-style), on a reduced training set. Train/val curves are logged every
`--eval-every` epochs to results/grok/<dataset>/<setting>.json; a figure per dataset goes
to results/grok/figs/. Uses the clean TabArena splits of configs/default.yaml.
"""
import argparse
import itertools
import json
import time

import numpy as np
import torch

from common import ROOT, get_device, load_config, make_generator, set_seed, write_json_atomic
from data.preprocess import load_dataset
from metrics import compute_metrics
from models.mlp import MLP
from train import make_loader, predict

OUT = ROOT / "results/grok"


def subsample_train(data, n, seed):
    if len(data.X_train) <= n:
        return data
    idx = np.random.default_rng(seed).choice(len(data.X_train), n, replace=False)
    return data.copy(X_train=data.X_train[idx], y_train=data.y_train[idx])


def run(cfg, ds, wd, scale, a):
    out = OUT / ds / f"wd{wd:g}_init{scale:g}.json"
    if out.exists():
        print(f"[grok] {out}: done, skip")
        return
    data = subsample_train(load_dataset(cfg["datasets"][ds], cfg["split_seed"], cfg["max_train"]), a.n_train, 0)
    device = get_device(cfg["device"])
    set_seed(a.seed)
    model = MLP(data.n_inputs, data.n_outputs, depth=a.depth, width=a.width).to(device)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(scale)
    Xtr = torch.as_tensor(data.X_train, device=device)
    ytr = torch.as_tensor(data.y_train, device=device)
    loader = make_loader(len(Xtr), a.batch_size, make_generator(a.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=wd)
    crit = torch.nn.CrossEntropyLoss()
    curve = []
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        for idx in loader:
            idx = idx.to(device)
            loss = crit(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch % a.eval_every == 0 or epoch == 1:
            row = {"epoch": epoch, "wnorm": float(sum(p.detach().pow(2).sum() for p in model.parameters()).sqrt())}
            for s in ("train", "val", "test"):
                X, y = data.split(s)
                m = compute_metrics(data.task_type, y, predict(model, X, device))
                row[f"{s}_acc"], row[f"{s}_loss"] = m["accuracy"], m["loss"]
            curve.append(row)
    write_json_atomic({"dataset": ds, "weight_decay": wd, "init_scale": scale, "args": vars(a),
                       "n_train": len(data.X_train), "summary": summarize(curve), "curve": curve,
                       "runtime_s": time.time() - t0}, out)
    print(f"[grok] {ds} wd={wd:g} init={scale:g} {summarize(curve)}", flush=True)


def summarize(curve, fit_acc=0.99):
    """t_fit: first epoch with train acc >= fit_acc. Grokking shows up as val acc at the end
    (or its best) clearly above val acc at t_fit."""
    ep = np.array([r["epoch"] for r in curve])
    tr = np.array([r["train_acc"] for r in curve])
    va = np.array([r["val_acc"] for r in curve])
    hit = np.flatnonzero(tr >= fit_acc)
    s = {"best_val_acc": float(va.max()), "best_val_epoch": int(ep[va.argmax()]), "final_val_acc": float(va[-1]),
         "final_train_acc": float(tr[-1]), "t_fit": None, "val_at_fit": None, "gain_after_fit": None}
    if len(hit):
        i = hit[0]
        s.update(t_fit=int(ep[i]), val_at_fit=float(va[i]), gain_after_fit=float(va[i:].max() - va[i]))
    return s


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    for d in sorted(p for p in OUT.iterdir() if p.is_dir() and p.name != "figs"):
        runs = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
        fig, axes = plt.subplots(1, len(runs), figsize=(3 * len(runs), 2.6), sharey=True, squeeze=False)
        for ax, r in zip(axes[0], runs):
            c = r["curve"]
            ep = [x["epoch"] for x in c]
            ax.plot(ep, [x["train_acc"] for x in c], label="train")
            ax.plot(ep, [x["val_acc"] for x in c], label="val")
            ax.set_xscale("log")
            ax.set_title(f"wd={r['weight_decay']:g}, init×{r['init_scale']:g}", fontsize=8)
            ax.set_xlabel("epoch")
        axes[0][0].set_ylabel("accuracy")
        axes[0][0].legend(fontsize=7)
        fig.suptitle(d.name, fontsize=9)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / "figs" / f"{d.name}.{ext}", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--datasets", nargs="+", default=["qsar-biodeg", "credit-g", "jm1"])
    ap.add_argument("--wd", nargs="+", type=float, default=[0.0, 1e-2, 1e-1, 1.0])
    ap.add_argument("--init", nargs="+", type=float, default=[1.0, 4.0, 8.0])
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out:
        OUT = ROOT / a.out
    if not a.plot_only:
        cfg = load_config(a.config)
        for ds, wd, s in itertools.product(a.datasets, a.wd, a.init):
            run(cfg, ds, wd, s, a)
    plot()
