"""Single training run. Supports early stopping (tuning only), fixed-epoch training,
saving a rewind checkpoint at step k, and resuming from such a checkpoint."""
import copy

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from common import make_generator
from metrics import compute_metrics


class _Rows(torch.utils.data.Dataset):
    """Dataset indexed by a list of row ids, so each batch is one tensor gather."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.as_tensor(idx)


def make_loader(n, batch_size, generator):
    sampler = BatchSampler(RandomSampler(range(n), generator=generator), batch_size, drop_last=False)
    return DataLoader(_Rows(n), sampler=sampler, batch_size=None)


def steps_per_epoch(n, batch_size):
    return -(-n // batch_size)


def make_scheduler(opt, hp, total_steps):
    kind = hp.get("scheduler", "none")
    if kind == "none":
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))
    raise ValueError(kind)


def loss_fn(task_type):
    if task_type == "classification":
        return torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    return lambda out, y: mse(out.squeeze(-1), y)


@torch.no_grad()
def predict(model, X, device, chunk=8192):
    model.eval()
    outs = [model(torch.as_tensor(X[i : i + chunk], device=device)).cpu() for i in range(0, len(X), chunk)]
    model.train()
    return torch.cat(outs).numpy()


def evaluate(model, data, device, splits=("train", "val", "test")):
    return {s: compute_metrics(data.task_type, data.split(s)[1], predict(model, data.split(s)[0], device)) for s in splits}


def _rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s):
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state_all(s["cuda"])


def train(
    model,
    data,
    hp,
    *,
    n_epochs,
    model_seed,
    device,
    masker=None,
    from_ckpt=None,
    save_ckpt_at=None,
    patience=None,
    eval_splits=("train", "val", "test"),
):
    """Train `model` in place.

    from_ckpt:    rewind checkpoint dict; training continues from its step with the same
                  data order (the loader is replayed with the model_seed generator).
    save_ckpt_at: global step k at which to snapshot model/optimizer/scheduler/RNG.
    patience:     if set, early stop on val loss and restore the best epoch (tuning only).
    Returns dict with metrics, best_epoch, total_steps and (if requested) the checkpoint.
    """
    model.to(device).train()
    Xtr = torch.as_tensor(data.X_train, device=device)
    ytr = torch.as_tensor(data.y_train, device=device)
    bs = hp["batch_size"]
    total_steps = n_epochs * steps_per_epoch(len(Xtr), bs)
    loader = make_loader(len(Xtr), bs, make_generator(model_seed))
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = make_scheduler(opt, hp, total_steps)
    crit = loss_fn(data.task_type)
    l1 = hp.get("l1", 0.0)

    start_step = 0
    if from_ckpt is not None:
        model.load_state_dict(from_ckpt["model"])
        opt.load_state_dict(from_ckpt["opt"])
        sched.load_state_dict(from_ckpt["sched"])
        _set_rng_state(from_ckpt["rng"])
        start_step = from_ckpt["step"]
    if masker is not None:
        masker.apply()

    ckpt = None

    def snapshot(step):
        return {
            "step": step,
            "model": copy.deepcopy(model.state_dict()),
            "opt": copy.deepcopy(opt.state_dict()),
            "sched": copy.deepcopy(sched.state_dict()),
            "rng": _rng_state(),
        }

    if save_ckpt_at == 0 and start_step == 0:
        ckpt = snapshot(0)

    best = (float("inf"), 0, None)  # val loss, epoch, state
    bad = 0
    step = 0
    for epoch in range(1, n_epochs + 1):
        for idx in loader:
            if step < start_step:  # replay data order up to the rewind point
                step += 1
                continue
            idx = idx.to(device)
            out = model(Xtr[idx])
            loss = crit(out, ytr[idx])
            if l1:
                loss = loss + l1 * sum(l.weight.abs().sum() for l in model.linears)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if masker is not None:
                masker.apply()  # AdamW momentum / weight decay must not revive pruned weights
            sched.step()
            step += 1
            if save_ckpt_at is not None and step == save_ckpt_at:
                ckpt = snapshot(step)
        if patience is not None:
            vl = compute_metrics(data.task_type, data.y_val, predict(model, data.X_val, device))["loss"]
            if not np.isfinite(vl):
                break
            if vl < best[0]:
                best = (vl, epoch, copy.deepcopy(model.state_dict()))
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if patience is not None and best[2] is not None:
        model.load_state_dict(best[2])
    return {
        "metrics": evaluate(model, data, device, eval_splits),
        "best_epoch": best[1] if patience is not None else n_epochs,
        "total_steps": total_steps,
        "ckpt": ckpt,
    }
