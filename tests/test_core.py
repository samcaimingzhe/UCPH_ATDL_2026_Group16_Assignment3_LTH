import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import torch

from common import get_device, set_seed
from data.preprocess import preprocess
from models.mlp import build_mlp
from pruning.masks import (
    Masker,
    magnitude_prune,
    random_mask_global,
    random_mask_layerwise,
    target_kept,
)
from train import train

HP = {"depth": 3, "width": 64, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 128}
DEVICE = get_device()


@pytest.fixture(scope="module", params=["classification", "regression"])
def data(request):
    rng = np.random.RandomState(0)
    n = 800
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(6)})
    X["cat"] = rng.choice(list("abc"), n)
    if request.param == "classification":
        y = pd.Series((X.x0 + X.x1 > 0).astype(int).astype(str))
    else:
        y = pd.Series(X.x0 * 3 + X.x1 + rng.randn(n) * 0.1)
    meta = {"name": "toy", "task_type": request.param, "categorical": ["cat"]}
    return preprocess(X, y, meta, split_seed=0)


def test_feature_map(data):
    fm = data.feature_map
    assert len(fm) == data.n_inputs == 6 + 3
    assert (fm[fm.is_categorical].original_feature == 6).all()


def test_pruned_weights_stay_zero(data):
    set_seed(0)
    model = build_mlp(data, HP).to(DEVICE)
    masker = Masker(model)
    masker.masks = magnitude_prune(model, masker.masks, round_idx=3)
    train(model, data, HP, n_epochs=5, model_seed=0, device=DEVICE, masker=masker)
    for n, l in model.named_linears().items():
        assert (l.weight[~masker.masks[n]] == 0).all(), n


def test_sparsity_matches_target(data):
    set_seed(0)
    model = build_mlp(data, HP).to(DEVICE)
    masker = Masker(model)
    sizes = masker.sizes()
    for r in range(1, 16):
        masker.masks = magnitude_prune(model, masker.masks, round_idx=r, rate=0.2, out_factor=0.5)
        k_hidden, k_out = target_kept(sizes, r, 0.2, 0.5)
        kept = masker.kept()
        assert abs(sum(v for k, v in kept.items() if k != "out") - k_hidden) <= 1
        assert abs(kept["out"] - k_out) <= 1
        with torch.no_grad():  # perturb weights so rankings change between rounds
            for l in model.linears:
                l.weight.add_(torch.randn_like(l.weight) * 0.01)
    assert abs(masker.remaining_frac() - 0.8**15) < 0.01


def test_masks_nested(data):
    set_seed(0)
    model = build_mlp(data, HP).to(DEVICE)
    masker = Masker(model)
    prev = masker.masks
    for r in range(1, 5):
        new = magnitude_prune(model, prev, round_idx=r)
        for n in new:
            assert not (new[n] & ~prev[n]).any()
        prev = new


def _run(data, seed):
    set_seed(seed)
    model = build_mlp(data, HP)
    res = train(model, data, HP, n_epochs=3, model_seed=seed, device=DEVICE, save_ckpt_at=7)
    return model, res


def test_determinism(data):
    m1, r1 = _run(data, 42)
    m2, r2 = _run(data, 42)
    assert r1["metrics"] == r2["metrics"]
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_rewind_resume_reproduces_training(data):
    """Training from the step-k checkpoint with an all-ones mask must reproduce the
    original run exactly (same data order, same dropout RNG)."""
    m1, r1 = _run(data, 42)
    set_seed(42)
    m2 = build_mlp(data, HP)
    r2 = train(m2, data, HP, n_epochs=3, model_seed=42, device=DEVICE, from_ckpt=r1["ckpt"], masker=Masker(m2.to(DEVICE)))
    assert r1["metrics"] == r2["metrics"]


def test_random_layerwise_matches_ticket(data):
    set_seed(0)
    model = build_mlp(data, HP).to(DEVICE)
    masks = magnitude_prune(model, Masker(model).masks, round_idx=6)
    rnd = random_mask_layerwise(masks, seed=2042)
    for n in masks:
        assert int(rnd[n].sum()) == int(masks[n].sum())
        assert rnd[n].shape == masks[n].shape
    g = random_mask_global(masks, seed=2042)
    assert sum(int(m.sum()) for m in g.values()) == sum(int(m.sum()) for m in masks.values())
    # reproducible
    rnd2 = random_mask_layerwise(masks, seed=2042)
    assert all(torch.equal(rnd[n], rnd2[n]) for n in masks)
