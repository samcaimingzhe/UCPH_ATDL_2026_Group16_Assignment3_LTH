import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import torch

from common import get_device, set_seed
from data.noise import apply_noise
from data.preprocess import preprocess
from models.mlp import build_mlp
from pruning.masks import Masker, magnitude_prune, target_kept

HP = {"depth": 3, "width": 64, "dropout": 0.0, "lr": 1e-3, "weight_decay": 0.0, "batch_size": 128}
DEVICE = get_device()


@pytest.fixture(scope="module")
def data():
    rng = np.random.RandomState(0)
    n = 600
    X = pd.DataFrame({f"x{i}": rng.randn(n) for i in range(40)})
    X["cat"] = rng.choice(list("abc"), n)
    y = pd.Series((X.x0 + X.x1 > 0).astype(int) + (X.x2 > 1).astype(int)).astype(str)
    return preprocess(X, y, {"name": "toy", "task_type": "classification", "categorical": ["cat"]})


def _prune_rounds(data, mode, rounds=8):
    set_seed(0)
    model = build_mlp(data, HP).to(DEVICE)
    masker = Masker(model)
    sizes = masker.sizes()
    for r in range(1, rounds + 1):
        prev = masker.masks
        masker.masks = magnitude_prune(model, prev, r, 0.2, 0.5, mode)
        for n in prev:  # nested
            assert not (masker.masks[n] & ~prev[n]).any()
        yield r, masker, sizes


def test_uniform_each_layer_matches(data):
    for r, masker, sizes in _prune_rounds(data, "uniform"):
        kept = masker.kept()
        for n in sizes:
            rate = 0.1 if n == "out" else 0.2
            assert abs(kept[n] - sizes[n] * (1 - rate) ** r) <= 1


@pytest.mark.parametrize("mode,layer", [("single_input", "fc0"), ("single_middle", "fc1"), ("single_last", "fc2")])
def test_single_layer_only_prunes_target(data, mode, layer):
    for r, masker, sizes in _prune_rounds(data, mode):
        kept = masker.kept()
        for n in sizes:
            if n != layer:
                assert kept[n] == sizes[n]
        k_hidden, k_out = target_kept(sizes, r, 0.2, 0.5)
        floor = max(1, round(0.01 * sizes[layer]))
        if kept[layer] > floor:  # feasible: total matches global within one weight
            assert abs(sum(kept.values()) - (k_hidden + k_out)) <= 1
        else:
            assert kept[layer] == floor


@pytest.mark.parametrize("kind,level", [("label_flip", 0.2), ("feature_gaussian", 0.5), ("uninformative", 1.0)])
def test_noise_reproducible_and_test_clean(data, kind, level):
    a = apply_noise(data, kind, level, noise_seed=1002)
    b = apply_noise(data, kind, level, noise_seed=1002)
    c = apply_noise(data, kind, level, noise_seed=1003)
    for s in ("train", "val", "test"):
        assert np.array_equal(a.split(s)[0], b.split(s)[0]) and np.array_equal(a.split(s)[1], b.split(s)[1])
    assert not (np.array_equal(a.X_train, c.X_train) and np.array_equal(a.y_train, c.y_train))
    d0 = data.n_inputs
    assert np.array_equal(a.X_test[:, :d0], data.X_test) and np.array_equal(a.y_test, data.y_test)


def test_label_flip_exact_fraction(data):
    a = apply_noise(data, "label_flip", 0.2, noise_seed=1002)
    changed = (a.y_train != data.y_train).mean()
    assert abs(changed - 0.2) < 1.5 / len(data.y_train)


def test_uninformative_feature_map(data):
    a = apply_noise(data, "uninformative", 0.5, noise_seed=1001)
    n_new = round(0.5 * data.n_original_features)
    assert a.n_inputs == data.n_inputs + n_new
    fm = a.feature_map
    assert fm.is_uninformative.sum() == n_new and len(fm) == a.n_inputs
    assert fm[fm.is_uninformative].original_feature.min() == data.n_original_features
    # permute keeps marginals: each new column is a permutation of some existing column
    col = a.X_train[:, data.n_inputs]
    assert any(np.array_equal(np.sort(col), np.sort(data.X_train[:, j])) for j in range(data.n_inputs))


def test_gaussian_noise_not_on_onehot(data):
    a = apply_noise(data, "feature_gaussian", 1.0, noise_seed=1003)
    cat_units = data.feature_map.unit[data.feature_map.is_categorical].to_numpy()
    assert np.array_equal(a.X_train[:, cat_units], data.X_train[:, cat_units])
