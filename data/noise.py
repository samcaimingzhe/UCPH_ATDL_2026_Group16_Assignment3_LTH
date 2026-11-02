"""Noise injection on preprocessed data. Deterministic given noise_seed, so every method at
the same noise level sees the identical noisy dataset. Test stays clean unless apply_to_test.

kinds:
  label_flip       level = fraction of rows whose label is changed to a different random class
  feature_gaussian level = std multiplier; adds level * train_std(col) * N(0,1) to numeric units
  uninformative    level = ratio; adds round(level * n_original_features) extra columns,
                   method "permute" (a random existing unit with rows shuffled) or "gaussian".
                   These columns are always added to every split (the model needs the same
                   input dim); for test they are generated independently but never carry signal.
"""
import numpy as np
import pandas as pd

from data.preprocess import TabularData

SPLITS = ("train", "val", "test")


def apply_noise(data: TabularData, kind, level, noise_seed, apply_to_test=False,
                uninformative_method="permute", gaussian_on_categorical=False) -> TabularData:
    if kind in (None, "none") or level == 0:
        return data
    rng = np.random.RandomState(noise_seed)
    noisy_splits = SPLITS if apply_to_test else ("train", "val")
    X = {s: data.split(s)[0].copy() for s in SPLITS}
    y = {s: data.split(s)[1].copy() for s in SPLITS}
    fmap = data.feature_map

    if kind == "label_flip":
        if data.task_type != "classification":
            raise ValueError("label_flip is classification only")
        K = data.n_classes
        for s in noisy_splits:
            n_flip = int(round(level * len(y[s])))
            idx = rng.choice(len(y[s]), n_flip, replace=False)
            y[s][idx] = (y[s][idx] + rng.randint(1, K, size=n_flip)) % K  # always a different class
    elif kind == "feature_gaussian":
        cols = np.arange(data.n_inputs) if gaussian_on_categorical else fmap.unit[~fmap.is_categorical].to_numpy()
        std = data.X_train[:, cols].std(0)
        for s in noisy_splits:
            X[s][:, cols] += (level * std * rng.randn(len(X[s]), len(cols))).astype(np.float32)
    elif kind == "uninformative":
        n_new = int(round(level * data.n_original_features))
        src = rng.randint(0, data.n_inputs, size=n_new)
        for s in SPLITS:
            if uninformative_method == "permute":
                new = np.stack([X[s][rng.permutation(len(X[s])), j] for j in src], axis=1) if n_new else np.empty((len(X[s]), 0))
            elif uninformative_method == "gaussian":
                new = rng.randn(len(X[s]), n_new)
            else:
                raise ValueError(uninformative_method)
            X[s] = np.concatenate([X[s], new.astype(np.float32)], axis=1)
        extra = pd.DataFrame({
            "unit": np.arange(data.n_inputs, data.n_inputs + n_new),
            "original_feature": np.arange(data.n_original_features, data.n_original_features + n_new),
            "original_name": [f"uninf{j}" for j in range(n_new)],
            "is_categorical": False,
            "is_uninformative": True,
        })
        fmap = pd.concat([fmap, extra], ignore_index=True)
        return data.copy(
            X_train=X["train"], X_val=X["val"], X_test=X["test"], feature_map=fmap,
            n_original_features=data.n_original_features + n_new,
        )
    else:
        raise ValueError(kind)
    return data.copy(X_train=X["train"], y_train=y["train"], X_val=X["val"], y_val=y["val"],
                     X_test=X["test"], y_test=y["test"])
