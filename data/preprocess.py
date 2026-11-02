"""Grinsztajn-style preprocessing: split, truncate train to 10k, quantile transform,
one-hot categoricals, standardize regression targets. Keeps an input_unit -> feature map."""
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, QuantileTransformer

from data.openml_loader import fetch_task


@dataclass
class TabularData:
    name: str
    task_type: str  # "classification" | "regression"
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int  # 0 for regression
    # one row per input unit: unit, original_feature, original_name, is_categorical, is_uninformative
    feature_map: pd.DataFrame
    n_original_features: int
    y_mean: float = 0.0
    y_std: float = 1.0

    @property
    def n_inputs(self):
        return self.X_train.shape[1]

    @property
    def n_outputs(self):
        return self.n_classes if self.task_type == "classification" else 1

    def split(self, name):
        return getattr(self, f"X_{name}"), getattr(self, f"y_{name}")

    def copy(self, **kw):
        return replace(self, **kw)


MAX_CATEGORIES = 64  # one-hot levels per categorical column; above all of the original 8 datasets


def split_indices(n, split_seed, max_train=10_000, max_val=50_000, max_test=50_000):
    """70% train (capped), remaining split 30/70 into val/test (capped), as in Grinsztajn."""
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(n)
    n_train = min(int(0.7 * n), max_train)
    rest = perm[n_train:]
    n_val = min(int(0.3 * len(rest)), max_val)
    val = rest[:n_val]
    test = rest[n_val : n_val + max_test]
    return perm[:n_train], val, test


def preprocess(X: pd.DataFrame, y: pd.Series, meta: dict, split_seed=0, max_train=10_000):
    task_type = meta["task_type"]
    X = X.copy()
    cat_cols = [c for c in X.columns if c in set(meta["categorical"])]
    # string columns the OpenML metadata does not mark as categorical (e.g. Marketing_Campaign
    # `Dt_Customer`, bank-marketing `poutcome`): dates become days since epoch, the rest categorical
    for c in X.columns:
        if c in cat_cols or pd.api.types.is_numeric_dtype(X[c]):
            continue
        dt = pd.to_datetime(X[c], errors="coerce", format="mixed")
        if dt.notna().mean() > 0.99:
            X[c] = (dt - pd.Timestamp("1970-01-01")).dt.days.astype(float)
        else:
            cat_cols.append(c)
    tr, va, te = split_indices(len(X), split_seed, max_train)

    blocks = {s: [] for s in ("train", "val", "test")}
    fmap = []
    unit = 0
    for j, col in enumerate(X.columns):
        parts = {s: X[col].iloc[idx] for s, idx in (("train", tr), ("val", va), ("test", te))}
        if col in cat_cols:
            # ID-like columns (e.g. SDSS17 `plate`, 4435 levels): keep the MAX_CATEGORIES - 1 most
            # frequent levels, the rest share one "infrequent" unit. No effect below the cap.
            enc = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=MAX_CATEGORIES,
                                sparse_output=False)
            enc.fit(parts["train"].astype(str).to_frame())
            for s in blocks:
                blocks[s].append(enc.transform(parts[s].astype(str).to_frame()))
            infreq = set(enc.infrequent_categories_[0]) if enc.infrequent_categories_[0] is not None else set()
            cats = [c for c in enc.categories_[0] if c not in infreq] + (["<infrequent>"] if infreq else [])
            for c in cats:
                fmap.append((unit, j, f"{col}={c}", True, False))
                unit += 1
        else:
            vals = {s: pd.to_numeric(p, errors="coerce").to_numpy(float) for s, p in parts.items()}
            med = np.nanmedian(vals["train"])
            qt = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(1000, len(tr)),
                random_state=split_seed,
            )
            qt.fit(np.nan_to_num(vals["train"], nan=med)[:, None])
            for s in blocks:
                blocks[s].append(qt.transform(np.nan_to_num(vals[s], nan=med)[:, None]))
            fmap.append((unit, j, str(col), False, False))
            unit += 1
    Xs = {s: np.concatenate(b, axis=1).astype(np.float32) for s, b in blocks.items()}

    y_mean, y_std = 0.0, 1.0
    if task_type == "classification":
        le = LabelEncoder().fit(y.astype(str))
        yall = le.transform(y.astype(str)).astype(np.int64)
        n_classes = len(le.classes_)
    else:
        yall = y.to_numpy(float)
        y_mean, y_std = float(yall[tr].mean()), float(yall[tr].std())
        yall = ((yall - y_mean) / y_std).astype(np.float32)
        n_classes = 0

    feature_map = pd.DataFrame(
        fmap, columns=["unit", "original_feature", "original_name", "is_categorical", "is_uninformative"]
    )
    return TabularData(
        name=meta["name"],
        task_type=task_type,
        X_train=Xs["train"], y_train=yall[tr],
        X_val=Xs["val"], y_val=yall[va],
        X_test=Xs["test"], y_test=yall[te],
        n_classes=n_classes,
        feature_map=feature_map,
        n_original_features=X.shape[1],
        y_mean=y_mean, y_std=y_std,
    )


@lru_cache(maxsize=8)
def load_dataset(task_id: int, split_seed: int = 0, max_train: int = 10_000) -> TabularData:
    X, y, meta = fetch_task(task_id)
    return preprocess(X, y, meta, split_seed, max_train)
