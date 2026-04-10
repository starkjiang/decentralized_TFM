"""
dicl/data.py
=============
Dataset loaders and IID / non-IID data partitioning.

Classification datasets
-----------------------
  breast_cancer · wine · iris · digits · diabetes_clf

Regression datasets
-------------------
  california · diabetes_reg · linnerud · energy · concrete

All loaders return (X_train, X_test, y_train, y_test, meta).
Features are standardised with StandardScaler.
Regression targets are also standardised (meta["y_scaler"] holds the fitted scaler).
"""

from typing import Tuple

import numpy as np
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_digits,
    load_iris, load_diabetes, load_linnerud,
    fetch_california_housing, fetch_openml,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import Config

SEED = 42

# ── Types ─────────────────────────────────────────────────────────────────────
DataTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]


# =============================================================================
# Classification loaders
# =============================================================================

def load_clf(name: str, cfg: Config) -> DataTuple:
    """Load and pre-process a classification dataset by short name."""
    if name == "breast_cancer":
        r      = load_breast_cancer()
        X, y   = r.data.astype(float), r.target.astype(int)
        label  = "Breast Cancer (UCI)"
        cnames = list(r.target_names)

    elif name == "wine":
        r      = load_wine()
        X, y   = r.data.astype(float), r.target.astype(int)
        label  = "Wine (UCI)"
        cnames = list(r.target_names)

    elif name == "iris":
        r      = load_iris()
        X, y   = r.data.astype(float), r.target.astype(int)
        label  = "Iris (Fisher)"
        cnames = list(r.target_names)

    elif name == "digits":
        r      = load_digits()
        X, y   = r.data.astype(float), r.target.astype(int)
        label  = "Digits (NIST)"
        cnames = [str(i) for i in range(10)]

    elif name == "diabetes_clf":
        r      = load_diabetes()
        X      = r.data.astype(float)
        y      = (r.target > np.median(r.target)).astype(int)
        label  = "Diabetes-Clf"
        cnames = ["low", "high"]

    else:
        raise ValueError(f"Unknown classification dataset: {name!r}")

    meta = {
        "name": label, "short": name, "task": "classification",
        "n_classes": int(len(np.unique(y))),
        "n_features": int(X.shape[1]),
        "N": int(len(y)),
        "class_names": cnames,
    }

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg.test_size, random_state=SEED, stratify=y
    )
    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    print(
        f"  [CLF] {label:35s}  "
        f"N_tr={len(ytr):5d}  d={meta['n_features']:3d}  C={meta['n_classes']}"
    )
    return Xtr, Xte, ytr, yte, meta


# =============================================================================
# Regression loaders
# =============================================================================

def load_reg(name: str, cfg: Config) -> DataTuple:
    """Load and pre-process a regression dataset by short name."""
    if name == "california":
        r     = fetch_california_housing()
        X, y  = r.data.astype(float), r.target.astype(float)
        label = "California Housing"

    elif name == "diabetes_reg":
        r     = load_diabetes()
        X, y  = r.data.astype(float), r.target.astype(float)
        label = "Diabetes-Reg (UCI)"

    elif name == "linnerud":
        r     = load_linnerud()
        X     = r.data.astype(float)
        y     = r.target[:, 0].astype(float)   # Waist column
        label = "Linnerud (Exercise)"

    elif name == "energy":
        data  = fetch_openml(name="energy-efficiency", version=1, as_frame=False)
        X     = data.data.astype(float)
        y     = data.target
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = y[:, 0]                         # Heating Load
        y     = y.astype(float)
        label = "Energy Efficiency (UCI)"

    elif name == "concrete":
        data  = fetch_openml(data_id=4353, as_frame=True)
        df    = data.frame
        target_col = "Concrete compressive strength(MPa. megapascals)"
        X     = df.drop(columns=[target_col]).values.astype(float)
        y     = df[target_col].values.astype(float)
        label = "Concrete Strength (UCI)"

    else:
        raise ValueError(f"Unknown regression dataset: {name!r}")

    meta = {
        "name": label, "short": name, "task": "regression",
        "n_features": int(X.shape[1]),
        "N": int(len(y)),
    }

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg.test_size, random_state=SEED
    )

    scx = StandardScaler()
    scy = StandardScaler()
    Xtr = scx.fit_transform(Xtr)
    Xte = scx.transform(Xte)
    ytr = scy.fit_transform(ytr.reshape(-1, 1)).ravel()
    yte = scy.transform(yte.reshape(-1, 1)).ravel()
    meta["y_scaler"] = scy

    print(
        f"  [REG] {label:35s}  "
        f"N_tr={len(ytr):5d}  d={meta['n_features']:3d}"
    )
    return Xtr, Xte, ytr, yte, meta


# =============================================================================
# Partitioning
# =============================================================================

def partition_iid(
    X: np.ndarray,
    y: np.ndarray,
    K: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Uniformly random IID split into K disjoint shards."""
    idx = np.random.permutation(len(X))
    return [(X[s], y[s]) for s in np.array_split(idx, K)]


def partition_noniid(
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    alpha: float = 0.5,
    m_0: int = 64,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Dirichlet non-IID split.

    For regression targets, the continuous target is first discretised into K
    quantile bins so that Dirichlet sampling can be applied per class.

    Parameters
    ----------
    alpha : Dirichlet concentration — lower → more heterogeneous split.
    m_0   : minimum shard size (ensures agents have enough context examples).
    """
    is_float = np.issubdtype(y.dtype, np.floating)
    y_cls    = (
        np.clip(
            np.digitize(y, np.quantile(y, np.linspace(0, 1, K + 1)[1:-1])),
            0, K - 1,
        )
        if is_float else y.astype(int)
    )

    classes   = np.unique(y_cls)
    per_class = {c: np.where(y_cls == c)[0].tolist() for c in classes}
    for c in classes:
        np.random.shuffle(per_class[c])

    agent_idx = [[] for _ in range(K)]
    for c in classes:
        props = np.random.dirichlet(np.full(K, alpha))
        cuts  = (props * len(per_class[c])).astype(int)
        cuts[-1] = len(per_class[c]) - cuts[:-1].sum()
        splits = np.split(per_class[c], np.cumsum(cuts)[:-1])
        for k, spl in enumerate(splits):
            agent_idx[k].extend(spl)

    result = []
    min_sz = max(5, m_0 // 2)
    for k in range(K):
        idx = np.array(agent_idx[k], dtype=int)
        if len(idx) < min_sz:
            idx = np.random.choice(len(X), min(min_sz, len(X)), replace=False)
        result.append((X[idx], y[idx]))
    return result
