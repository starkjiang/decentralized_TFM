"""
dicl/data.py
=============
Dataset loaders and IID / non-IID data partitioning.

Classification datasets
-----------------------
  phoneme · vehicle · kr_vs_kp · wine_red · digits

Regression datasets
-------------------
  bike · diabetes_reg · wine · energy · concrete

All loaders return (X_train, X_test, y_train, y_test, meta).
Features are standardised with StandardScaler.
Regression targets are also standardised (meta["y_scaler"] holds the fitted scaler).

Data sources:
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/index.php
- Scikit-learn datasets: https://scikit-learn.org/stable/datasets.html
- OpenML: https://www.openml.org/

For datasets sourced from OpenML, you will need to download them manually to 
your local cache directory first. Then you need to provide the correct data path
in the code. The data format is .arff. There is deciated function defined to load .arff files.
Without manually downloading, the code will throw error when loading datasets from OpenML.
"""

import pandas as pd
from typing import Tuple

import io
import numpy as np
from pathlib import Path
from sklearn.datasets import load_digits
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from .config import Config

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"  # package-relative dataset dir

SEED = 42

# ── Types ─────────────────────────────────────────────────────────────────────
DataTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]

# =============================================================================
# Main data loading function
# =============================================================================
def encode_arff(data_path):
    data, meta = arff.loadarff(str(data_path))
    df = pd.DataFrame(data)
    # Note: ARFF loading often results in byte strings (e.g., b'1'),
    # so we decode them to ensure string/numeric types are correct.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')
    return meta, df

# =============================================================================
# Classification loaders
# =============================================================================

def load_clf(name: str, cfg: Config) -> DataTuple:
    """Load and pre-process a classification dataset by short name."""
    if name == "phoneme":
        meta, df = encode_arff(DATA_DIR / "phoneme.arff")
        target_col = meta.names()[-1]
        # Separate features and target
        X_raw = df.drop(columns=[target_col])
        y_raw = df[target_col]
        # Apply your original processing logic
        X = X_raw.select_dtypes(include="number").fillna(0).values.astype(float)
        y = LabelEncoder().fit_transform(y_raw.astype(str))
        label = "Phoneme (UCI)"
        cnames = ["non-nasal", "nasal"]
    elif name == "vehicle":
        meta, df = encode_arff(DATA_DIR / "vehicle.arff")
        target_col = meta.names()[-1]
        # Split features and target
        df_data = df.drop(columns=[target_col])
        df_target = df[target_col]
        # Apply your original processing logic
        X = df_data.select_dtypes(include="number").fillna(0).values.astype(float)
        # Label Encoding for target and capturing class names
        le = LabelEncoder()
        y = le.fit_transform(df_target.astype(str))
        label = "Vehicle Silhouettes (UCI)"
        cnames = list(le.classes_)
    elif name == "kr_vs_kp":
        meta, df = encode_arff(DATA_DIR / "kr-vs-kp.arff")
        target_col = meta.names()[-1]
        df_data = df.drop(columns=[target_col])
        df_target = df[target_col]
        # Apply your original processing logic
        # We use .astype(str) to ensure the OrdinalEncoder receives clean strings
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X = enc.fit_transform(df_data.astype(str)).astype(float)
        y = LabelEncoder().fit_transform(df_target.astype(str))
        label = "Chess KR-vs-KP (UCI)"
        cnames = ["no-win", "win"]
    elif name == "wine_red":
        meta, df = encode_arff(DATA_DIR / "wine_quality_red.arff")
        # Identify target column (usually 'class' or 'quality')
        target_col = meta.names()[-1]
        df_data = df.drop(columns=[target_col])
        y_raw = df[target_col].values
        # Features (matching your np.float32 requirement)
        X = np.asarray(df_data, dtype=np.float32)
        # Labels (matching your integer encoding logic)
        # Note: SciPy often loads target as bytes; we decode if necessary
        if y_raw.dtype.kind in 'OS':  # Object or String (including bytes)
            # Decode bytes if they exist
            y_raw = np.array([val.decode('utf-8') if isinstance(val, bytes) else val for val in y_raw])
        # Apply your logic to ensure integer encoding
        y = y_raw.astype(int) if np.issubdtype(y_raw.dtype, np.integer) or y_raw.dtype.kind in "iu" else LabelEncoder().fit_transform(y_raw)
        label = "Wine Quality (Red)"
        # class names (sorted unique labels)
        cnames = [str(c) for c in sorted(np.unique(y))]
    elif name == "digits":
        r      = load_digits()
        X, y   = r.data.astype(float), r.target.astype(int)
        label  = "Digits (NIST)"
        cnames = [str(i) for i in range(10)]
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
    if name == "bike":
        meta, df = encode_arff(DATA_DIR / "bike.arff")
        X = df.drop(columns=['count', 'time', 'dayOfWeek', 'datetime']).values.astype(float)
        y = df['count'].values.astype(float)
        label = "Bike Sharing (Hourly)"

    elif name == "wine":
        meta, df = encode_arff(DATA_DIR / "wine_quality.arff")
        # We maintain your logic using np.float64 for both X and y
        # Note: If the column in your ARFF is named 'class', change 'quality' accordingly
        target_name = 'quality' if 'quality' in df.columns else meta.names()[-1]
        X = df.drop(columns=[target_name]).values.astype(np.float64)
        y = df[target_name].values.astype(np.float64)
        label = "Wine Quality"

    elif name=="diabetes_reg":
        meta, df = encode_arff(DATA_DIR / "diabetes.arff")
        # Handle target (usually the last column)
        target_col = meta.names()[-1]
        # Extract and cast to float as per your original code
        X = df.drop(columns=[target_col]).values.astype(float)
        y = df[target_col].values.astype(float)
        label = "Diabetes-Reg (UCI)"
    elif name == "energy":
        meta, df = encode_arff(DATA_DIR / "energy.arff")
        # Separate Features and Target
        # In this dataset, targets are usually the last two columns:
        # 'y1' (Heating) and 'y2' (Cooling) or similar names.
        # We identify them by their position to match your original logic.
        target_cols = meta.names()[-2:] # Take the last two columns
        df_data = df.drop(columns=target_cols)
        df_targets = df[target_cols]
        # Features logic
        X = df_data.values.astype(float)
        # Target logic (Heating Load)
        # Your logic: if y has multiple columns, take the first one (Heating Load)
        y_raw = df_targets.values
        if y_raw.ndim > 1:
            y = y_raw[:, 0]  # Take Heating Load
        else:
            y = y_raw
        y = y.astype(float)
        label = "Energy Efficiency (UCI)"
    elif name == "concrete":
        with open(DATA_DIR / "concrete.arff", 'r') as f:
            lines = f.readlines()
        # Find the @data trigger
        data_start = 0
        for i, line in enumerate(lines):
            if line.lower().startswith('@data'):
                data_start = i + 1
                break
        # Extract lines after @data and filter out comments (%) or empty lines
        data_lines = [
            l for l in lines[data_start:]
            if l.strip() and not l.strip().startswith('%')
        ]
        # Load the cleaned lines into a DataFrame
        # Using engine='python' and sep=',' to be robust
        df = pd.read_csv(io.StringIO("".join(data_lines)), header=None, sep=',')
        # Processing logic
        # Concrete typically has 8 features + 1 target
        X = df.iloc[:, :-1].values.astype(float)
        y = df.iloc[:, -1].values.astype(float)
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
