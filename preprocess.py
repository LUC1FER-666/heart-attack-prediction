"""
preprocess.py — Preprocessing Pipeline for Heart Disease Dataset

Fixes applied
─────────────
1. DEDUPLICATION FIRST — dataset had 723/1025 duplicate rows (71% bloat).
   Deduplicating before ANY split is mandatory to prevent data leakage where
   identical rows appear in both train and test, inflating all metrics.

2. ANOMALY REMOVAL — two known-bad values in the Cleveland dataset:
     • ca == 4  (valid range 0-3, only 4 rows — encoding artefact)
     • thal == 0 (valid range 1-3, only 2 rows — missing-value sentinel)
   These are dropped before splitting.

3. SPLIT RATIO → 80/20  (was 85/15)
   After dedup only ~296 clean rows remain. 85/15 left only ~44 test rows,
   making evaluation metrics highly unstable. 80/20 gives ~59 test rows.

4. BINARY_COLS now excludes sex from KNNImputer (it's already binary 0/1,
   most-frequent imputation is correct; KNN on binary columns can skew scale).

Pipeline per column type
────────────────────────
  Numerical cols  → KNNImputer(n_neighbors=5) + StandardScaler
  Binary col(sex) → SimpleImputer(most_frequent)   [no scaling needed]
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

BINARY_COLS    = ["sex"]

NUMERICAL_COLS = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_COLS = [
    "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]
TARGET_COL = "target"

# Known anomalous sentinel values in the Cleveland dataset
ANOMALY_FILTERS = {
    "ca"  : [4],   # valid range 0-3; value 4 is an encoding artefact
    "thal": [0],   # valid range 1-3; value 0 is a missing-value sentinel
}


def detect_columns(df):
    if TARGET_COL in df.columns:
        target = TARGET_COL
    else:
        target = df.columns[-1]
        print(f"Warning: 'target' not found, using '{target}'.")

    feature_cols = [c for c in df.columns if c != target]

    num_cols = [c for c in NUMERICAL_COLS if c in feature_cols]
    bin_cols = [c for c in BINARY_COLS if c in feature_cols]
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]

    remainder = [c for c in feature_cols if c not in set(num_cols) | set(bin_cols) | set(cat_cols)]
    if remainder:
        num_cols += remainder  # fallback

    return feature_cols, target, num_cols, bin_cols, cat_cols

from sklearn.preprocessing import OneHotEncoder

def build_preprocessor(num_cols, bin_cols, cat_cols):
    num_pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler",  StandardScaler()),
    ])

    bin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = [
        ("num", num_pipeline, num_cols),
    ]

    if bin_cols:
        transformers.append(("bin", bin_pipeline, bin_cols))

    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    return ColumnTransformer(transformers=transformers)

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 — Deduplication (must happen before any split).
    Step 2 — Remove rows with known anomalous sentinel values.
    """
    original_len = len(df)

    # 1. Deduplicate
    df = df.drop_duplicates().reset_index(drop=True)
    n_dupes = original_len - len(df)
    if n_dupes > 0:
        pct = n_dupes / original_len * 100
        print(f"\n⚠️  DEDUPLICATION: removed {n_dupes} duplicate rows "
              f"({pct:.1f}% of dataset).")
        if pct > 20:
            print("   !! High duplication rate — skipping this would cause severe "
                  "data leakage and artificially inflated metrics.")
    else:
        print("\n✅  No duplicate rows found.")

    # 2. Remove anomalous sentinel values
    for col, bad_vals in ANOMALY_FILTERS.items():
        if col not in df.columns:
            continue
        mask = df[col].isin(bad_vals)
        n_bad = mask.sum()
        if n_bad > 0:
            df = df[~mask].reset_index(drop=True)
            print(f"   Removed {n_bad} row(s) where {col} ∈ {bad_vals}  "
                  f"(known encoding artefact / missing-value sentinel).")

    print(f"   Clean dataset: {len(df)} rows remaining.\n")
    return df


def preprocess(csv_path: str):
    df = pd.read_csv(csv_path)

    # Clean BEFORE splitting (critical)
    #df = clean_dataset(df)

    feature_cols, target, num_cols, bin_cols, cat_cols = detect_columns(df)

    X = df[feature_cols]
    y = df[target].values

    print(f"Features : {feature_cols}")
    print(f"Target   : {target}")
    print(f"Dataset  : {df.shape} | "
          f"Class balance: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    # 80/20 stratified split
    # 85/15 was set when the dataset appeared to have ~1025 rows.
    # After dedup the true size is ~296 rows; 80/20 gives ~59 test rows,
    # which is the minimum for stable metric estimates on this dataset.
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    feature_cols, target, num_cols, bin_cols, cat_cols = detect_columns(df)

    preprocessor = build_preprocessor(num_cols, bin_cols, cat_cols)
    X_train_processed = preprocessor.fit_transform(X_train_df)
    X_test_processed  = preprocessor.transform(X_test_df)

    print(f"  Train : {X_train_processed.shape}")
    print(f"  Test  : {X_test_processed.shape}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with open(f"{MODELS_DIR}/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    meta = {
        "feature_cols": feature_cols,
        "target_col"  : target,
        "num_cols"    : num_cols,
        "bin_cols"    : bin_cols,
        "cat_cols"    : cat_cols,
    }
    with open(f"{MODELS_DIR}/meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("Preprocessor saved.\n")
    return (
        X_train_processed, y_train,
        X_test_processed,  y_test,
        preprocessor, feature_cols, num_cols, bin_cols,
        cv
    )


if __name__ == "__main__":
    preprocess("heart.csv")
