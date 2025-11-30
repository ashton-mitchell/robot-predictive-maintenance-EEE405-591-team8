import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
try:
    from sklearn.model_selection import GroupShuffleSplit
except Exception as e:
    raise ImportError(
        "scikit-learn is required to run this script. "
        "Install with: pip install scikit-learn"
    ) from e

from data_prep import (
    load_failure_data,
    load_degradation_data,
    load_pseudo_truth_data,
    load_testing_degradation_data,
    make_training_snapshot_table,
    make_testing_snapshot_table,
    build_training_feature_table,
    build_testing_feature_table,
)

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    print(f"Saved: {path} (shape={df.shape})")

def split_items_by_group(df_snapshot: pd.DataFrame, group_col="item_id", train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state=42):
    """
    Split item ids into train/val/test groups by item id (no leakage).
    Returns dict of lists: {'train': [...], 'val': [...], 'test': [...]}
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"

    unique_items = df_snapshot[group_col].unique()
    unique_items = np.array(unique_items)
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_items)
    n = len(unique_items)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    train_items = unique_items[:n_train].tolist()
    val_items = unique_items[n_train:n_train + n_val].tolist()
    test_items = unique_items[n_train + n_val:].tolist()

    return {"train": train_items, "val": val_items, "test": test_items}

def filter_by_items(df: pd.DataFrame, item_list, item_col="item_id"):
    return df[df[item_col].isin(item_list)].reset_index(drop=True)

def main(base_dir: str, results_dir: str, train_frac: float, val_frac: float, test_frac: float, random_state: int):
    base = Path(base_dir).resolve()
    results = Path(results_dir).resolve()
    ensure_dir(results)

    print("Loading data...")
    failure_df = load_failure_data(str(base))
    degradation_df = load_degradation_data(str(base))
    testing_degradation_df = load_testing_degradation_data(str(base))
    # pseudo is optional; load if available
    try:
        pseudo_df = load_pseudo_truth_data(str(base))
    except Exception:
        pseudo_df = None

    print("Building snapshot tables and features...")
    train_snapshot = make_training_snapshot_table(degradation_df, failure_df) 
    test_snapshot = make_testing_snapshot_table(testing_degradation_df)
    train_features = build_training_feature_table(degradation_df, failure_df) 
    test_features = build_testing_feature_table(testing_degradation_df)

    # split training snapshot item_ids into train/val/test (by item)
    print("Splitting items into train/val/test by item_id...")
    splits = split_items_by_group(train_snapshot, group_col="item_id",
                                  train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
                                  random_state=random_state)
    save_csv(pd.DataFrame({"item_id": splits["train"]}), results / "train_items.csv")
    save_csv(pd.DataFrame({"item_id": splits["val"]}), results / "val_items.csv")
    save_csv(pd.DataFrame({"item_id": splits["test"]}), results / "test_items.csv")

    train_snapshot_df = filter_by_items(train_snapshot, splits["train"])
    val_snapshot_df = filter_by_items(train_snapshot, splits["val"])
    test_snapshot_from_train_df = filter_by_items(train_snapshot, splits["test"])

    train_features_df = filter_by_items(train_features, splits["train"])
    val_features_df = filter_by_items(train_features, splits["val"])
    test_features_from_train_df = filter_by_items(train_features, splits["test"])


    save_csv(train_snapshot_df, results / "train_snapshot.csv")
    save_csv(val_snapshot_df, results / "val_snapshot.csv")
    save_csv(test_snapshot_from_train_df, results / "test_snapshot.csv")

    save_csv(train_features_df, results / "train_features.csv")
    save_csv(val_features_df, results / "val_features.csv")
    save_csv(test_features_from_train_df, results / "test_features_from_train.csv")
    save_csv(test_snapshot, results / "official_test_snapshot.csv")
    save_csv(test_features, results / "official_test_features.csv")

    # print quick class balance for fail_in_4m if present
    if "fail_in_4m" in train_features_df.columns:
        print("\nTrain fail_in_4m distribution:")
        print(train_features_df["fail_in_4m"].value_counts(normalize=True))
    if pseudo_df is not None:
        print("\nPseudo test sample:")
        print(pseudo_df.head())

    print("\nAll done. CSVs saved to:", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare train/val/test snapshots & features (by item_id)")
    parser.add_argument("--base-dir", type=str, default=".", help="Path to repo root (where data_prep.py and data folders live)")
    parser.add_argument("--results-dir", type=str, default="results", help="Where to save CSV outputs")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Fraction of items for training")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Fraction of items for validation")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Fraction of items for test (from training set)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(args.base_dir, args.results_dir, args.train_frac, args.val_frac, args.test_frac, args.random_state)