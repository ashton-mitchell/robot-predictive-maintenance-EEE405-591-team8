import os
import glob
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load failure data
def load_failure_data(base_dir: str) -> pd.DataFrame:
    path = os.path.join(base_dir, "training_data", "training_data", "failure_data.csv")
    failure_df = pd.read_csv(path)
    return failure_df

# load training degradation data
def load_degradation_data(base_dir: str) -> pd.DataFrame:
    pattern = os.path.join(base_dir, "training_data", "training_data", "degradation_data", "item_*.csv")
    frames = []

    for path in sorted(glob.glob(pattern)):
        item_name = os.path.splitext(os.path.basename(path))[0]
        item_idx = int(item_name.split("_")[1])

        df_item = pd.read_csv(path)
        df_item["item_id"] = item_idx
        frames.append(df_item)

    if not frames:
        raise FileNotFoundError(f"No degradation_data files found with pattern: {pattern}")

    degradation_df = pd.concat(frames, ignore_index=True)
    return degradation_df

# load pseudo truth data
def load_pseudo_truth_data(base_dir: str) -> pd.DataFrame:
    path = os.path.join(base_dir, "pseudo_testing_data_with_truth", "Solution.csv")
    pseudo_df = pd.read_csv(path)

    pseudo_df["item_id"] = (
        pseudo_df["item_index"]
        .str.replace("item_", "", regex=False)
        .astype(int)
    )

    pseudo_df["fail_in_4m"] = (pseudo_df["true_rul"] <= 4).astype(int)

    return pseudo_df

# load testing degradation data
def load_testing_degradation_data(base_dir: str) -> pd.DataFrame:
    pattern = os.path.join(base_dir, "testing_data", "testing_item_*.csv")
    frames = []

    for path in sorted(glob.glob(pattern)):
        item_name = os.path.splitext(os.path.basename(path))[0]
        idx_str = item_name.replace("testing_item_", "")
        item_idx = int(idx_str)

        df_item = pd.read_csv(path)
        df_item["item_id"] = item_idx
        frames.append(df_item)

    if not frames:
        raise FileNotFoundError(f"No testing_data files found with pattern: {pattern}")

    testing_degradation_df = pd.concat(frames, ignore_index=True)
    return testing_degradation_df
 
def make_last_observation_table(degradation_df: pd.DataFrame) -> pd.DataFrame:
    last_obs_df = (
        degradation_df
        .sort_values(["item_id", "time (months)"])
        .groupby("item_id")
        .tail(1)
        .reset_index(drop=True)
    )
    return last_obs_df

def make_training_snapshot_table(degradation_df: pd.DataFrame, failure_df: pd.DataFrame) -> pd.DataFrame:
    last_obs_df = make_last_observation_table(degradation_df)

    train_rul_df = last_obs_df.merge(
        failure_df[["item_id", "Time to failure (months)", "Failure mode"]],
        on="item_id",
        how="left",
    )

    train_rul_df.rename(
        columns={"Time to failure (months)": "ttf_months"},
        inplace=True,
    )

    train_rul_df = train_rul_df.drop('rul (months)', axis=1)

    train_rul_df["fail_in_4m"] = (train_rul_df["ttf_months"] <= 4).astype(int)


    return train_rul_df

def make_testing_snapshot_table(testing_degradation_df: pd.DataFrame) -> pd.DataFrame:
    test_last_obs_df = (
        testing_degradation_df
        .sort_values(["item_id", "time (months)"])
        .groupby("item_id")
        .tail(1)
        .reset_index(drop=True)
    )
    return test_last_obs_df

def compute_slope(x, y):
    if len(x) < 2:
        return np.nan
    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return m


def compute_curvature(x, y):
    if len(x) < 3:
        return np.nan
    coeffs = np.polyfit(x, y, 2)
    a = coeffs[0]
    return a


def build_feature_engineered_snapshot(degradation_df):
    feature_rows = []

    for item_id, df_item in degradation_df.groupby("item_id"):
        df_item = df_item.sort_values("time (months)")

        time = df_item["time (months)"].values
        crack = df_item["crack length (arbitrary unit)"].values

        last_time = time[-1]
        last_crack = crack[-1]

        mean_crack = np.mean(crack)
        max_crack = np.max(crack)
        std_crack = np.std(crack)

        global_slope = compute_slope(time, crack)

        slope_3 = compute_slope(time[-3:], crack[-3:]) if len(time) >= 3 else np.nan
        slope_5 = compute_slope(time[-5:], crack[-5:]) if len(time) >= 5 else np.nan

        curvature = compute_curvature(time, crack)

        crack_time_ratio = last_crack / last_time if last_time > 0 else np.nan

        feature_rows.append({
            "item_id": item_id,
            "last_time": last_time,
            "last_crack": last_crack,
            "mean_crack": mean_crack,
            "max_crack": max_crack,
            "std_crack": std_crack,
            "slope_global": global_slope,
            "slope_last3": slope_3,
            "slope_last5": slope_5,
            "curvature": curvature,
            "crack_time_ratio": crack_time_ratio,
        })

    return pd.DataFrame(feature_rows)

def build_training_feature_table(degradation_df, failure_df):
    feat = build_feature_engineered_snapshot(degradation_df)

    labels = failure_df.rename(columns={
        "Time to failure (months)": "rul_months"
    })

    feat = feat.merge(labels[["item_id", "rul_months"]], on="item_id", how="left")

    feat["fail_in_4m"] = (feat["rul_months"] <= 4).astype(int)

    return feat

def build_testing_feature_table(testing_degradation_df):
    return build_feature_engineered_snapshot(testing_degradation_df)

def main():

    # load data
    failure_df = load_failure_data(BASE_DIR)
    degradation_df = load_degradation_data(BASE_DIR)
    pseudo_df = load_pseudo_truth_data(BASE_DIR)
    testing_degradation_df = load_testing_degradation_data(BASE_DIR)

    # build tables
    train_rul_df = make_training_snapshot_table(degradation_df, failure_df)
    test_last_obs_df = make_testing_snapshot_table(testing_degradation_df)

    train_rul_features_df = build_feature_engineered_snapshot(degradation_df)

    # print results
    print("\nfailure_df")
    print(failure_df.head(50))
    print("shape:", failure_df.shape)

    print("\ndegradation_df")
    print(degradation_df.head(50))
    print("shape:", degradation_df.shape)

    print("\npseudo_df")
    print(pseudo_df.head(50))
    print("shape:", pseudo_df.shape)
    print("Agreement between pseudo label and (true_rul <= 4):", (pseudo_df["label"] == pseudo_df["fail_in_4m"]).mean())

    print("\ntrain_rul_df (snapshot training table)")
    print(train_rul_df.head(50))
    print("shape:", train_rul_df.shape)
    print("fail_in_4m value counts:")
    print(train_rul_df["fail_in_4m"].value_counts())

    print("\ntest_last_obs_df (snapshot testing table)")
    print(test_last_obs_df.head(50))
    print("shape:", test_last_obs_df.shape)

    print("\ntrain_rul_features_df (engineered snapshot training table)")
    print(train_rul_features_df.head(50))

if __name__ == "__main__":
    main()
