import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

import data_prep

def build_train_sequences(degradation_df: pd.DataFrame, failure_df: pd.DataFrame,) -> Tuple[List[torch.Tensor], torch.Tensor, List[int]]:

    sequences: List[torch.Tensor] = []
    targets: List[float] = []
    item_ids: List[int] = []

    for item_id, df_item in degradation_df.groupby("item_id"):
        df_item = df_item.sort_values("time (months)")

        time = df_item["time (months)"].to_numpy(dtype=np.float32)
        crack = df_item["crack length (arbitary unit)"].to_numpy(dtype=np.float32)

        features = np.stack([time, crack], axis=1)
        seq_tensor = torch.from_numpy(features)

        row = failure_df.loc[failure_df["item_id"] == item_id]
        if row.empty:
            continue
        ttf = float(row["Time to failure (months)"].iloc[0])

        sequences.append(seq_tensor)
        targets.append(ttf)
        item_ids.append(int(item_id))

    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    return sequences, targets_tensor, item_ids


def build_test_sequences(testing_degradation_df: pd.DataFrame,) -> Tuple[List[torch.Tensor], List[int]]:

    sequences: List[torch.Tensor] = []
    item_ids: List[int] = []

    for item_id, df_item in testing_degradation_df.groupby("item_id"):
        df_item = df_item.sort_values("time (months)")

        time = df_item["time (months)"].to_numpy(dtype=np.float32)
        crack = df_item["crack length (arbitary unit)"].to_numpy(dtype=np.float32)

        features = np.stack([time, crack], axis=1)
        seq_tensor = torch.from_numpy(features)

        sequences.append(seq_tensor)
        item_ids.append(int(item_id))

    return sequences, item_ids