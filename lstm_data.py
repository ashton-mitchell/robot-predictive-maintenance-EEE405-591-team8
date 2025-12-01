import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import data_prep


def compute_crack_norm_stats(degradation_df: pd.DataFrame) -> Tuple[float, float]:
    crack_all = degradation_df["crack length (arbitary unit)"].to_numpy(dtype=np.float32)
    crack_mean = float(crack_all.mean())
    crack_std = float(crack_all.std())
    if crack_std == 0.0:
        crack_std = 1.0
    return crack_mean, crack_std


def build_train_sequences(
    degradation_df: pd.DataFrame,
    failure_df: pd.DataFrame,
) -> Tuple[List[torch.Tensor], torch.Tensor, List[int]]:
    sequences: List[torch.Tensor] = []
    targets: List[float] = []
    item_ids: List[int] = []

    crack_mean, crack_std = compute_crack_norm_stats(degradation_df)

    max_prefixes_per_item = 5
    min_prefix_len = 3
    late_start_frac = 0.6  # start sampling prefixes from ~60% into the sequence

    for item_id, df_item in degradation_df.groupby("item_id"):
        df_item = df_item.sort_values("time (months)")

        time_raw = df_item["time (months)"].to_numpy(dtype=np.float32)
        crack_raw = df_item["crack length (arbitary unit)"].to_numpy(dtype=np.float32)

        row = failure_df.loc[failure_df["item_id"] == item_id]
        if row.empty:
            continue
        t_fail = float(row["Time to failure (months)"].iloc[0])

        n = len(time_raw)
        if n < min_prefix_len:
            continue

        # Prefer prefixes from the later part of life
        late_start_idx = int(n * late_start_frac)
        base_start_idx = min_prefix_len - 1
        start_idx = max(base_start_idx, late_start_idx)
        end_idx = n - 1

        if start_idx > end_idx:
            start_idx = max(base_start_idx, n - 2)
            end_idx = n - 1
            if start_idx > end_idx:
                continue

        num_candidates = end_idx - start_idx + 1
        if num_candidates <= 0:
            continue

        num_prefixes = min(max_prefixes_per_item, num_candidates)
        prefix_indices = np.linspace(
            start_idx,
            end_idx,
            num=num_prefixes,
            dtype=int,
        )

        t_max = time_raw.max()
        if t_max <= 0:
            t_max = 1.0

        for k in prefix_indices:
            time_prefix_raw = time_raw[: k + 1]
            crack_prefix_raw = crack_raw[: k + 1]

            time_norm = time_prefix_raw / t_max
            crack_norm = (crack_prefix_raw - crack_mean) / crack_std

            features = np.stack([time_norm, crack_norm], axis=1)
            seq_tensor = torch.from_numpy(features)

            t_current = float(time_prefix_raw[-1])
            rul_k = t_fail - t_current

            sequences.append(seq_tensor)
            targets.append(rul_k)
            item_ids.append(int(item_id))

    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    return sequences, targets_tensor, item_ids




def build_test_sequences(
    testing_degradation_df: pd.DataFrame,
    crack_mean: float,
    crack_std: float,
) -> Tuple[List[torch.Tensor], List[int]]:
    sequences: List[torch.Tensor] = []
    item_ids: List[int] = []

    for item_id, df_item in testing_degradation_df.groupby("item_id"):
        df_item = df_item.sort_values("time (months)")

        time_raw = df_item["time (months)"].to_numpy(dtype=np.float32)
        crack_raw = df_item["crack length (arbitary unit)"].to_numpy(dtype=np.float32)

        t_max = time_raw.max()
        if t_max > 0:
            time = time_raw / t_max
        else:
            time = time_raw

        crack = (crack_raw - crack_mean) / crack_std

        features = np.stack([time, crack], axis=1)
        seq_tensor = torch.from_numpy(features)

        sequences.append(seq_tensor)
        item_ids.append(int(item_id))

    return sequences, item_ids


class RULSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: List[torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        item_ids: Optional[List[int]] = None,
    ) -> None:
        self.sequences = sequences
        self.targets = targets
        self.item_ids = item_ids if item_ids is not None else list(range(len(sequences)))

        if self.targets is not None:
            assert len(self.sequences) == len(self.targets) == len(self.item_ids)
        else:
            assert len(self.sequences) == len(self.item_ids)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        item_id = self.item_ids[idx]
        if self.targets is None:
            return seq, item_id
        else:
            target = self.targets[idx]
            return seq, target, item_id


def collate_train(batch):
    sequences, targets, item_ids = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    padded_seqs = pad_sequence(sequences, batch_first=True)
    targets = torch.stack([torch.tensor(t, dtype=torch.float32) for t in targets])
    item_ids = torch.tensor(item_ids, dtype=torch.long)
    return padded_seqs, lengths, targets, item_ids


def collate_test(batch):
    sequences, item_ids = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    padded_seqs = pad_sequence(sequences, batch_first=True)
    item_ids = torch.tensor(item_ids, dtype=torch.long)
    return padded_seqs, lengths, item_ids


class RULLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)

        if self.bidirectional:
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_last = h_n[-1]

        out = self.fc(h_last)
        return out.squeeze(-1)


def create_train_val_dataloaders(
    base_dir: Optional[str] = None,
    batch_size: int = 8,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    failure_df = data_prep.load_failure_data(base_dir)
    degradation_df = data_prep.load_degradation_data(base_dir)

    sequences, targets, item_ids = build_train_sequences(degradation_df, failure_df)
    dataset = RULSequenceDataset(sequences, targets, item_ids)

    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_train,
    )

    return train_loader, val_loader


def create_test_dataloader(
    base_dir: Optional[str] = None,
    batch_size: int = 8,
) -> DataLoader:
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    train_deg_df = data_prep.load_degradation_data(base_dir)
    crack_mean, crack_std = compute_crack_norm_stats(train_deg_df)

    testing_degradation_df = data_prep.load_testing_degradation_data(base_dir)
    sequences, item_ids = build_test_sequences(testing_degradation_df, crack_mean, crack_std)

    test_dataset = RULSequenceDataset(sequences, targets=None, item_ids=item_ids)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_test,
    )

    return test_loader


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    train_loader, val_loader = create_train_val_dataloaders(base_dir=base, batch_size=4)
    test_loader = create_test_dataloader(base_dir=base, batch_size=4)

    model = RULLSTM(input_dim=2, hidden_dim=32, num_layers=1, bidirectional=False)
    print(model)

    for batch in train_loader:
        x, lengths, y, ids = batch
        print("Train batch x shape:", x.shape)
        print("Train batch lengths:", lengths)
        print("Train batch targets:", y.shape)
        preds = model(x, lengths)
        print("Model preds shape:", preds.shape)
        break

    for batch in test_loader:
        x_t, lengths_t, ids_t = batch
        print("Test batch x shape:", x_t.shape)
        print("Test batch lengths:", lengths_t)
        print("Test batch item_ids:", ids_t)
        break
