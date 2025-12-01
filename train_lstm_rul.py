import os
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd

import lstm_data


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for x, lengths, y, _ in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x, lengths)
        loss = loss_fn(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / max(n_samples, 1)


def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for x, lengths, y, _ in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            preds = model(x, lengths)
            loss = loss_fn(preds, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

    return total_loss / max(n_samples, 1)


def predict_on_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    all_item_ids = []
    all_preds = []

    with torch.no_grad():
        for x, lengths, item_ids in loader:
            x = x.to(device)
            lengths = lengths.to(device)

            preds = model(x, lengths)
            preds = preds.cpu()

            all_item_ids.extend(item_ids.tolist())
            all_preds.extend(preds.tolist())

    return all_item_ids, all_preds


def main(
    base_dir: str,
    results_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_ratio: float,
    hidden_dim: int,
    num_layers: int,
    bidirectional: bool,
    device_str: str,
):
    base_path = os.path.abspath(base_dir)
    results_path = Path(results_dir).resolve()
    results_path.mkdir(parents=True, exist_ok=True)

    if "cuda" in device_str and torch.cuda.is_available():
        device = torch.device(device_str)
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    train_loader, val_loader = lstm_data.create_train_val_dataloaders(
        base_dir=base_path,
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=42,
    )

    test_loader = lstm_data.create_test_dataloader(
        base_dir=base_path,
        batch_size=batch_size,
    )

    model = lstm_data.RULLSTM(
        input_dim=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=0.2,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_model_path = results_path / "lstm_rul_best.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_one_epoch(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved to {best_model_path}")

    print(f"Best validation loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    item_ids, preds = predict_on_test(model, test_loader, device)

    df_pred = pd.DataFrame(
        {
            "item_id": item_ids,
            "pred_rul_months": preds,
        }
    ).sort_values("item_id")

    df_pred["item_index"] = df_pred["item_id"].apply(lambda i: f"item_{i}")
    df_pred["label"] = (df_pred["pred_rul_months"] <= 4.0).astype(int)

    submission = df_pred[["item_index", "label", "pred_rul_months"]]

    out_path = results_path / "lstm_rul_predictions.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path} (shape={submission.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM for RUL and predict on test sequences.")
    parser.add_argument("--base-dir", type=str, default=".", help="Repo root (where data_prep.py and data folders live)")
    parser.add_argument("--results-dir", type=str, default="lstm_results", help="Where to save model and predictions")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", help="e.g. 'cuda', 'cuda:0', or 'cpu'")
    args = parser.parse_args()

    main(
        base_dir=args.base_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        device_str=args.device,
    )
