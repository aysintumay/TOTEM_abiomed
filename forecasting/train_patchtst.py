"""
Train the PatchTST baseline (lib/models/patchtst.py) -- channel-independent, no
tokenization -- for comparison against the GORMPO tokenizer pipeline.

Reuses data_provider.data_factory.data_provider directly (same Dataset_Neuro/
Dataset_ETT_hour classes the rest of the repo uses), so metrics are computed in the
same globally-standardized space Dataset_Neuro/Dataset_ETT_hour already produce
(scale=True), directly comparable to the existing TOTEM+LLM baseline's reported
mse/mae without any extra renormalization step.

Run with: python train_patchtst.py --data_type abiomed
"""
import argparse
import os

import torch
import torch.nn.functional as F

from data_provider.data_factory import data_provider
from lib.models.metrics import dtw, pearsoncor
from lib.models.patchtst import PatchTST
from lib.utils.checkpoint import EarlyStopping
from lib.utils.env import seed_all_rng

DATASET_CONFIGS = {
    "abiomed": dict(data="abiomed", root_path="data_raw/abiomed", data_path="", enc_in=12, batchsize=128, freq="t"),
    "ETTh1": dict(data="ETTh1", root_path="data_raw/ETTh1", data_path="ETTh1.csv", enc_in=7, batchsize=32, freq="h"),
}


def build_dataloaders(data_type, seq_len, pred_len, num_workers):
    cfg = DATASET_CONFIGS[data_type]
    args = argparse.Namespace(
        data=cfg["data"], root_path=cfg["root_path"], data_path=cfg["data_path"],
        features="M", target="OT", freq=cfg["freq"], embed="timeF",
        seq_len=seq_len, label_len=0, pred_len=pred_len,
        batch_size=cfg["batchsize"], num_workers=num_workers,
    )
    loaders = {}
    for split in ("train", "val", "test"):
        _, loaders[split] = data_provider(args, split)
    return loaders, cfg["enc_in"], cfg["batchsize"]


def run_eval(loader, model, device, tag):
    model.eval()
    running_mse, running_mae, running_cor, running_dtw, total_num, total_num_c = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
            _, pred = model.shared_eval((batch_x, batch_y), None, "val")
            running_mse += F.mse_loss(pred, batch_y, reduction="sum").item()
            running_mae += (pred - batch_y).abs().sum().item()
            running_cor += pearsoncor(pred, batch_y, reduction="sum").item()
            running_dtw += dtw(pred, batch_y, reduction="sum").item()
            total_num += batch_y.numel()
            total_num_c += batch_y.shape[0] * batch_y.shape[2]
    mse, mae = running_mse / total_num, running_mae / total_num
    cor, dt = running_cor / total_num_c, running_dtw / total_num_c
    print(f"| [{tag}] mse {mse:.4f} mae {mae:.4f} corr {cor:.4f} dtw {dt:.4f}")
    return mse, mae


def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    seed_all_rng(args.seed)

    loaders, n_vars, default_batchsize = build_dataloaders(
        args.data_type, args.seq_len, args.pred_len, args.num_workers
    )
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]
    print(f"train batches: {len(train_loader)}  val: {len(val_loader)}  test: {len(test_loader)}  n_vars={n_vars}")

    model = PatchTST(
        seq_len=args.seq_len, pred_len=args.pred_len, n_vars=n_vars,
        patch_len=args.patch_len, stride=args.stride, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
    )
    model.to(device)
    print("Total trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = model.configure_optimizers(lr=args.lr)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    early_stopping = EarlyStopping(patience=args.patience, path=args.checkpoint_path)

    for epoch in range(args.epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch_x, batch_y, _, _ in train_loader:
            batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
            loss, _ = model.shared_eval((batch_x, batch_y), optimizer, "train")
            running_loss += loss.item()
            n_batches += 1
        print(f"| epoch {epoch:3d} | train mse {running_loss / n_batches:.4f}")

        val_mse, val_mae = run_eval(val_loader, model, device, "Val")
        counter = early_stopping(val_mse, val_mae, {"patchtst": model})
        run_eval(test_loader, model, device, "Test")

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch} (counter {counter})")
            break

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="abiomed", choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--pred_len", type=int, default=6)
    parser.add_argument("--patch_len", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--checkpoint_path", type=str, default="saved_models/patchtst/checkpoints/")
    args = parser.parse_args()

    train(args)
