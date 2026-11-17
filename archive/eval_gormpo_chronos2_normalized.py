"""
Supplementary eval for a trained GormpoChronos2Model checkpoint (see
train_gormpo_chronos2.py): that script's own per-epoch run_eval reports aggregate
MSE/MAE/corr/DTW in raw physical units, not comparable across channels of wildly
different physical scale -- this re-expresses the same predictions in the normalized
z-score convention every other model in this project's comparison uses (mirrors
forecasting/eval_gormpo_llm_forecaster_normalized.py's role for the Qwen2.5-7B+GORMPO
fine-tune).

Run with: python eval_gormpo_chronos2_normalized.py --checkpoint_path <checkpoint>
"""
import argparse
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, "forecasting")
from train_gormpo_llm_forecaster import create_gormpo_dataloader  # noqa: E402
from lib.models.metrics import dtw, pearsoncor  # noqa: E402

from train_gormpo_chronos2 import compute_channel_bounds, flatten_batch, forward_and_loss, decode_predictions  # noqa: E402

FEATURE_NAMES = [
    "PumpPressure", "PumpSpeed", "PumpFlow", "LVP", "LVEDP", "SYSTOLIC",
    "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate", "ESE_lv",
]


def load_mean_std(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    mean, std = data["mean"], data["std"]
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)
    columns = [i for i in range(13) if i != 11]  # matches FEATURE_NAMES order project-wide
    return mean[columns][:11], std[columns][:11]  # (11,) each, physiological channels only


def main(args):
    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tokenizer = torch.load(args.tokenizer_path, map_location=device, weights_only=False)
    tokenizer.to(device)
    tokenizer.eval()
    num_channels = tokenizer.num_channels

    model = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    m, s = load_mean_std(args.mcs_pickle_path)
    m, s = m.to(device), s.to(device)

    bound_low, bound_high = compute_channel_bounds(args.data_root)
    bound_low, bound_high = bound_low.to(device), bound_high.to(device)

    dataloaders = create_gormpo_dataloader(args.data_root, args.batchsize, args.num_workers)

    all_pred, all_true = [], []
    with torch.no_grad():
        for data in dataloaders["test"]:
            batch = flatten_batch(data, device, num_channels)
            _, _, _, _, shape_logits, scalar_logits = forward_and_loss(model, batch, scalar_loss_weight=1.0)
            pred_phys = decode_predictions(shape_logits, scalar_logits, batch, tokenizer, bound_low, bound_high)
            pred_time = pred_phys.transpose(1, 2)            # (B, patch_len, 11) physical
            true_time = batch["y_orig"][:, :, :11]            # (B, patch_len, 11) physical

            all_pred.append(((pred_time - m) / s).cpu())
            all_true.append(((true_time - m) / s).cpu())

    pred = torch.cat(all_pred, dim=0)  # (N, patch_len, 11)
    true = torch.cat(all_true, dim=0)
    print(f"Evaluated on {pred.shape[0]} test episodes (full test set)")

    mae = (pred - true).abs().mean(dim=(0, 1))
    mse = (pred - true).pow(2).mean(dim=(0, 1))
    corr = torch.tensor([
        pearsoncor(pred[:, :, j:j + 1], true[:, :, j:j + 1], reduction="mean").item()
        for j in range(pred.shape[-1])
    ])
    dtw_dist = torch.tensor([
        dtw(pred[:, :, j:j + 1], true[:, :, j:j + 1], reduction="mean").item()
        for j in range(pred.shape[-1])
    ])

    print("\n=== Per-variable MAE / MSE / DTW / Pearson corr (normalized/z-score units, averaged over all samples) ===")
    for j in range(11):
        print(f"  {FEATURE_NAMES[j]:<14} MAE={mae[j]:7.4f}  MSE={mse[j]:7.4f}  DTW={dtw_dist[j]:7.4f}  corr={corr[j]:7.4f}")
    print(
        f"\nOverall (mean across 11 forecast variables): "
        f"MAE={mae.mean():.4f}  MSE={mse.mean():.4f}  DTW={dtw_dist.mean():.4f}  corr={corr.mean():.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_id", default=0, type=int)
    parser.add_argument("--tokenizer_path", type=str,
                         default="forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth")
    parser.add_argument("--checkpoint_path", type=str,
                         default="saved_models/gormpo_chronos2_mcs/checkpoints/gormpo_chronos2_checkpoint.pth")
    parser.add_argument("--data_root", type=str, default="forecasting/data/gormpo_tokens_mcs_scratch_long2000")
    parser.add_argument("--mcs_pickle_path", type=str, default="/public/gormpo/10min_1hr_all_data.pkl")
    parser.add_argument("--batchsize", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    args = parser.parse_args()
    main(args)
