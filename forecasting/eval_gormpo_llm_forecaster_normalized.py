"""
Supplementary eval for a trained GormpoTokenLLMForecaster checkpoint (see
train_gormpo_llm_forecaster.py / eval_gormpo_llm_forecaster.py): those two scripts
report aggregate MSE/MAE/corr/DTW in raw physical units, which isn't comparable to
every other model's eval in this project (all of which report per-channel MAE/MSE/DTW/
Pearson-corr in normalized z-score space, using /public/gormpo/10min_1hr_all_data.pkl's
mean/std). This reuses train_gormpo_llm_forecaster.py's own inference() (so decoding
is identical to what that script already validated) and just re-expresses the same
predictions in the comparable convention.

Run with: python eval_gormpo_llm_forecaster_normalized.py --llm_path <checkpoint> --tokenizer_path <checkpoint>
"""
import argparse
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from train_gormpo_llm_forecaster import create_gormpo_dataloader, inference  # noqa: E402
from lib.models.metrics import dtw, pearsoncor  # noqa: E402


def compute_channel_bounds(data_root, low_pct=1, high_pct=99):
    """Per-channel (low, high) physical-unit clipping bounds from the train split's raw
    values -- same role/reasoning as eval_gormpo_chronos.py's compute_variable_ylims-
    style bound (and gormpo_world_model.py's compute_variable_ylims): nothing in the
    GORMPO tokenizer's decoder inherently bounds its output, and a bad shape-code
    prediction can decode to a physically impossible value (observed here too: SYSTOLIC/
    PULSAT MSE of 24/36 vs. everything else under 0.5) that dominates an MSE average."""
    x_orig = np.load(f"{data_root}/train_x_original.npy")  # (E, patch_len, 12)
    flat = x_orig.reshape(-1, x_orig.shape[-1])
    low = np.percentile(flat, low_pct, axis=0)
    high = np.percentile(flat, high_pct, axis=0)
    return torch.tensor(low, dtype=torch.float32), torch.tensor(high, dtype=torch.float32)

FEATURE_NAMES = [
    "PumpPressure", "PumpSpeed", "PumpFlow", "LVP", "LVEDP", "SYSTOLIC",
    "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate", "ESE_lv", "Pump Level",
]


def main(args):
    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tokenizer = torch.load(args.tokenizer_path, map_location=device, weights_only=False)
    tokenizer.to(device)
    tokenizer.eval()

    model = torch.load(args.llm_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    with open(args.mcs_pickle_path, "rb") as f:
        data = pickle.load(f)
    mean, std = data["mean"], data["std"]
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)
    columns = [i for i in range(13) if i != 11]  # matches FEATURE_NAMES order project-wide
    m, s = mean[columns].to(device), std[columns].to(device)  # (12,)

    bound_low, bound_high = compute_channel_bounds(args.data_root)
    bound_low, bound_high = bound_low.to(device), bound_high.to(device)

    dataloaders = create_gormpo_dataloader(args.data_root, args.batchsize, args.num_workers)

    all_pred, all_true = [], []
    with torch.no_grad():
        for data_batch in dataloaders["test"]:
            pred_time, labels_time = inference(data_batch, model, tokenizer, device)  # (B, patch_len, 12) physical
            pred_time = torch.clamp(pred_time, bound_low, bound_high)
            all_pred.append(((pred_time - m) / s).cpu())
            all_true.append(((labels_time - m) / s).cpu())

    pred = torch.cat(all_pred, dim=0)  # (N, patch_len, 12)
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
        f"MAE={mae[:11].mean():.4f}  MSE={mse[:11].mean():.4f}  DTW={dtw_dist[:11].mean():.4f}  corr={corr[:11].mean():.4f}"
    )
    print(
        f"\n[secondary] Pump Level (channel 12, predicted but not a forecast target): "
        f"MAE={mae[11]:7.4f}  MSE={mse[11]:7.4f}  DTW={dtw_dist[11]:7.4f}  corr={corr[11]:7.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_id", default=0, type=int)
    parser.add_argument("--tokenizer_path", type=str, default="saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth")
    parser.add_argument("--llm_path", type=str, default="saved_models/gormpo_llm_qwen7b/checkpoints/gormpo_llm_forecaster_checkpoint.pth")
    parser.add_argument("--data_root", type=str, default="data/gormpo_tokens_mcs_scratch_long2000")
    parser.add_argument("--mcs_pickle_path", type=str, default="/public/gormpo/10min_1hr_all_data.pkl")
    parser.add_argument("--batchsize", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    args = parser.parse_args()
    main(args)
