"""
Merge shard .npz files from eval_llm_world_model.py --output_npz (one per parallel
shard process) into one final per-channel MAE/MSE/DTW/Pearson-corr report, computed
once over the concatenated full set -- not a weighted average of per-shard metrics,
which would only be exact for MAE/MSE (simple means) and not for DTW/corr as computed
here (pearsoncor/dtw with reduction="mean" is itself a mean over per-sample values,
so concatenating raw predictions first and computing once is the correct way, not an
approximation).

Run with: python merge_shards.py "logs/qwen7b_zeroshot_shard*.npz"
"""
import argparse
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, "forecasting")
from lib.models.metrics import dtw, pearsoncor  # noqa: E402

FEATURE_NAMES = [
    "PumpPressure", "PumpSpeed", "PumpFlow", "LVP", "LVEDP", "SYSTOLIC",
    "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate", "ESE_lv",
]


def main(args):
    paths = sorted(glob.glob(args.glob_pattern))
    if not paths:
        raise FileNotFoundError(f"no files matched {args.glob_pattern!r}")
    print(f"Merging {len(paths)} shard files:")
    for p in paths:
        print(f"  {p}")

    all_pred, all_true, total_n, total_fallback = [], [], 0, 0
    for p in paths:
        d = np.load(p)
        all_pred.append(d["pred"])
        all_true.append(d["true"])
        total_n += int(d["n"])
        total_fallback += int(d["n_fallback"])

    pred = torch.tensor(np.concatenate(all_pred, axis=0), dtype=torch.float32)  # (N, T, 11)
    true = torch.tensor(np.concatenate(all_true, axis=0), dtype=torch.float32)
    print(f"\nCombined: {pred.shape[0]} episodes (expected {total_n})")

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
        f"\nOverall (mean across all 11 variables): "
        f"MAE={mae.mean():.4f}  MSE={mse.mean():.4f}  DTW={dtw_dist.mean():.4f}  corr={corr.mean():.4f}"
    )
    print(f"Fallback rate: {total_fallback}/{pred.shape[0]} examples had all samples unparseable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("glob_pattern", type=str)
    args = parser.parse_args()
    main(args)
