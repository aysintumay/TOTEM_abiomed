"""
Zero-shot sanity check: feed GORMPO's cross-channel-attended, frozen x_tilde
representation (gormpo_xtilde.py) into PRETRAINED, UNTOUCHED Chronos-2 -- no
fine-tuning yet -- instead of discrete VQ-quantized shape codes (contrast
train_gormpo_chronos2.py) or raw physical values (contrast
eval_chronos2_world_model.py). Cheap, fast check of whether this representation is
even in a range/shape Chronos-2 can make sense of zero-shot, before investing in a
fine-tuning run.

Task framing is otherwise identical to eval_chronos2_world_model.py: the 11
physiological channels are the joint multivariate target; Pump Level is a known-future
covariate -- except now BOTH are expressed in x_tilde space (computed jointly across
all 12 channels at once, since cross_channel_attn needs every channel present to
compute attention), not raw physical units.

Point forecast = the median (q=0.5) of Chronos-2's predicted quantile grid, same as
eval_chronos2_world_model.py. Decoded back to physical units via the direct affine
restoration (gormpo_xtilde.affine_decode), using the CONTEXT patch's own mu/sigma as
the placeholder scale/location (the "incorporate mu/sigma properly later" deferral).

Run with: python eval_chronos2_xtilde_zeroshot.py --model_id amazon/chronos-2
"""
import argparse
import pickle
import sys

import numpy as np
import torch

from chronos import Chronos2Pipeline
from gormpo_xtilde import compute_x_tilde, affine_decode

sys.path.insert(0, "forecasting")
from train_tokenizer import load_mcs_episode_halves  # noqa: E402
from lib.models.metrics import dtw, pearsoncor  # noqa: E402

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
    columns = [i for i in range(13) if i != 11]
    return mean[columns][:11], std[columns][:11]  # (11,) each, physiological channels only, for final normalized metrics


def main(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"

    tokenizer = torch.load(args.tokenizer_path, map_location=device, weights_only=False)
    tokenizer.to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)  # frozen -- see gormpo_xtilde.py's module docstring for why

    print(f"Loading Chronos-2 pipeline: {args.model_id}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=device)
    q_idx = pipeline.quantiles.index(0.5)

    mean, std = load_mean_std(args.mcs_pickle_path)
    mean, std = mean.to(device), std.to(device)

    x_half, y_half = load_mcs_episode_halves(args.data_root, "test", args.patch_len)
    x_half, y_half = x_half.to(device), y_half.to(device)  # (E, 12, patch_len) channel-major physical units
    n = min(args.num_examples, x_half.shape[0])
    rng = np.random.RandomState(args.random_seed)
    indices = rng.choice(x_half.shape[0], size=n, replace=False) if args.random_sample else np.arange(n)
    print(f"Evaluating on {n} test episodes (of {x_half.shape[0]})")

    all_pred, all_true = [], []
    for start in range(0, n, args.batch_size):
        chunk = indices[start : start + args.batch_size]
        x_batch = x_half[chunk]  # (b, 12, patch_len)
        y_batch = y_half[chunk]

        # cross_channel_attn needs every channel present at once -- compute x_tilde
        # jointly for all 12 channels, for both context and future patches.
        x_tilde_ctx, mu_ctx, sigma_ctx = compute_x_tilde(tokenizer, x_batch)   # (b,12,k), (b,12), (b,12)
        x_tilde_fut, _, _ = compute_x_tilde(tokenizer, y_batch)                # future patch's own x_tilde (Pump Level's known-future covariate)

        x_tilde_ctx_np = x_tilde_ctx.cpu().numpy()
        x_tilde_fut_np = x_tilde_fut.cpu().numpy()

        batch_inputs = [
            {
                "target": x_tilde_ctx_np[b, 0:11, :],
                "past_covariates": {"pump_level": x_tilde_ctx_np[b, 11, :]},
                "future_covariates": {"pump_level": x_tilde_fut_np[b, 11, :]},
            }
            for b in range(len(chunk))
        ]
        preds = pipeline.predict(batch_inputs, prediction_length=args.patch_len, batch_size=args.batch_size)

        for b, i in enumerate(chunk):
            x_tilde_pred = preds[b][:, q_idx, :].to(device)  # (11, patch_len) in x_tilde space
            # affine-decode using the CONTEXT patch's own mu/sigma for the 11 target
            # channels (placeholder for "incorporate the future patch's true mu/sigma
            # properly later" -- see gormpo_xtilde.py).
            pred_phys = affine_decode(x_tilde_pred, mu_ctx[b, 0:11], sigma_ctx[b, 0:11])  # (11, patch_len)
            all_pred.append(pred_phys.cpu())
            all_true.append(y_half[i, 0:11, :].cpu())
        print(f"[{min(start + args.batch_size, n)}/{n}] examples done")

    pred_phys = torch.stack(all_pred, dim=0)  # (n, 11, patch_len)
    true_phys = torch.stack(all_true, dim=0)

    m = mean.cpu().view(1, 11, 1)
    s = std.cpu().view(1, 11, 1)
    pred = ((pred_phys - m) / s).permute(0, 2, 1)  # (n, patch_len, 11) -- time-major, matching other eval scripts
    true = ((true_phys - m) / s).permute(0, 2, 1)

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
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2")
    parser.add_argument("--tokenizer_path", type=str,
                         default="forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth")
    parser.add_argument("--data_root", type=str, default="forecasting/data_raw/abiomed")
    parser.add_argument("--mcs_pickle_path", type=str, default="/public/gormpo/10min_1hr_all_data.pkl")
    parser.add_argument("--patch_len", type=int, default=6)
    parser.add_argument("--num_examples", type=int, default=3876)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
