"""
Evaluate Chronos2Pipeline (amazon/chronos-2, or a fine-tuned checkpoint of it) on the
real MCS test split, native multivariate + covariate-conditioned forecasting, no
GORMPO tokenizer involved -- this is Chronos-2 used exactly as designed, not a
tokenizer swap (see mcs.md for why: Chronos-2 has no discrete tokenizer to swap in the
first place, unlike original Chronos's ChronosTokenizer interface).

Task framing: the 11 physiological channels are the joint multivariate target
(forecast together in one call, sharing information via GroupSelfAttention); Pump
Level (the action) is NOT a forecast target -- its past values go in as
past_covariates and its FUTURE values (known, since it's the action chosen for that
window) go in as future_covariates. This maps the actual MCS task structure (predict
state given a chosen action) onto Chronos-2's native covariate-conditioning mechanism,
rather than tokenizing Pump Level as just another channel the way the GORMPO-tokenizer
models do.

Point forecast = the median (q=0.5) of Chronos-2's predicted quantile grid -- the
natural point estimate for a direct quantile-regression model (Chronos-2 doesn't
sample trajectories the way original Chronos/GORMPO-LLM decoding does, so there's no
"mean of samples" to use instead).

Metrics (MAE, MSE, DTW, Pearson correlation) are computed in normalized (z-score)
space using /public/gormpo/10min_1hr_all_data.pkl's mean/std, same convention as every
other eval_*_world_model.py script in this project.

Run with: python eval_chronos2_world_model.py --model_id amazon/chronos-2
"""
import argparse
import pickle
import sys

import numpy as np
import torch

from chronos import Chronos2Pipeline

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
    # pickle has 13 raw columns; index 11 is an unused NaN placeholder, index 12 is the
    # real Pump Level -- this selection (project-wide convention) yields the 12 *used*
    # channels in FEATURE_NAMES order with Pump Level last; drop it here since Pump
    # Level isn't a forecast target for Chronos-2's covariate-conditioned setup.
    columns = [i for i in range(13) if i != 11]
    return mean[columns][:11], std[columns][:11]  # (11,) each, physiological channels only


def main(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    print(f"Loading Chronos-2 pipeline: {args.model_id}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=device)
    q_idx = pipeline.quantiles.index(0.5)

    mean, std = load_mean_std(args.mcs_pickle_path)  # (11,) each

    x_half, y_half = load_mcs_episode_halves(args.data_root, "test", args.patch_len)
    x_half, y_half = x_half.numpy(), y_half.numpy()  # (E, 12, patch_len) channel-major physical units
    n = min(args.num_examples, x_half.shape[0])
    rng = np.random.RandomState(args.random_seed)
    indices = rng.choice(x_half.shape[0], size=n, replace=False) if args.random_sample else np.arange(n)
    print(f"Evaluating on {n} test episodes (of {x_half.shape[0]})")

    all_pred, all_true = [], []
    for start in range(0, n, args.batch_size):
        chunk = indices[start : start + args.batch_size]
        batch_inputs = [
            {
                "target": x_half[i, 0:11, :],
                "past_covariates": {"pump_level": x_half[i, 11, :]},
                "future_covariates": {"pump_level": y_half[i, 11, :]},
            }
            for i in chunk
        ]
        preds = pipeline.predict(batch_inputs, prediction_length=args.patch_len, batch_size=args.batch_size)
        for pred, i in zip(preds, chunk):
            median_pred = pred[:, q_idx, :].cpu().numpy()  # (11, patch_len) physical
            all_pred.append(median_pred)
            all_true.append(y_half[i, 0:11, :])
        print(f"[{min(start + args.batch_size, n)}/{n}] examples done")

    pred_phys = torch.tensor(np.stack(all_pred), dtype=torch.float32)  # (n, 11, patch_len)
    true_phys = torch.tensor(np.stack(all_true), dtype=torch.float32)

    m = mean.view(1, 11, 1)
    s = std.view(1, 11, 1)
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
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2",
                         help="HF hub id or local fine-tuned checkpoint dir")
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
