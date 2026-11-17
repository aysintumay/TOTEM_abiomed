"""
Full-test-set eval of "version C" (train_chronos2_xtilde_notimegroup.py): GORMPO's
frozen cross_channel_attn output x_tilde as input, Chronos-2's GroupSelfAttention
stripped -- channels mix exactly once. Same metrics and pairwise inter-channel |corr|
diagnostic as eval_chronos2_xbar_notimegroup.py, for direct comparison with the
x_bar/no-group ablation and the x_tilde/with-group fine-tunes.

Decode: affine_decode with the CONTEXT patch's mu/sigma (gormpo_xtilde.affine_decode)
-- approximate, since x_tilde is x_bar after residual attention + LayerNorm, not a
plain affine transform of it (same caveat as eval_chronos2_xtilde_zeroshot.py).

Pump Level covariate: globally z-scored, (P - mean) / std, past and future -- must
match train_chronos2_xtilde_notimegroup.py (see its docstring for why not x_tilde).

Run with: python eval_chronos2_xtilde_notimegroup.py --model_id saved_models/chronos2_xtilde_notimegroup_globalpl_mcs/finetuned-ckpt
"""
import argparse
import pickle
import sys

import numpy as np
import torch

from chronos import Chronos2Pipeline
from chronos2_time_only_encoder import strip_group_self_attention
from gormpo_xtilde import compute_x_tilde, affine_decode, load_pump_level_stats

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
    return mean[columns][:11], std[columns][:11]


def pairwise_abs_corr(values):
    """values: (n_channels, n_points) numpy array. Returns (avg |corr| over all
    distinct channel pairs, full n_channels x n_channels corr matrix)."""
    corr_mat = np.corrcoef(values)
    n = corr_mat.shape[0]
    off_diag = corr_mat[~np.eye(n, dtype=bool)]
    return np.abs(off_diag).mean(), corr_mat


def main(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"

    tokenizer = torch.load(args.tokenizer_path, map_location=device, weights_only=False)
    tokenizer.to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    print(f"Loading Chronos-2 pipeline: {args.model_id}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=device)
    strip_group_self_attention(pipeline.model)
    print("GroupSelfAttention stripped (eval-time, must match training-time ablation).")
    q_idx = pipeline.quantiles.index(0.5)

    mean, std = load_mean_std(args.mcs_pickle_path)
    pl_mean, pl_std = load_pump_level_stats(args.mcs_pickle_path)
    mean, std = mean.to(device), std.to(device)

    x_half, y_half = load_mcs_episode_halves(args.data_root, "test", args.patch_len)
    x_half, y_half = x_half.to(device), y_half.to(device)
    n = min(args.num_examples, x_half.shape[0])
    rng = np.random.RandomState(args.random_seed)
    indices = rng.choice(x_half.shape[0], size=n, replace=False) if args.random_sample else np.arange(n)
    print(f"Evaluating on {n} test episodes (of {x_half.shape[0]})")

    all_pred, all_true = [], []
    for start in range(0, n, args.batch_size):
        chunk = indices[start : start + args.batch_size]
        x_batch = x_half[chunk]
        y_batch = y_half[chunk]

        with torch.no_grad():
            x_tilde_ctx, mu_ctx, sigma_ctx = compute_x_tilde(tokenizer, x_batch)

        x_tilde_ctx_np = x_tilde_ctx.cpu().numpy()
        pl_ctx_np = ((x_batch[:, 11, :] - pl_mean) / pl_std).cpu().numpy()
        pl_fut_np = ((y_batch[:, 11, :] - pl_mean) / pl_std).cpu().numpy()

        batch_inputs = [
            {
                "target": x_tilde_ctx_np[b, 0:11, :],
                "past_covariates": {"pump_level": pl_ctx_np[b]},
                "future_covariates": {"pump_level": pl_fut_np[b]},
            }
            for b in range(len(chunk))
        ]
        preds = pipeline.predict(batch_inputs, prediction_length=args.patch_len, batch_size=args.batch_size)

        for b, i in enumerate(chunk):
            x_tilde_pred = preds[b][:, q_idx, :].to(device)  # (11, patch_len) in x_tilde space
            pred_phys = affine_decode(x_tilde_pred, mu_ctx[b, 0:11], sigma_ctx[b, 0:11])  # approximate, see docstring
            all_pred.append(pred_phys.cpu())
            all_true.append(y_half[i, 0:11, :].cpu())
        print(f"[{min(start + args.batch_size, n)}/{n}] examples done")

    pred_phys = torch.stack(all_pred, dim=0)  # (n, 11, patch_len)
    true_phys = torch.stack(all_true, dim=0)

    m = mean.cpu().view(1, 11, 1)
    s = std.cpu().view(1, 11, 1)
    pred = ((pred_phys - m) / s).permute(0, 2, 1)  # (n, patch_len, 11)
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

    # --- Pairwise inter-channel |Pearson correlation| diagnostic ---
    # flatten each channel's values across (episodes, timesteps) -> (11, n*patch_len)
    true_flat = true.permute(2, 0, 1).reshape(11, -1).numpy()
    pred_flat = pred.permute(2, 0, 1).reshape(11, -1).numpy()

    avg_true, corr_mat_true = pairwise_abs_corr(true_flat)
    avg_pred, corr_mat_pred = pairwise_abs_corr(pred_flat)

    print("\n=== Pairwise inter-channel |Pearson corr| (redundancy/entanglement diagnostic) ===")
    print(f"  Ground truth avg |corr| across all channel pairs: {avg_true:.4f}")
    print(f"  Prediction   avg |corr| across all channel pairs: {avg_pred:.4f}")

    lvedp_idx, systolic_idx, hr_idx = FEATURE_NAMES.index("LVEDP"), FEATURE_NAMES.index("SYSTOLIC"), FEATURE_NAMES.index("Heart Rate")
    print(f"\n  LVEDP-SYSTOLIC:      ground truth |corr|={abs(corr_mat_true[lvedp_idx, systolic_idx]):.4f}   prediction |corr|={abs(corr_mat_pred[lvedp_idx, systolic_idx]):.4f}")
    print(f"  Heart Rate-SYSTOLIC: ground truth |corr|={abs(corr_mat_true[hr_idx, systolic_idx]):.4f}   prediction |corr|={abs(corr_mat_pred[hr_idx, systolic_idx]):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="saved_models/chronos2_xtilde_notimegroup_globalpl_mcs/finetuned-ckpt")
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
