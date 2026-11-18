"""
python visualize_predictions_chronos2_xtilde_notimegroup.py --sample_idxs 1240 3704 321
Plots ground truth vs "version C" (train_chronos2_xtilde_notimegroup.py): GORMPO's
frozen cross_channel_attn output x_tilde as input, Chronos-2's GroupSelfAttention
stripped -- channels mix exactly once, before Chronos-2. Same layout as
visualize_predictions_chronos2.py (90% q05-q95 band, per-step Pump Level action,
y-limits widened to show every plotted value) for panel-for-panel comparison with
native Chronos-2.

Decode: gormpo_xtilde.affine_decode with the CONTEXT patch's mu/sigma (same
approximation as eval_chronos2_xtilde_notimegroup.py). Pump Level covariate globally
z-scored, matching train_chronos2_xtilde_notimegroup.py.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from chronos import Chronos2Pipeline
from chronos2_time_only_encoder import strip_group_self_attention
from gormpo_xtilde import compute_x_tilde, affine_decode, load_pump_level_stats

sys.path.insert(0, "forecasting")
from train_tokenizer import load_mcs_episode_halves  # noqa: E402

from gormpo_world_model import FEATURE_NAMES, compute_variable_ylims  # noqa: E402

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
DATA_ROOT = "forecasting/data_raw/abiomed"
SAMPLE_IDXS = [0, 1, 2, 3]  # same test samples as every other visualize_predictions* script
OUT_DIR = "figures"


def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    YLIMS = compute_variable_ylims(DATA_PATH)
    pl_mean, pl_std = load_pump_level_stats(DATA_PATH)

    tokenizer = torch.load(args.tokenizer_path, map_location=args.device, weights_only=False)
    tokenizer.to(args.device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    print(f"Loading Chronos-2 pipeline: {args.model_id}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=args.device)
    strip_group_self_attention(pipeline.model)  # must match the training-time ablation
    q05_idx, q50_idx, q95_idx = pipeline.quantiles.index(0.05), pipeline.quantiles.index(0.5), pipeline.quantiles.index(0.95)

    x_half, y_half = load_mcs_episode_halves(DATA_ROOT, "test", 6)
    x_half, y_half = x_half.to(args.device), y_half.to(args.device)

    t_ctx = np.arange(-5, 1) * 10    # -50, -40, ..., 0
    t_pred = np.arange(1, 7) * 10    # +10, +20, ..., +60

    for idx in args.sample_idxs:
        x_t = x_half[idx : idx + 1]  # (1, 12, patch_len)
        y_t = y_half[idx : idx + 1]

        with torch.no_grad():
            x_tilde_ctx, mu_ctx, sigma_ctx = compute_x_tilde(tokenizer, x_t)

        batch_inputs = [{
            "target": x_tilde_ctx[0, 0:11, :].cpu().numpy(),
            "past_covariates": {"pump_level": ((x_t[0, 11, :] - pl_mean) / pl_std).cpu().numpy()},
            "future_covariates": {"pump_level": ((y_t[0, 11, :] - pl_mean) / pl_std).cpu().numpy()},
        }]
        preds = pipeline.predict(batch_inputs, prediction_length=6, batch_size=1)
        pred_x_tilde = preds[0].to(args.device)  # (11, n_quantiles, patch_len) in x_tilde space

        q05 = affine_decode(pred_x_tilde[:, q05_idx, :], mu_ctx[0, 0:11], sigma_ctx[0, 0:11]).cpu().numpy()
        q50 = affine_decode(pred_x_tilde[:, q50_idx, :], mu_ctx[0, 0:11], sigma_ctx[0, 0:11]).cpu().numpy()
        q95 = affine_decode(pred_x_tilde[:, q95_idx, :], mu_ctx[0, 0:11], sigma_ctx[0, 0:11]).cpu().numpy()

        x_ep = x_t[0].cpu().numpy()  # (12, patch_len) physical
        y_ep = y_t[0].cpu().numpy()
        pl_seq = " ".join(f"{v:g}" for v in np.round(y_ep[11], 1))

        n_features = 12
        ncols = 4
        nrows = (n_features + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
        axes = axes.flatten()
        fig.suptitle(
            f"Test sample {idx}  |  realized P-level = [{pl_seq}]  |  x_tilde + Chronos-2, no GroupSelfAttention ({args.model_id})",
            fontsize=12,
        )

        for j in range(11):
            ax = axes[j]
            ax.plot(t_ctx, x_ep[j], color="steelblue", marker="o", markersize=3, label="context")
            ax.plot(t_pred, y_ep[j], color="green", marker="o", markersize=3, linestyle="--", label="ground truth")
            ax.plot(t_pred, q50[j], color="tomato", marker="o", markersize=3, label="prediction (median)")
            ax.fill_between(t_pred, q05[j], q95[j], color="tomato", alpha=0.2, label="90% interval (q05-q95)")
            ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
            # widen the fixed train-percentile limits so every plotted value is visible
            lo = min(YLIMS[FEATURE_NAMES[j]][0], x_ep[j].min(), y_ep[j].min(), q05[j].min())
            hi = max(YLIMS[FEATURE_NAMES[j]][1], x_ep[j].max(), y_ep[j].max(), q95[j].max())
            pad = 0.05 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(FEATURE_NAMES[j], fontsize=9)
            ax.set_xlabel("time (min)", fontsize=8)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.legend(fontsize=6)

        ax = axes[11]
        ax.plot(t_ctx, x_ep[11], color="steelblue", marker="o", markersize=3, label="context")
        ax.plot(t_pred, y_ep[11], color="black", marker="s", markersize=3, linestyle="-", label="action (known)")
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylim(*YLIMS[FEATURE_NAMES[11]])
        ax.set_title(f"{FEATURE_NAMES[11]} (action, not forecast)", fontsize=9)
        ax.set_xlabel("time (min)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"chronos2_xtilde_notimegroup_globalpl_sample_{idx:03d}.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"Saved {path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="saved_models/chronos2_xtilde_notimegroup_globalpl_mcs/finetuned-ckpt")
    parser.add_argument("--tokenizer_path", type=str,
                        default="forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth")
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--sample_idxs", type=int, nargs="+", default=SAMPLE_IDXS)
    args = parser.parse_args()
    main(args)
