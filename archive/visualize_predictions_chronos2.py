"""
python visualize_predictions_chronos2.py
Plots ground truth vs native Chronos-2's predictions (amazon/chronos-2, or a
fine-tuned checkpoint -- defaults to the fine-tuned saved_models/chronos2_mcs
checkpoint, MAE=0.1821, the best full-test-set result in the whole project
comparison) for the same few test samples every other visualize_predictions_*.py
script uses, for direct panel-for-panel comparability.

Native Chronos-2, no GORMPO tokenizer involved (contrast every other
visualize_predictions_chronos2_*.py variant): the 11 physiological channels are
forecast jointly in one call (sharing information via GroupSelfAttention, fully
active here -- no ablation), raw physical units throughout (Chronos-2's own
InstanceNorm handles scale internally), Pump Level as a known past+future
covariate rather than a forecast target.

Uncertainty band: Chronos-2's own genuine quantile regression (q05-q95, 90%), not a
sample-based approximation.

Run with: python visualize_predictions_chronos2.py
          python visualize_predictions_chronos2.py --model_id amazon/chronos-2   # zero-shot instead
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from chronos import Chronos2Pipeline

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

    print(f"Loading Chronos-2 pipeline: {args.model_id}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=args.device)
    q05_idx, q50_idx, q95_idx = pipeline.quantiles.index(0.05), pipeline.quantiles.index(0.5), pipeline.quantiles.index(0.95)

    x_half, y_half = load_mcs_episode_halves(DATA_ROOT, "test", 6)
    x_half, y_half = x_half.numpy(), y_half.numpy()  # (E, 12, patch_len) physical

    t_ctx = np.arange(-5, 1) * 10    # -50, -40, ..., 0
    t_pred = np.arange(1, 7) * 10    # +10, +20, ..., +60

    for idx in args.sample_idxs:
        x_ep = x_half[idx]  # (12, patch_len)
        y_ep = y_half[idx]

        batch_inputs = [{
            "target": x_ep[0:11, :],
            "past_covariates": {"pump_level": x_ep[11, :]},
            "future_covariates": {"pump_level": y_ep[11, :]},
        }]
        preds = pipeline.predict(batch_inputs, prediction_length=6, batch_size=1)
        pred = preds[0].cpu().numpy()  # (11, n_quantiles, patch_len) physical

        q05 = pred[:, q05_idx, :]
        q50 = pred[:, q50_idx, :]
        q95 = pred[:, q95_idx, :]

        pl_seq = " ".join(f"{v:g}" for v in np.round(y_ep[11], 1))

        n_features = 12
        ncols = 4
        nrows = (n_features + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
        axes = axes.flatten()
        fig.suptitle(
            f"Test sample {idx}  |  realized P-level = [{pl_seq}]  |  native Chronos-2 ({args.model_id})",
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
        path = os.path.join(OUT_DIR, f"chronos2_native_sample_{idx:03d}.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"Saved {path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="saved_models/chronos2_mcs/finetuned-ckpt")
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--sample_idxs", type=int, nargs="+", default=SAMPLE_IDXS)
    args = parser.parse_args()
    main(args)
