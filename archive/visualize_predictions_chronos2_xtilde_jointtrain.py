"""
python visualize_predictions_chronos2_xtilde_jointtrain.py
Plots ground truth vs the x_tilde + Chronos-2 (jointly-trained cross_channel_attn)
model's predictions for a few test samples -- the highest-correlation model in this
project's whole comparison (full-test-set corr=0.3039 vs. the next-best 0.0828), run
here to see qualitatively what that correlation gap actually looks like, and to check
the MAE/MSE-vs-corr story (predicted SHAPE close, absolute SCALE off -- see
eval_chronos2_xtilde_zeroshot.py's last run's analysis) visually rather than only in
aggregate numbers.

Mirrors every other visualize_predictions_*.py script's layout/style exactly (same
FEATURE_NAMES, same fixed y-limits via gormpo_world_model.compute_variable_ylims, same
time axis, same sample indices) for direct panel-for-panel comparability.

Uncertainty band: Chronos-2 is a genuine quantile-regression model (not
sample-based), so the shaded band here is the REAL predicted q10-q90 interval from its
own quantile grid -- a more principled uncertainty estimate than the sample-mean±std
bands every other model in this project uses (those approximate uncertainty via
repeated stochastic draws; this one is the model's own calibrated-in-training quantile
output directly).

Decode: same direct affine restoration as eval_chronos2_xtilde_zeroshot.py
(gormpo_xtilde.affine_decode), using the CONTEXT patch's own mu/sigma as the
placeholder for the (not yet properly handled -- see that eval's analysis) future
patch's true scale/location.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from chronos import Chronos2Pipeline
from gormpo_xtilde import compute_x_tilde, affine_decode

sys.path.insert(0, "forecasting")
from train_tokenizer import load_mcs_episode_halves  # noqa: E402

from gormpo_world_model import FEATURE_NAMES, compute_variable_ylims  # noqa: E402

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
DATA_ROOT = "forecasting/data_raw/abiomed"
TOKENIZER_PATH = "saved_models/chronos2_xtilde_jointtrain_mcs/tokenizer_with_jointtrained_cross_channel_attn.pth"
MODEL_ID = "saved_models/chronos2_xtilde_jointtrain_mcs/finetuned-ckpt"
DEVICE = "cuda:1"
SAMPLE_IDXS = [0, 1, 2, 3]  # same test samples as every other visualize_predictions* script
OUT_DIR = "figures"

os.makedirs(OUT_DIR, exist_ok=True)

YLIMS = compute_variable_ylims(DATA_PATH)

tokenizer = torch.load(TOKENIZER_PATH, map_location=DEVICE, weights_only=False)
tokenizer.to(DEVICE)
tokenizer.eval()
for p in tokenizer.parameters():
    p.requires_grad_(False)

print(f"Loading Chronos-2 pipeline: {MODEL_ID}")
pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=DEVICE)
q10_idx, q50_idx, q90_idx = pipeline.quantiles.index(0.1), pipeline.quantiles.index(0.5), pipeline.quantiles.index(0.9)

x_half, y_half = load_mcs_episode_halves(DATA_ROOT, "test", 6)
x_half, y_half = x_half.to(DEVICE), y_half.to(DEVICE)

t_ctx = np.arange(-5, 1) * 10    # -50, -40, ..., 0
t_pred = np.arange(1, 7) * 10    # +10, +20, ..., +60

for idx in SAMPLE_IDXS:
    x_ep = x_half[idx : idx + 1]  # (1, 12, patch_len)
    y_ep = y_half[idx : idx + 1]

    x_tilde_ctx, mu_ctx, sigma_ctx = compute_x_tilde(tokenizer, x_ep)
    x_tilde_fut, _, _ = compute_x_tilde(tokenizer, y_ep)

    batch_inputs = [{
        "target": x_tilde_ctx[0, 0:11, :].cpu().numpy(),
        "past_covariates": {"pump_level": x_tilde_ctx[0, 11, :].cpu().numpy()},
        "future_covariates": {"pump_level": x_tilde_fut[0, 11, :].cpu().numpy()},
    }]
    with torch.no_grad():
        preds = pipeline.predict(batch_inputs, prediction_length=6, batch_size=1)
    pred_x_tilde = preds[0].to(DEVICE)  # (11, n_quantiles, patch_len)

    q10 = affine_decode(pred_x_tilde[:, q10_idx, :], mu_ctx[0, 0:11], sigma_ctx[0, 0:11]).cpu().numpy()
    q50 = affine_decode(pred_x_tilde[:, q50_idx, :], mu_ctx[0, 0:11], sigma_ctx[0, 0:11]).cpu().numpy()
    q90 = affine_decode(pred_x_tilde[:, q90_idx, :], mu_ctx[0, 0:11], sigma_ctx[0, 0:11]).cpu().numpy()

    ctx_phys = x_ep[0].cpu().numpy()      # (12, patch_len)
    y_phys = y_ep[0].cpu().numpy()        # (12, patch_len)
    pl_int = int(round(float(y_phys[11].mean())))

    n_features = 12
    ncols = 4
    nrows = (n_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    fig.suptitle(
        f"Test sample {idx}  |  realized P-level = {pl_int}  |  x_tilde+Chronos-2 jointtrain ({MODEL_ID})",
        fontsize=12,
    )

    for j in range(11):
        ax = axes[j]
        ax.plot(t_ctx, ctx_phys[j], color="steelblue", marker="o", markersize=3, label="context")
        ax.plot(t_pred, y_phys[j], color="green", marker="o", markersize=3, linestyle="--", label="ground truth")
        ax.plot(t_pred, q50[j], color="tomato", marker="o", markersize=3, label="prediction (median)")
        ax.fill_between(t_pred, q10[j], q90[j], color="tomato", alpha=0.2, label="q10-q90")
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylim(*YLIMS[FEATURE_NAMES[j]])
        ax.set_title(FEATURE_NAMES[j], fontsize=9)
        ax.set_xlabel("time (min)", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=6)

    # Pump Level panel: known-future covariate here, not predicted -- context + the
    # realized action held forward, same convention as the other native-Chronos-2
    # visualize scripts.
    ax = axes[11]
    ax.plot(t_ctx, ctx_phys[11], color="steelblue", marker="o", markersize=3, label="context")
    ax.plot(t_pred, [pl_int] * len(t_pred), color="black", marker="s", markersize=3, linestyle="-", label="action (known)")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(*YLIMS[FEATURE_NAMES[11]])
    ax.set_title(f"{FEATURE_NAMES[11]} (action, not forecast)", fontsize=9)
    ax.set_xlabel("time (min)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"chronos2_xtilde_jointtrain_sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

print("Done.")
