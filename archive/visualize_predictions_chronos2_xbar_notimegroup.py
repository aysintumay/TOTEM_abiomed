"""
python visualize_predictions_chronos2_xbar_notimegroup.py
Plots ground truth vs the x_bar + Chronos-2 (both cross-channel-mixing mechanisms
removed: no cross_channel_attn, no GroupSelfAttention) model's predictions for the
same few test samples every other visualize_predictions_*.py script uses, for direct
panel-for-panel comparability -- this is the ablation run in response to the
LVEDP-SYSTOLIC entanglement found in the x_tilde jointtrain model (see
chronos2_time_only_encoder.py / train_chronos2_xbar_notimegroup.py docstrings,
eval_chronos2_xbar_notimegroup.py's pairwise-correlation diagnostic for the
full-test-set confirmation that this ablation fixes it).

Decode is EXACT here, not approximate (contrast the x_tilde visualize script):
x_bar = (x - mu) / sigma is a plain affine transform, so
pred_phys = sigma_ctx * x_bar_pred + mu_ctx exactly inverts patch_and_scale.

Uncertainty band: Chronos-2's own genuine quantile regression (q10-q90), not a
sample-based approximation -- same as the x_tilde jointtrain visualize script.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from chronos import Chronos2Pipeline
from chronos2_time_only_encoder import strip_group_self_attention

sys.path.insert(0, "forecasting")
from train_tokenizer import load_mcs_episode_halves  # noqa: E402

from gormpo_world_model import FEATURE_NAMES, compute_variable_ylims  # noqa: E402

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
DATA_ROOT = "forecasting/data_raw/abiomed"
TOKENIZER_PATH = "forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth"
MODEL_ID = "saved_models/chronos2_xbar_notimegroup_mcs/finetuned-ckpt"
DEVICE = "cuda:5"
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
strip_group_self_attention(pipeline.model)
print("GroupSelfAttention stripped (eval-time, must match training-time ablation).")
q10_idx, q50_idx, q90_idx = pipeline.quantiles.index(0.1), pipeline.quantiles.index(0.5), pipeline.quantiles.index(0.9)

x_half, y_half = load_mcs_episode_halves(DATA_ROOT, "test", 6)
x_half, y_half = x_half.to(DEVICE), y_half.to(DEVICE)

t_ctx = np.arange(-5, 1) * 10    # -50, -40, ..., 0
t_pred = np.arange(1, 7) * 10    # +10, +20, ..., +60

for idx in SAMPLE_IDXS:
    x_ep = x_half[idx : idx + 1]  # (1, 12, patch_len)
    y_ep = y_half[idx : idx + 1]

    with torch.no_grad():
        x_bar_ctx, mu_ctx, sigma_ctx = tokenizer.patch_and_scale(x_ep)
        x_bar_fut, _, _ = tokenizer.patch_and_scale(y_ep)

    batch_inputs = [{
        "target": x_bar_ctx[0, 0:11, :].cpu().numpy(),
        "past_covariates": {"pump_level": x_bar_ctx[0, 11, :].cpu().numpy()},
        "future_covariates": {"pump_level": x_bar_fut[0, 11, :].cpu().numpy()},
    }]
    with torch.no_grad():
        preds = pipeline.predict(batch_inputs, prediction_length=6, batch_size=1)
    pred_x_bar = preds[0].to(DEVICE)  # (11, n_quantiles, patch_len)

    sigma_11 = sigma_ctx[0, 0:11].unsqueeze(-1)  # (11,1)
    mu_11 = mu_ctx[0, 0:11].unsqueeze(-1)
    q10 = (sigma_11 * pred_x_bar[:, q10_idx, :] + mu_11).cpu().numpy()
    q50 = (sigma_11 * pred_x_bar[:, q50_idx, :] + mu_11).cpu().numpy()
    q90 = (sigma_11 * pred_x_bar[:, q90_idx, :] + mu_11).cpu().numpy()

    ctx_phys = x_ep[0].cpu().numpy()      # (12, patch_len)
    y_phys = y_ep[0].cpu().numpy()        # (12, patch_len)
    pl_int = int(round(float(y_phys[11].mean())))

    n_features = 12
    ncols = 4
    nrows = (n_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    fig.suptitle(
        f"Test sample {idx}  |  realized P-level = {pl_int}  |  x_bar+Chronos-2, both attentions removed ({MODEL_ID})",
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
    path = os.path.join(OUT_DIR, f"chronos2_xbar_notimegroup_sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

print("Done.")
