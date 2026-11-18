"""
python visualize_predictions_gormpo_chronos.py
Plots ground truth vs the fine-tuned GORMPO+Chronos model's predictions
(train_gormpo_chronos.py / eval_gormpo_chronos.py) for a few test samples, all 12
variables. Mirrors visualize_predictions.py's layout/style exactly (same FEATURE_NAMES,
same fixed y-limits via gormpo_world_model.compute_variable_ylims, same time axis, same
sample indices) so figures from both models are directly comparable panel-for-panel.

Unlike visualize_predictions.py's Pump Level panel (shown as context + a fixed action
line, since the medllama few-shot model is given the action as free text and never
predicts it), this model genuinely predicts Pump Level's own token too (see
train_gormpo_chronos.py's docstring for why -- Chronos has no free-text side-channel to
inject an action into, so all 12 channels are tokenized and predicted uniformly). Its
panel is drawn the same way as every other forecast panel here, with a title note that
it's a secondary/diagnostic prediction, not a genuine forecast target.

Sample indices line up 1:1 with visualize_predictions.py's: verified the pickle
(DATA_PATH, used by gormpo_world_model.py) and the raw npy split
(forecasting/data_raw/abiomed, used by train/eval_gormpo_chronos.py) are the same
underlying episodes in the same order (test[idx] matches exactly across both formats).
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM

sys.path.insert(0, "forecasting")
from train_tokenizer import load_mcs_episode_halves  # noqa: E402

from chronos import ChronosModel  # noqa: E402
from eval_gormpo_chronos import compute_channel_bounds  # noqa: E402
from gormpo_world_model import FEATURE_NAMES, compute_variable_ylims  # noqa: E402
from train_gormpo_chronos import build_tokenizer  # noqa: E402

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
DATA_ROOT = "forecasting/data_raw/abiomed"
TOKENIZER_PATH = "forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth"
CHECKPOINT_DIR = "saved_models/gormpo_chronos/checkpoints/epoch_60"  # lowest val loss (2.3836) before it started rising
DEVICE = "cuda:3"
NUM_SAMPLES = 10
SAMPLE_IDXS = [0, 1, 2, 3]  # same test samples as visualize_predictions.py, for direct comparison
OUT_DIR = "figures"

os.makedirs(OUT_DIR, exist_ok=True)

# Same fixed, model-independent per-variable y-axis bounds visualize_predictions.py uses --
# what makes the two figures directly comparable panel-for-panel.
YLIMS = compute_variable_ylims(DATA_PATH)

tokenizer, chronos_config = build_tokenizer(TOKENIZER_PATH, DEVICE)
chronos_config.num_samples = NUM_SAMPLES
hf_model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_DIR).to(DEVICE).eval()
chronos_model = ChronosModel(config=chronos_config, model=hf_model)

bound_low, bound_high = compute_channel_bounds(DATA_ROOT, tokenizer.gormpo_tokenizer.patch_len)
bound_low, bound_high = bound_low.to(DEVICE), bound_high.to(DEVICE)

x_test, y_test = load_mcs_episode_halves(DATA_ROOT, "test", tokenizer.gormpo_tokenizer.patch_len)

for idx in SAMPLE_IDXS:
    x = x_test[idx : idx + 1].to(DEVICE)  # (1, N=12, k=6) physical, channel-major
    y_true = y_test[idx].numpy()          # (N=12, k=6) physical, channel-major

    with torch.no_grad():
        context_ids, context_mask, state = tokenizer.context_input_transform(x)  # (12, context_len)
        samples = chronos_model(
            input_ids=context_ids, attention_mask=context_mask.long(),
            prediction_length=tokenizer.label_len, num_samples=NUM_SAMPLES,
        )  # (12, num_samples, 6)
        decoded = tokenizer.output_transform(samples, state)  # (12, num_samples, 6) physical
        decoded = torch.clamp(decoded, bound_low.view(-1, 1, 1), bound_high.view(-1, 1, 1))

    pred_phys = decoded.mean(dim=1).cpu().numpy()  # (12, 6)
    std_phys = decoded.std(dim=1).cpu().numpy()    # (12, 6)
    ctx_phys = x.squeeze(0).cpu().numpy()          # (12, 6)
    pl_int = int(round(float(y_true[11].mean())))  # realized Pump Level for this target patch

    t_ctx = np.arange(-5, 1) * 10    # -50, -40, ..., 0
    t_pred = np.arange(1, 7) * 10    # +10, +20, ..., +60

    n_features = 12
    ncols = 4
    nrows = (n_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    fig.suptitle(f"Test sample {idx}  |  realized P-level = {pl_int}  |  GORMPO+Chronos ({CHECKPOINT_DIR})", fontsize=12)

    for j in range(11):
        ax = axes[j]
        ax.plot(t_ctx, ctx_phys[j], color="steelblue", marker="o", markersize=3, label="context")
        ax.plot(t_pred, y_true[j], color="green", marker="o", markersize=3, linestyle="--", label="ground truth")
        mu, sig = pred_phys[j], std_phys[j]
        ax.plot(t_pred, mu, color="tomato", marker="o", markersize=3, label="prediction")
        ax.fill_between(t_pred, mu - sig, mu + sig, color="tomato", alpha=0.2)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylim(*YLIMS[FEATURE_NAMES[j]])
        ax.set_title(FEATURE_NAMES[j], fontsize=9)
        ax.set_xlabel("time (min)", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

    # Pump Level panel: this model DOES predict it (unlike the medllama few-shot model,
    # which is only ever given it as free-text action -- see module docstring), so it's
    # drawn as a genuine prediction, just labeled as secondary/not the point of the model.
    ax = axes[11]
    j = 11
    ax.plot(t_ctx, ctx_phys[j], color="steelblue", marker="o", markersize=3, label="context")
    ax.plot(t_pred, y_true[j], color="green", marker="o", markersize=3, linestyle="--", label="ground truth")
    mu, sig = pred_phys[j], std_phys[j]
    ax.plot(t_pred, mu, color="tomato", marker="o", markersize=3, label="prediction")
    ax.fill_between(t_pred, mu - sig, mu + sig, color="tomato", alpha=0.2)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(*YLIMS[FEATURE_NAMES[11]])
    ax.set_title(f"{FEATURE_NAMES[11]} (secondary -- action, not the forecast target)", fontsize=9)
    ax.set_xlabel("time (min)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"gormpo_chronos_sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

print("Done.")
