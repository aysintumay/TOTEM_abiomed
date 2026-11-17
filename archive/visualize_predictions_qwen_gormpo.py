"""
python visualize_predictions_qwen_gormpo.py
Plots ground truth vs the fine-tuned Qwen2.5-7B+GORMPO model's predictions
(forecasting/train_gormpo_llm_forecaster.py, LoRA + GORMPO tokenizer) for a few test
samples, all 12 variables. Mirrors visualize_predictions_gormpo_chronos.py's layout/
style exactly (same FEATURE_NAMES, same fixed y-limits via
gormpo_world_model.compute_variable_ylims, same time axis, same sample indices) so
figures from every model in this project are directly comparable panel-for-panel.

Reuses train_gormpo_llm_forecaster.py's own create_gormpo_dataloader (for loading/
dtype-casting the extracted-token test split -- the exact same tensors
eval_gormpo_llm_forecaster_normalized.py's full-test-set eval used) and inference()
(the exact decode path already validated there), rather than re-reading the raw .npy
files by hand.

Unlike the other GORMPO-tokenizer models (GORMPO+Chronos, GORMPO few-shot), this
model's own generate_codes() is fully deterministic (argmax decoding, no sampling) --
see lib/models/gormpo_llm_forecaster.py's GormpoTokenLLMForecaster.generate_codes. So
there is no genuine sample-to-sample uncertainty to show; the prediction line has no
shaded band, unlike every other model's figure here (a real difference in what the
model does, not an oversight in this script).

Sample indices line up 1:1 with every other visualize_predictions*.py script: the
extracted GORMPO tokens (data/gormpo_tokens_mcs_scratch_long2000) come from
load_mcs_episode_halves on the same test split, in the same order, as everything else.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "forecasting")
from train_gormpo_llm_forecaster import create_gormpo_dataloader, inference  # noqa: E402

from gormpo_world_model import FEATURE_NAMES, compute_variable_ylims  # noqa: E402

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
TOKEN_DATA_ROOT = "forecasting/data/gormpo_tokens_mcs_scratch_long2000"
TOKENIZER_PATH = "forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth"
LLM_CHECKPOINT = "forecasting/saved_models/gormpo_llm_qwen7b/checkpoints/gormpo_llm_forecaster_checkpoint.pth"
DEVICE = "cuda:6"
SAMPLE_IDXS = [0, 1, 2, 3]  # same test samples as every other visualize_predictions* script
OUT_DIR = "figures"

os.makedirs(OUT_DIR, exist_ok=True)

# Same fixed, model-independent per-variable y-axis bounds every other visualize_predictions*
# script uses -- what makes all these figures directly comparable panel-for-panel.
YLIMS = compute_variable_ylims(DATA_PATH)

device = torch.device(DEVICE)
tokenizer = torch.load(TOKENIZER_PATH, map_location=device, weights_only=False)
tokenizer.to(device)
tokenizer.eval()

model = torch.load(LLM_CHECKPOINT, map_location=device, weights_only=False)
model.to(device)
model.eval()

# Reuse the exact same loader eval_gormpo_llm_forecaster_normalized.py's full-test-set
# run used, then pull out just SAMPLE_IDXS from its already-loaded/cast tensors instead
# of re-reading the .npy files by hand.
dataloaders = create_gormpo_dataloader(TOKEN_DATA_ROOT, batchsize=1, num_workers=0)
test_tensors = dataloaders["test"].dataset.tensors  # 13-tuple, same order train_gormpo_llm_forecaster._unpack expects
batch = tuple(t[SAMPLE_IDXS] for t in test_tensors)

with torch.no_grad():
    pred_time, labels_time = inference(batch, model, tokenizer, device)  # (4, patch_len, 12) physical each

pred_time = pred_time.cpu().numpy()
labels_time = labels_time.cpu().numpy()
ctx_phys_all = test_tensors[0][SAMPLE_IDXS].numpy()  # x_original, (4, patch_len, 12) physical, time-major

t_ctx = np.arange(-5, 1) * 10    # -50, -40, ..., 0
t_pred = np.arange(1, 7) * 10    # +10, +20, ..., +60

for row, idx in enumerate(SAMPLE_IDXS):
    ctx_phys = ctx_phys_all[row].T   # (12, patch_len)
    y_true = labels_time[row].T      # (12, patch_len)
    pred_phys = pred_time[row].T     # (12, patch_len)
    pl_int = int(round(float(y_true[11].mean())))

    n_features = 12
    ncols = 4
    nrows = (n_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    fig.suptitle(
        f"Test sample {idx}  |  realized P-level = {pl_int}  |  Qwen2.5-7B+GORMPO fine-tuned ({LLM_CHECKPOINT})",
        fontsize=12,
    )

    for j in range(11):
        ax = axes[j]
        ax.plot(t_ctx, ctx_phys[j], color="steelblue", marker="o", markersize=3, label="context")
        ax.plot(t_pred, y_true[j], color="green", marker="o", markersize=3, linestyle="--", label="ground truth")
        # no uncertainty band: generate_codes() is deterministic (argmax), not sampled --
        # a single prediction line is the whole story for this model, not an omission.
        ax.plot(t_pred, pred_phys[j], color="tomato", marker="o", markersize=3, label="prediction")
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylim(*YLIMS[FEATURE_NAMES[j]])
        ax.set_title(FEATURE_NAMES[j], fontsize=9)
        ax.set_xlabel("time (min)", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

    # Pump Level panel: this model DOES predict it (same convention as GORMPO+Chronos --
    # all 12 channels tokenized/predicted uniformly, see train_gormpo_chronos.py's
    # docstring for why that convention exists), drawn the same as every other panel,
    # just labeled secondary since it's not the point of the model.
    ax = axes[11]
    j = 11
    ax.plot(t_ctx, ctx_phys[j], color="steelblue", marker="o", markersize=3, label="context")
    ax.plot(t_pred, y_true[j], color="green", marker="o", markersize=3, linestyle="--", label="ground truth")
    ax.plot(t_pred, pred_phys[j], color="tomato", marker="o", markersize=3, label="prediction")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(*YLIMS[FEATURE_NAMES[11]])
    ax.set_title(f"{FEATURE_NAMES[11]} (secondary -- action, not the forecast target)", fontsize=9)
    ax.set_xlabel("time (min)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"qwen_gormpo_sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

print("Done.")
