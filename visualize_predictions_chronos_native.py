"""
python visualize_predictions_chronos_native.py
Plots ground truth vs the Chronos-native fine-tuned model's predictions (their own
scripts/training/train.py pipeline + MeanScaleUniformBins tokenizer, fine-tuned on MCS
via prepare_chronos_native_data.py's data -- see saved_models/chronos_native) for a few
test samples. Mirrors visualize_predictions.py's layout/style exactly (same
FEATURE_NAMES, same fixed y-limits via gormpo_world_model.compute_variable_ylims, same
time axis, same sample indices) so figures from every model in this project are
directly comparable panel-for-panel.

Uses chronos_world_model.ChronosWorldModel exactly as eval_chronos_world_model.py does
for zero-shot Chronos, just pointed at the local fine-tuned checkpoint instead of the
HF hub id -- ChronosWorldModel only ever forecasts the 11 physiological channels (see
its own docstring), so Pump Level's panel here is context + the realized action held
forward, same convention as visualize_predictions.py's medllama panel (not a genuine
prediction -- this model was never asked to forecast Pump Level, unlike GORMPO+Chronos).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from chronos_world_model import ChronosWorldModel, FEATURE_NAMES
from gormpo_world_model import compute_variable_ylims

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
MODEL_ID = "saved_models/chronos_native/run-0/checkpoint-final"
DEVICE = "cuda:3"
NUM_SAMPLES = 20
SAMPLE_IDXS = [0, 1, 2, 3]  # same test samples as every other visualize_predictions* script
OUT_DIR = "figures"

os.makedirs(OUT_DIR, exist_ok=True)

# Same fixed, model-independent per-variable y-axis bounds every other visualize_predictions*
# script uses -- what makes all these figures directly comparable panel-for-panel.
YLIMS = compute_variable_ylims(DATA_PATH)

model = ChronosWorldModel(model_id=MODEL_ID, forecast_horizon=6, num_samples=NUM_SAMPLES, device=DEVICE)
model.load_data(DATA_PATH)

for idx in SAMPLE_IDXS:
    x, pl, y = model.data_test[idx]
    context = torch.as_tensor(np.asarray(x)).float()  # (6, 12) normalized
    pl_int = int(round(model.unnorm_pl(torch.as_tensor(np.asarray(pl)).float().mean()).item()))

    mean_norm, std_norm = model.predict(context, p_level_int=pl_int)

    # unnormalize everything to physical units
    ctx_phys = model.unnorm_output(context)                        # (6, 12)
    pred_phys = model.unnorm_output(mean_norm)                     # (6, 12)
    std_phys = std_norm.float() * model.std[model.columns].cpu()   # (6, 12) -- scale only

    # ground truth future: y is (66,) = (6, 11) normalized (no pump level)
    y_norm = torch.as_tensor(np.asarray(y)).float().reshape(6, 11)
    cols_no_pl = model.columns[:-1]
    y_phys = y_norm.cpu() * model.std[cols_no_pl] + model.mean[cols_no_pl]  # (6, 11)

    t_ctx = np.arange(-5, 1) * 10    # -50, -40, ..., 0
    t_pred = np.arange(1, 7) * 10    # +10, +20, ..., +60

    n_features = 12
    ncols = 4
    nrows = (n_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    fig.suptitle(f"Test sample {idx}  |  P-level action = {pl_int}  |  Chronos-native fine-tuned ({MODEL_ID})", fontsize=12)

    for j in range(11):
        ax = axes[j]
        ax.plot(t_ctx, ctx_phys[:, j].numpy(), color="steelblue", marker="o", markersize=3, label="context")
        ax.plot(t_pred, y_phys[:, j].numpy(), color="green", marker="o", markersize=3, linestyle="--", label="ground truth")
        mu = pred_phys[:, j].numpy()
        sig = std_phys[:, j].numpy()
        ax.plot(t_pred, mu, color="tomato", marker="o", markersize=3, label="prediction")
        ax.fill_between(t_pred, mu - sig, mu + sig, color="tomato", alpha=0.2)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylim(*YLIMS[FEATURE_NAMES[j]])
        ax.set_title(FEATURE_NAMES[j], fontsize=9)
        ax.set_xlabel("time (min)", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

    # Pump Level panel: not a forecast target for this model (see module docstring) --
    # context history + the fixed new action level held forward.
    ax = axes[11]
    ax.plot(t_ctx, ctx_phys[:, 11].numpy(), color="steelblue", marker="o", markersize=3, label="context")
    ax.plot(t_pred, [pl_int] * len(t_pred), color="black", marker="s", markersize=3,
            linestyle="-", label="action (fixed)")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(*YLIMS[FEATURE_NAMES[11]])
    ax.set_title(f"{FEATURE_NAMES[11]} (action, not forecast)", fontsize=9)
    ax.set_xlabel("time (min)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"chronos_native_sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

print("Done.")
