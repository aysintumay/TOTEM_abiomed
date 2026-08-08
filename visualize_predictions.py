"""
python visualize_predictions.py
Plots ground truth vs GORMPO few-shot LLM world model predictions for a few test
samples, all 12 variables (11 physiological + Pump Level as the action/context
panel -- it's not forecast, but shown for completeness since it's part of state).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from gormpo_world_model import FEATURE_NAMES, GormpoFewShotWorldModel, compute_variable_ylims

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
TOKENIZER_PATH = "forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth"
MODEL_ID = "m42-health/Llama3-Med42-8B"
DEVICE = "cuda:3"
K_SHOT = 3
NUM_SAMPLES = 3
SAMPLE_IDXS = [0, 1, 2, 3]  # which test samples to visualize
OUT_DIR = "figures"

os.makedirs(OUT_DIR, exist_ok=True)

# Fixed, model-independent per-variable y-axis bounds (1st/99th training percentile),
# shared with GormpoFewShotWorldModel's own output-clipping bounds (see
# gormpo_world_model.py::compute_variable_ylims). Applying the SAME dict here on every
# panel, regardless of sample or model, is what makes figures from different models
# directly comparable -- any other model's visualization script should import and use
# this exact same function/data_path rather than computing its own ranges.
YLIMS = compute_variable_ylims(DATA_PATH)

model = GormpoFewShotWorldModel(
    tokenizer_path=TOKENIZER_PATH, model_id=MODEL_ID, forecast_horizon=6,
    k_shot=K_SHOT, num_samples=NUM_SAMPLES, device=DEVICE,
)
model.load_data(DATA_PATH)

for idx in SAMPLE_IDXS:
    x, pl, y = model.data_test[idx]
    context = torch.as_tensor(np.asarray(x)).float()  # (6, 12) normalized
    pl_int = int(round(model.unnorm_pl(torch.as_tensor(np.asarray(pl)).float().mean()).item()))

    mean_norm, std_norm = model.predict(context, p_level_int=pl_int)

    # unnormalize everything to physical units
    ctx_phys = model.unnorm_output(context)                        # (6, 12)
    pred_phys = model.unnorm_output(mean_norm)                     # (6, 12)
    std_phys = std_norm.float() * model.std[model.columns].cpu()   # (6, 12) — scale only

    # ground truth future: y is (66,) = (6, 11) normalized (no pump level)
    y_norm = torch.as_tensor(np.asarray(y)).float().reshape(6, 11)
    cols_no_pl = model.columns[:-1]
    y_phys = y_norm.cpu() * model.std[cols_no_pl] + model.mean[cols_no_pl]  # (6, 11)

    t_ctx = np.arange(-5, 1) * 10    # -50, -40, ..., 0
    t_pred = np.arange(1, 7) * 10    # +10, +20, ..., +60

    # plot all 12 variables: 11 physiological (forecast) + Pump Level (action, not forecast)
    n_features = 12
    ncols = 4
    nrows = (n_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    fig.suptitle(f"Test sample {idx}  |  P-level action = {pl_int}  |  GORMPO few-shot ({MODEL_ID})", fontsize=12)

    for j in range(11):
        ax = axes[j]
        ax.plot(t_ctx, ctx_phys[:, j].numpy(), color="steelblue", marker="o",
                markersize=3, label="context")
        ax.plot(t_pred, y_phys[:, j].numpy(), color="green", marker="o",
                markersize=3, linestyle="--", label="ground truth")
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

    # Pump Level panel: not a forecast target (it's the action), shown as context
    # history + the fixed new action level held forward, distinct style so it's
    # not mistaken for a genuine prediction.
    ax = axes[11]
    ax.plot(t_ctx, ctx_phys[:, 11].numpy(), color="steelblue", marker="o", markersize=3, label="context")
    ax.plot(t_pred, [pl_int] * len(t_pred), color="black", marker="s", markersize=3,
            linestyle="-", label="action (fixed)")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(*YLIMS[FEATURE_NAMES[11]])  # fixed (1, 11): P-level's natural range (2-10);
    # avoids a confusing floating-point-noise auto-scale when context is ~constant.
    ax.set_title(f"{FEATURE_NAMES[11]} (action, not forecast)", fontsize=9)
    ax.set_xlabel("time (min)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"gormpo_sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

print("Done.")
