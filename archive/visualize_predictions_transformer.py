"""
python visualize_predictions_transformer.py
Plots ground truth vs abiomed_env.model.WorldModel (the purpose-trained Transformer
digital twin) predictions for a few test samples, all 12 variables. Mirrors
visualize_predictions.py's layout/style exactly (same FEATURE_NAMES, same fixed
y-limits via gormpo_world_model.compute_variable_ylims, same time axis, same sample
indices) so figures from all three models (medllama few-shot, GORMPO+Chronos,
Transformer) are directly comparable panel-for-panel.

Like visualize_predictions.py's Pump Level panel (not visualize_predictions_gormpo_
chronos.py's): this model is GIVEN the future Pump Level as an input to condition on
(WorldModel.forward's `pl` argument), not something it predicts, so its panel is drawn
as context + the realized action held forward, not a genuine prediction.

Uncertainty comes from WorldModel's own dropout-based stochastic sampling
(TimeSeriesTransformer.sample_multiple, train()-mode dropout across repeated forward
passes), matching abiomed_env/evaluate_transformer.py's protocol.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "abiomed_env")
from config import model_kwargs_10min_1hr_full  # noqa: E402
from model import WorldModel  # noqa: E402

from gormpo_world_model import FEATURE_NAMES, compute_variable_ylims  # noqa: E402

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
MODEL_PATH = "/public/gormpo/models/10min_1hr_all_data_model.pth"
DEVICE = "cuda:0"
NUM_SAMPLES = 10
SAMPLE_IDXS = [0, 1, 2, 3]  # same test samples as the other visualize_predictions* scripts
OUT_DIR = "figures"

os.makedirs(OUT_DIR, exist_ok=True)

# Same fixed, model-independent per-variable y-axis bounds every other visualize_predictions*
# script uses -- what makes all these figures directly comparable panel-for-panel.
YLIMS = compute_variable_ylims(DATA_PATH)

model_kwargs = dict(model_kwargs_10min_1hr_full)
model_kwargs["device"] = DEVICE
model = WorldModel(**model_kwargs)
model.load_data(DATA_PATH)
model.load_model(MODEL_PATH)
model.model.eval()

for idx in SAMPLE_IDXS:
    x, pl, y = model.data_test[idx]  # x: (6,12) norm, pl: (6,) norm, y: (66,) norm flattened
    src = x.unsqueeze(0).float()      # (1, 6, 12)
    pl_in = pl.unsqueeze(0).float()   # (1, 6)

    with torch.no_grad():
        samples = model.model.sample_multiple(src.to(model.device), pl_in.to(model.device), num_samples=NUM_SAMPLES)
        # (num_samples, 1, 66) -> (num_samples, 1, 6, 11)
        samples = samples.reshape(NUM_SAMPLES, 1, model.forecast_horizon, model.num_features - 1)

    pred_norm = samples.mean(dim=0).squeeze(0).cpu()  # (6, 11)
    std_norm = samples.std(dim=0).squeeze(0).cpu()    # (6, 11)

    # unnormalize everything to physical units
    cols_no_pl = model.columns[:-1]
    ctx_phys = x.cpu() * model.std[model.columns] + model.mean[model.columns]         # (6, 12)
    pred_phys = pred_norm * model.std[cols_no_pl] + model.mean[cols_no_pl]            # (6, 11)
    std_phys = std_norm * model.std[cols_no_pl]                                       # (6, 11) -- scale only
    y_phys = y.cpu().reshape(6, 11) * model.std[cols_no_pl] + model.mean[cols_no_pl]   # (6, 11)
    pl_phys = (pl.cpu() * model.std[-1] + model.mean[-1])                             # (6,) realized P-level
    pl_int = int(round(pl_phys.mean().item()))

    t_ctx = np.arange(-5, 1) * 10    # -50, -40, ..., 0
    t_pred = np.arange(1, 7) * 10    # +10, +20, ..., +60

    n_features = 12
    ncols = 4
    nrows = (n_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()
    fig.suptitle(f"Test sample {idx}  |  realized P-level = {pl_int}  |  Transformer WorldModel ({MODEL_PATH})", fontsize=12)

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

    # Pump Level panel: given as an input (the action), not predicted -- drawn the same
    # way visualize_predictions.py's medllama panel is, for consistency.
    ax = axes[11]
    ax.plot(t_ctx, ctx_phys[:, 11].numpy(), color="steelblue", marker="o", markersize=3, label="context")
    ax.plot(t_pred, pl_phys.numpy(), color="black", marker="s", markersize=3, linestyle="-", label="action (given)")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(*YLIMS[FEATURE_NAMES[11]])
    ax.set_title(f"{FEATURE_NAMES[11]} (action, not forecast)", fontsize=9)
    ax.set_xlabel("time (min)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"transformer_sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

print("Done.")
