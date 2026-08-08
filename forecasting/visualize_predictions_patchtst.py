"""
Plots ground truth vs the tuned PatchTST baseline (official HF PatchTSTForPrediction,
lib/models/patchtst.py) for the same 4 test samples used in ../visualize_predictions.py
and visualize_predictions_totem.py, using the same unified per-variable y-axis bounds
(gormpo_world_model.compute_variable_ylims) for direct visual comparability.

PatchTST uses the same data pipeline (data_provider -> Dataset_Neuro, root_path=
data_raw/abiomed) as TOTEM's own forecaster here, so the same StandardScaler
inverse-transform applies to recover physical units for plotting -- refit
identically to Dataset_Neuro's own __read_data__, since the pipeline never exposes
the fitted scaler directly.

Run with: python visualize_predictions_patchtst.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gormpo_world_model import FEATURE_NAMES, compute_variable_ylims  # noqa: E402

from train_patchtst import build_dataloaders  # noqa: E402

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
CHECKPOINT_PATH = "saved_models/patchtst_abiomed_tuned/checkpoints/patchtst_checkpoint.pth"
RAW_TRAIN_PATH = "data_raw/abiomed/train_data.npy"
DEVICE = "cuda:3"
SAMPLE_IDXS = [0, 1, 2, 3]
OUT_DIR = "../figures"

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

# Same StandardScaler Dataset_Neuro fits internally (root_path=data_raw/abiomed, shared
# with TOTEM's pipeline) -- refit here since it's never exposed by data_provider.
train_raw = np.load(RAW_TRAIN_PATH)
scaler = StandardScaler().fit(train_raw.reshape(-1, train_raw.shape[-1]))
scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32)

model = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.to(device).eval()

loaders, n_vars, _ = build_dataloaders("abiomed", seq_len=6, pred_len=6, num_workers=4)
test_loader = loaders["test"]
batch_x, batch_y, _, _ = next(iter(test_loader))  # shuffle=False for test -> preserves file order
batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)

with torch.no_grad():
    _, pred_norm = model.shared_eval((batch_x, batch_y), None, "val")  # (B, 6, 12), Dataset_Neuro-standardized space

pred_phys_all = pred_norm.cpu() * scaler_scale + scaler_mean
y_phys_all = batch_y.cpu() * scaler_scale + scaler_mean
x_phys_all = batch_x.cpu() * scaler_scale + scaler_mean

YLIMS = compute_variable_ylims(DATA_PATH)
t_ctx = np.arange(-5, 1) * 10
t_pred = np.arange(1, 7) * 10

for idx in SAMPLE_IDXS:
    ctx_phys = x_phys_all[idx]
    pred_phys = pred_phys_all[idx]
    y_phys = y_phys_all[idx]
    pl_int = int(round(y_phys[0, 11].item()))

    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    axes = axes.flatten()
    fig.suptitle(f"Test sample {idx}  |  P-level = {pl_int}  |  PatchTST (official HF PatchTSTForPrediction)", fontsize=12)

    for j in range(11):
        ax = axes[j]
        ax.plot(t_ctx, ctx_phys[:, j].numpy(), color="steelblue", marker="o", markersize=3, label="context")
        ax.plot(t_pred, y_phys[:, j].numpy(), color="green", marker="o", markersize=3, linestyle="--", label="ground truth")
        ax.plot(t_pred, pred_phys[:, j].numpy(), color="tomato", marker="o", markersize=3, label="prediction")
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylim(*YLIMS[FEATURE_NAMES[j]])
        ax.set_title(FEATURE_NAMES[j], fontsize=9)
        ax.set_xlabel("time (min)", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

    # PatchTST is channel-independent with no action-conditioning: it forecasts Pump
    # Level like any other channel rather than treating it as a fixed given action.
    ax = axes[11]
    ax.plot(t_ctx, ctx_phys[:, 11].numpy(), color="steelblue", marker="o", markersize=3, label="context")
    ax.plot(t_pred, y_phys[:, 11].numpy(), color="green", marker="o", markersize=3, linestyle="--", label="ground truth")
    ax.plot(t_pred, pred_phys[:, 11].numpy(), color="tomato", marker="o", markersize=3, label="prediction (unconditioned)")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(*YLIMS[FEATURE_NAMES[11]])
    ax.set_title(f"{FEATURE_NAMES[11]} (forecast, not given as action)", fontsize=9)
    ax.set_xlabel("time (min)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"patchtst_sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

print("Done.")
