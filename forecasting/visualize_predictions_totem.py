"""
Plots ground truth vs TOTEM's own baseline forecaster (XcodeYtimeDecoder + MuStdModel,
lib/models/decode.py -- a from-scratch Transformer, NOT an LLM) for the same 4 test
samples used in ../visualize_predictions.py, using the same unified per-variable
y-axis bounds (gormpo_world_model.compute_variable_ylims) for direct visual
comparability against the GORMPO few-shot and PatchTST baselines.

TOTEM's own pipeline (train_forecaster.py) normalizes via Dataset_Neuro's global
StandardScaler (fit on train_data.npy), NOT the same per-channel mean/std the GORMPO
pickle uses -- inverse-transformed here with a scaler refit identically to
Dataset_Neuro's own __read_data__, to recover true physical units for plotting.

Run with: python visualize_predictions_totem.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gormpo_world_model import FEATURE_NAMES, compute_variable_ylims  # noqa: E402

from train_forecaster import create_time_series_dataloader, inference  # noqa: E402

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
FORECASTER_DATAROOT = "data/abiomed/Tin6_Tout6"
CHECKPOINT_DIR = "saved_models/abiomed/forecaster_checkpoints_generalist/abiomed_Tin6_Tout6_seed2021"
RAW_TRAIN_PATH = "data_raw/abiomed/train_data.npy"
DEVICE = "cuda:3"
COMPRESSION = 2
SAMPLE_IDXS = [0, 1, 2, 3]
OUT_DIR = "../figures"

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

# Refit the exact same StandardScaler Dataset_Neuro fits internally, to invert its
# normalization back to physical units (the pipeline itself never exposes the scaler).
train_raw = np.load(RAW_TRAIN_PATH)  # (episodes, T, 12) physical units
scaler = StandardScaler().fit(train_raw.reshape(-1, train_raw.shape[-1]))
scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32)   # (12,)
scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32)  # (12,)

codebook = np.load(f"{FORECASTER_DATAROOT}/codebook.npy", allow_pickle=True)
codebook = torch.from_numpy(codebook).to(device=device, dtype=torch.float32)

model_decode = torch.load(f"{CHECKPOINT_DIR}/decode_checkpoint.pth", map_location=device, weights_only=False)
model_mustd = torch.load(f"{CHECKPOINT_DIR}/mustd_checkpoint.pth", map_location=device, weights_only=False)
model_decode.to(device).eval()
model_mustd.to(device).eval()

dataloaders = create_time_series_dataloader(datapath=FORECASTER_DATAROOT, batchsize=128)
test_loader = dataloaders["test"]
batch = next(iter(test_loader))  # shuffle=False for test -> first batch preserves file order
x, y, codeids_x, codeids_y_labels = batch

YLIMS = compute_variable_ylims(DATA_PATH)

with torch.no_grad():
    pred_time_norm = inference(
        batch, model_decode, model_mustd, codebook, COMPRESSION, device, onehot=False, scheme=2,
    )  # (B, Tout=6, Sout=12), Dataset_Neuro-standardized space

pred_phys_all = pred_time_norm.cpu() * scaler_scale + scaler_mean   # (B, 6, 12)
y_phys_all = y.float() * scaler_scale + scaler_mean                 # (B, 6, 12) ground truth
x_phys_all = x.float() * scaler_scale + scaler_mean                 # (B, 6, 12) context

t_ctx = np.arange(-5, 1) * 10
t_pred = np.arange(1, 7) * 10

for idx in SAMPLE_IDXS:
    ctx_phys = x_phys_all[idx]     # (6, 12)
    pred_phys = pred_phys_all[idx]  # (6, 12)
    y_phys = y_phys_all[idx]        # (6, 12)
    pl_int = int(round(y_phys[0, 11].item()))  # Pump Level is fixed/known, not forecast

    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    axes = axes.flatten()
    fig.suptitle(f"Test sample {idx}  |  P-level = {pl_int}  |  TOTEM (XcodeYtimeDecoder + MuStdModel)", fontsize=12)

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

    ax = axes[11]
    ax.plot(t_ctx, ctx_phys[:, 11].numpy(), color="steelblue", marker="o", markersize=3, label="context")
    ax.plot(t_pred, y_phys[:, 11].numpy(), color="green", marker="o", markersize=3, linestyle="--", label="ground truth (=action)")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(*YLIMS[FEATURE_NAMES[11]])
    ax.set_title(f"{FEATURE_NAMES[11]} (action, not forecast)", fontsize=9)
    ax.set_xlabel("time (min)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"totem_sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

print("Done.")
