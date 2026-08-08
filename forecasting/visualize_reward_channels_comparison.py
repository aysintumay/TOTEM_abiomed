"""
For one test sample, plots context + ground truth + all 3 models' predictions
(GORMPO few-shot, TOTEM, PatchTST) overlaid on the SAME axes, one figure per
variable, for the 3 clinical reward channels (MAP/PumpPressure, PULSAT, Heart Rate
-- abiomed_env.cost_func's MAP_IDX/PULSATILITY_IDX/HR_IDX).

TOTEM and PatchTST are otherwise deterministic point-forecasters (no sampling
anywhere in their inference path), so PatchTST's uncertainty band comes from
MC-dropout: re-enable each model's own nn.Dropout submodules (left at their
trained p) while keeping every other submodule (esp. PatchTST's BatchNorm, which
must keep using running stats, not this one batch's) in eval mode, then run
MC_SAMPLES stochastic forward passes and take the empirical mean/std -- same
spread-from-repeated-sampling idea as GORMPO few-shot's own num_samples LLM
generations, just via dropout noise instead of temperature.

TOTEM has NO usable dropout signal under inference()'s scheme=2 path (the scheme
used here and by every other TOTEM script in this repo): XcodeYtimeDecoder (which
picks the codes) was trained with dropout=0.0 throughout, and MuStdModel's
dropout=0.2 output IS computed but then silently discarded -- scheme=2 denormalizes
with model_mustd.revin_in's fixed input statistics only (train_forecaster.py
inference(), the `elif scheme == 2` branch), never touching the MLP forward pass
that has dropout in it. So MC-dropout on TOTEM here is architecturally a no-op --
std is exactly 0.0 at every timestep, confirmed numerically, not just "small."
TOTEM is plotted as a genuine deterministic point forecast, no band -- adding one
would misrepresent it as having quantified uncertainty it doesn't have in this
pipeline. PatchTST's band is real (dropout=0.1 actually sits in its live forward
path) but is typically <0.02 normalized units here, i.e. a small fraction of one
physical unit -- expect it to look like a thin line, not a wide band; that's the
model being genuinely confident under MC-dropout, not a rendering issue.

Run with: python visualize_reward_channels_comparison.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gormpo_world_model import FEATURE_NAMES, GormpoFewShotWorldModel, compute_variable_ylims  # noqa: E402

from train_forecaster import create_time_series_dataloader, inference  # noqa: E402
from train_patchtst import build_dataloaders  # noqa: E402

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
GORMPO_TOKENIZER_PATH = "saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth"
GORMPO_MODEL_ID = "m42-health/Llama3-Med42-8B"
TOTEM_DATAROOT = "data/abiomed/Tin6_Tout6"
TOTEM_CHECKPOINT_DIR = "saved_models/abiomed/forecaster_checkpoints_generalist/abiomed_Tin6_Tout6_seed2021"
PATCHTST_CHECKPOINT_PATH = "saved_models/patchtst_abiomed_tuned/checkpoints/patchtst_checkpoint.pth"
RAW_TRAIN_PATH = "data_raw/abiomed/train_data.npy"
DEVICE = "cuda:3"
SAMPLE_IDX = 1
MC_SAMPLES = 30
CHANNELS = {"Heart Rate": 9, "PumpPressure": 0, "PULSAT": 7}  # PumpPressure = "MAP" in reward_func.py
OUT_DIR = "../figures"


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """model.eval() (freezes BatchNorm to running stats, disables every other
    train-mode side effect), then flips only the nn.Dropout submodules back to
    train() so repeated forward passes are stochastic -- MC-dropout."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    return model

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
YLIMS = compute_variable_ylims(DATA_PATH)
t_ctx = np.arange(-5, 1) * 10
t_pred = np.arange(1, 7) * 10

# Dataset_Neuro's own scaler (shared by TOTEM and PatchTST's pipelines)
train_raw = np.load(RAW_TRAIN_PATH)
neuro_scaler = StandardScaler().fit(train_raw.reshape(-1, train_raw.shape[-1]))
neuro_mean = torch.tensor(neuro_scaler.mean_, dtype=torch.float32)
neuro_scale = torch.tensor(neuro_scaler.scale_, dtype=torch.float32)

# ---- GORMPO few-shot ----
print("Running GORMPO few-shot...")
gormpo_model = GormpoFewShotWorldModel(
    tokenizer_path=GORMPO_TOKENIZER_PATH, model_id=GORMPO_MODEL_ID, forecast_horizon=6,
    k_shot=3, num_samples=3, device=DEVICE,
)
gormpo_model.load_data(DATA_PATH)
x, pl, y = gormpo_model.data_test[SAMPLE_IDX]
context_norm = torch.tensor(np.asarray(x), dtype=torch.float32)
pl_int = int(round(gormpo_model.unnorm_pl(torch.tensor(np.asarray(pl)).float().mean()).item()))
gormpo_mean_norm, gormpo_std_norm = gormpo_model.predict(context_norm, p_level_int=pl_int)
gormpo_ctx_phys = gormpo_model.unnorm_output(context_norm)
gormpo_pred_phys = gormpo_model.unnorm_output(gormpo_mean_norm)
gormpo_std_phys = gormpo_std_norm.float() * gormpo_model.std[gormpo_model.columns].cpu()
y_norm = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(6, 11)
cols_no_pl = gormpo_model.columns[:-1]
gt_phys = y_norm.cpu() * gormpo_model.std[cols_no_pl] + gormpo_model.mean[cols_no_pl]

# ---- TOTEM (deterministic under scheme=2 -- see module docstring; no dropout ----
# ---- signal reaches the output, so this is a single forward pass, no band) -----
print("Running TOTEM (deterministic, single forward pass)...")
codebook = np.load(f"{TOTEM_DATAROOT}/codebook.npy", allow_pickle=True)
codebook = torch.from_numpy(codebook).to(device=device, dtype=torch.float32)
model_decode = torch.load(f"{TOTEM_CHECKPOINT_DIR}/decode_checkpoint.pth", map_location=device, weights_only=False)
model_mustd = torch.load(f"{TOTEM_CHECKPOINT_DIR}/mustd_checkpoint.pth", map_location=device, weights_only=False)
model_decode.to(device).eval()
model_mustd.to(device).eval()
totem_loader = create_time_series_dataloader(datapath=TOTEM_DATAROOT, batchsize=128)["test"]
totem_batch = next(iter(totem_loader))
with torch.no_grad():
    totem_pred_norm = inference(totem_batch, model_decode, model_mustd, codebook, 2, device, onehot=False, scheme=2)
totem_pred_phys = totem_pred_norm[SAMPLE_IDX].cpu() * neuro_scale + neuro_mean

# ---- PatchTST (MC-dropout: MC_SAMPLES stochastic passes -> empirical mean/std) ----
print(f"Running PatchTST ({MC_SAMPLES} MC-dropout samples)...")
patchtst_model = torch.load(PATCHTST_CHECKPOINT_PATH, map_location=device, weights_only=False)
patchtst_model.to(device)
enable_mc_dropout(patchtst_model)
patchtst_loader = build_dataloaders("abiomed", seq_len=6, pred_len=6, num_workers=4)[0]["test"]
p_batch_x, p_batch_y, _, _ = next(iter(patchtst_loader))
p_batch_x, p_batch_y = p_batch_x.float().to(device), p_batch_y.float().to(device)
patchtst_mc_preds = []
with torch.no_grad():
    for _ in range(MC_SAMPLES):
        _, pred_norm = patchtst_model.shared_eval((p_batch_x, p_batch_y), None, "val")
        patchtst_mc_preds.append(pred_norm[SAMPLE_IDX].cpu())
patchtst_mc_preds = torch.stack(patchtst_mc_preds)  # (MC_SAMPLES, 6, 12)
patchtst_pred_phys = patchtst_mc_preds.mean(dim=0) * neuro_scale + neuro_mean
patchtst_std_phys = patchtst_mc_preds.std(dim=0) * neuro_scale
for _name, _j in CHANNELS.items():
    print(f"  PatchTST MC-dropout std ({_name}, physical units): {patchtst_std_phys[:, _j].numpy().round(3)}")

# ---- plot: one figure per variable, all 3 models overlaid ----
for name, j in CHANNELS.items():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_ctx, gormpo_ctx_phys[:, j].numpy(), color="steelblue", marker="o", markersize=4, label="context")
    ax.plot(t_pred, gt_phys[:, j].numpy(), color="green", marker="o", markersize=4, linestyle="--", label="ground truth")

    mu = gormpo_pred_phys[:, j].numpy()
    sig = gormpo_std_phys[:, j].numpy()
    ax.plot(t_pred, mu, color="tomato", marker="o", markersize=4, label="GORMPO few-shot")
    ax.fill_between(t_pred, mu - sig, mu + sig, color="tomato", alpha=0.15)

    totem_mu = totem_pred_phys[:, j].numpy()
    ax.plot(t_pred, totem_mu, color="purple", marker="s", markersize=4, label="TOTEM (deterministic)")

    patchtst_mu = patchtst_pred_phys[:, j].numpy()
    patchtst_sig = patchtst_std_phys[:, j].numpy()
    ax.plot(t_pred, patchtst_mu, color="darkorange", marker="^", markersize=4, label="PatchTST (MC-dropout)")
    ax.fill_between(t_pred, patchtst_mu - patchtst_sig, patchtst_mu + patchtst_sig, color="darkorange", alpha=0.15)

    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylim(*YLIMS[name])
    label = f"{name} (MAP)" if name == "PumpPressure" else name
    ax.set_title(f"Test sample {SAMPLE_IDX}  |  P-level = {pl_int}  |  {label}", fontsize=11)
    ax.set_xlabel("time (min)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    path = os.path.join(OUT_DIR, f"compare_{fname}_sample{SAMPLE_IDX:03d}.png")
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Saved {path}")

print("Done.")
