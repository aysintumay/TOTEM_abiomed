"""
Evaluate TOTEM's own forecaster (XcodeYtimeDecoder + MuStdModel) and PatchTST on the
FULL MCS test set, with predictions/labels converted into the SAME reference
normalized space GORMPO's eval_gormpo_world_model.py uses (the pickle's own
per-channel mean/std), so all three models' MAE/MSE/DTW/corr numbers are directly
comparable -- not just visually, on the same 4 samples, but numerically, on the
full 3,876-example test set.

Both TOTEM and PatchTST's own pipelines run through Dataset_Neuro (root_path=
data_raw/abiomed), which standardizes with its own StandardScaler (fit on
train_data.npy) -- refit identically here (never exposed by data_provider) purely
to invert it back to physical units, then re-normalized with the pickle's mean/std
to land in the exact same space as the GORMPO evaluation.

Run with: python eval_baselines_normalized.py
"""
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from lib.models.metrics import dtw, pearsoncor
from train_forecaster import create_time_series_dataloader, inference
from train_patchtst import build_dataloaders

FEATURE_NAMES = ["PumpPressure", "PumpSpeed", "PumpFlow", "LVP", "LVEDP", "SYSTOLIC",
                  "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate", "ESE_lv"]
DEVICE = "cuda:3"
PICKLE_PATH = "/public/gormpo/10min_1hr_all_data.pkl"

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

# Dataset_Neuro's own scaler (shared by both TOTEM and PatchTST's pipelines here)
train_raw = np.load("data_raw/abiomed/train_data.npy")
neuro_scaler = StandardScaler().fit(train_raw.reshape(-1, train_raw.shape[-1]))
neuro_mean = torch.tensor(neuro_scaler.mean_, dtype=torch.float32)
neuro_scale = torch.tensor(neuro_scaler.scale_, dtype=torch.float32)

# The pickle's own per-channel mean/std -- the reference space GORMPO's evaluation used.
with open(PICKLE_PATH, "rb") as f:
    pkl = pickle.load(f)
columns = [i for i in range(13) if i != 11]
pkl_mean = torch.tensor(np.asarray(pkl["mean"])[columns[:-1]], dtype=torch.float32)
pkl_std = torch.tensor(np.asarray(pkl["std"])[columns[:-1]], dtype=torch.float32)


def to_reference_space(x_neuro_norm):
    """(*, 11) in Dataset_Neuro-standardized space -> (*, 11) in pickle-mean/std space."""
    phys = x_neuro_norm * neuro_scale[:-1] + neuro_mean[:-1]
    return (phys - pkl_mean) / pkl_std


def report(tag, pred_all, true_all):
    pred = torch.cat(pred_all, dim=0)  # (N, 6, 11) reference-normalized
    true = torch.cat(true_all, dim=0)
    mae = (pred - true).abs().mean(dim=(0, 1))
    mse = (pred - true).pow(2).mean(dim=(0, 1))
    corr = torch.tensor([pearsoncor(pred[:, :, j:j+1], true[:, :, j:j+1], reduction="mean").item() for j in range(11)])
    dtw_dist = torch.tensor([dtw(pred[:, :, j:j+1], true[:, :, j:j+1], reduction="mean").item() for j in range(11)])

    print(f"\n=== {tag}: full test set ({pred.shape[0]} examples), reference-normalized space ===")
    for j in range(11):
        print(f"  {FEATURE_NAMES[j]:<14} MAE={mae[j]:7.4f}  MSE={mse[j]:7.4f}  DTW={dtw_dist[j]:7.4f}  corr={corr[j]:7.4f}")
    print(f"Overall: MAE={mae.mean():.4f}  MSE={mse.mean():.4f}  DTW={dtw_dist.mean():.4f}  corr={corr.mean():.4f}")


# ---- TOTEM (XcodeYtimeDecoder + MuStdModel) ----
codebook = np.load("data/abiomed/Tin6_Tout6/codebook.npy", allow_pickle=True)
codebook = torch.from_numpy(codebook).to(device=device, dtype=torch.float32)
model_decode = torch.load(
    "saved_models/abiomed/forecaster_checkpoints_generalist/abiomed_Tin6_Tout6_seed2021/decode_checkpoint.pth",
    map_location=device, weights_only=False,
)
model_mustd = torch.load(
    "saved_models/abiomed/forecaster_checkpoints_generalist/abiomed_Tin6_Tout6_seed2021/mustd_checkpoint.pth",
    map_location=device, weights_only=False,
)
model_decode.to(device).eval()
model_mustd.to(device).eval()

totem_loaders = create_time_series_dataloader(datapath="data/abiomed/Tin6_Tout6", batchsize=128)
totem_pred, totem_true = [], []
with torch.no_grad():
    for batch in totem_loaders["test"]:
        pred_norm = inference(batch, model_decode, model_mustd, codebook, 2, device, onehot=False, scheme=2)
        totem_pred.append(to_reference_space(pred_norm[:, :, :-1].cpu()))
        totem_true.append(to_reference_space(batch[1][:, :, :-1].cpu()))
report("TOTEM (XcodeYtimeDecoder + MuStdModel)", totem_pred, totem_true)


# ---- PatchTST ----
patchtst_model = torch.load("saved_models/patchtst_abiomed_tuned/checkpoints/patchtst_checkpoint.pth",
                             map_location=device, weights_only=False)
patchtst_model.to(device).eval()

patchtst_loaders, _, _ = build_dataloaders("abiomed", seq_len=6, pred_len=6, num_workers=4)
patchtst_pred, patchtst_true = [], []
with torch.no_grad():
    for batch_x, batch_y, _, _ in patchtst_loaders["test"]:
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
        _, pred_norm = patchtst_model.shared_eval((batch_x, batch_y), None, "val")
        patchtst_pred.append(to_reference_space(pred_norm[:, :, :-1].cpu()))
        patchtst_true.append(to_reference_space(batch_y[:, :, :-1].cpu()))
report("PatchTST (official HF PatchTSTForPrediction)", patchtst_pred, patchtst_true)
