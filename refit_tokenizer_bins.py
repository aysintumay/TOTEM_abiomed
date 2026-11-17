"""
python refit_tokenizer_bins.py

Refits the GORMPO tokenizer's fixed scalar-bin quantizers (q_mu, q_sigma, q_min,
q_max) on the IMPUTED training data (impute_outliers_and_save.py -- the 2 corrupted
SYSTOLIC/PULSAT timesteps replaced with sane values), reconstructed the same way
extract_abiomed_data.py + load_mcs_patches would (same column selection
abiomed_env/model.py uses) rather than requiring the intermediate
data_raw/abiomed/*.npy files or a WorldModel checkpoint. patch_and_scale() itself is
back to its original, unmodified form (an earlier masking-based approach here was
reverted -- imputing the raw data once, upstream, is cleaner and doesn't risk
silently changing shape-code assignments the way masking mu/sigma computation would).

Everything else (encoder/decoder/VQ codebook/cross-channel-attention weights) is
left untouched -- this only refits the 4 fixed quantizers' bin edges. Saved to a
NEW checkpoint path so the original (which every existing LoRA checkpoint and
extracted-token cache was built against) is left intact.
"""
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "forecasting")
from train_tokenizer import fit_bin_quantizers  # noqa: E402 (reuse the exact original fitting logic)

DATA_PATH = "forecasting/data_raw/10min_1hr_all_data_imputed.pkl"
TOKENIZER_PATH = "forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth"
OUT_PATH = "forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000_fixed/checkpoints/final_tokenizer.pth"
COLUMNS = [i for i in range(0, 13) if i != 11]  # abiomed_env/model.py's exact convention, 12 channels
DEVICE = "cuda:0"
SYSTOLIC, PULSAT = 5, 7

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

# Reconstruct the exact same (num_patches, 12, 6) training patches
# extract_abiomed_data.py + load_mcs_patches would have produced: physical units,
# 12-channel (COLUMNS-filtered), context half + target half of every episode.
train = np.asarray(data["train"])[:, :, COLUMNS]  # (N, 12, 12): (episodes, T=2*patch_len, C)
ctx_half = np.transpose(train[:, 0:6, :], (0, 2, 1))   # (N, 12, 6)
tgt_half = np.transpose(train[:, 6:12, :], (0, 2, 1))  # (N, 12, 6)
train_patches = torch.from_numpy(np.concatenate([ctx_half, tgt_half], axis=0)).float()
print(f"train_patches: {train_patches.shape}")

device = torch.device(DEVICE)
tokenizer = torch.load(TOKENIZER_PATH, map_location=device, weights_only=False)
tokenizer.to(device)
tokenizer.eval()

# ---- BEFORE snapshot ----
before_mu_edges = tokenizer.q_mu.edges.clone()
before_sigma_edges = tokenizer.q_sigma.edges.clone()
before_min_edges = tokenizer.q_min.edges.clone()
before_max_edges = tokenizer.q_max.edges.clone()

train_loader = DataLoader(TensorDataset(train_patches), batch_size=256, shuffle=False, num_workers=0)
fit_bin_quantizers(tokenizer, train_loader, device)

torch.save(tokenizer, OUT_PATH)
print(f"Saved refit tokenizer to {OUT_PATH}")

print("\n=== q_sigma top-bin edges: before vs after refit ===")
for ch, name in [(SYSTOLIC, "SYSTOLIC"), (PULSAT, "PULSAT")]:
    b_lo, b_hi = before_sigma_edges[ch, -2].item(), before_sigma_edges[ch, -1].item()
    a_lo, a_hi = tokenizer.q_sigma.edges[ch, -2].item(), tokenizer.q_sigma.edges[ch, -1].item()
    print(f"{name:<10} BEFORE top bin [{b_lo:.2f}, {b_hi:.2f}] center={(b_lo+b_hi)/2:.2f}"
          f"   AFTER top bin [{a_lo:.2f}, {a_hi:.2f}] center={(a_lo+a_hi)/2:.2f}")

print("\n=== q_mu top/bottom-bin edges: before vs after refit ===")
for ch, name in [(SYSTOLIC, "SYSTOLIC"), (PULSAT, "PULSAT")]:
    b = before_mu_edges[ch]
    a = tokenizer.q_mu.edges[ch]
    print(f"{name:<10} BEFORE [min={b[0].item():.2f}, max={b[-1].item():.2f}]"
          f"   AFTER [min={a[0].item():.2f}, max={a[-1].item():.2f}]")
