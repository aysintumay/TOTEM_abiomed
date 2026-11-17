"""
python eval_gormpo_lora_normalized.py

Runs full-test-set inference for the two LoRA-fine-tuned GormpoTokenLLMForecaster
checkpoints (Med42-8B, Qwen2.5-7B) and reports MAE/MSE/DTW/corr in per-channel
NORMALIZED (z-score) units, averaged over the 11 physiological variables -- the same
convention eval_gormpo_world_model.py uses for the ICL models, so these numbers are
directly comparable across all four models. train_gormpo_llm_forecaster.py's own
run_eval() instead pools every element (all channels, all physical scales) into one
global mean, which is not the same quantity.
"""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, "forecasting")
from train_gormpo_llm_forecaster import create_gormpo_dataloader, inference  # noqa: E402

from gormpo_world_model import FEATURE_NAMES  # noqa: E402 (sets up sys.path)
from lib.models.metrics import dtw, pearsoncor  # noqa: E402

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
TOKEN_DATA_ROOT = "forecasting/data/gormpo_tokens_mcs_scratch_long2000"
TOKENIZER_PATH = "forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth"
DEVICE = "cuda:0"
COLUMNS = [i for i in range(0, 13) if i != 11]  # matches GormpoFewShotWorldModel.columns

CHECKPOINTS = {
    "Med42-8B LoRA": "forecasting/saved_models/gormpo_llm_med42_mcs/checkpoints/gormpo_llm_forecaster_checkpoint.pth",
    "Qwen2.5-7B LoRA": "forecasting/saved_models/gormpo_llm_qwen7b/checkpoints/gormpo_llm_forecaster_checkpoint.pth",
}

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)
mean = torch.tensor(data["mean"], dtype=torch.float32)[COLUMNS]  # (12,)
std = torch.tensor(data["std"], dtype=torch.float32)[COLUMNS]

device = torch.device(DEVICE)
tokenizer = torch.load(TOKENIZER_PATH, map_location=device, weights_only=False)
tokenizer.to(device)
tokenizer.eval()

dataloaders = create_gormpo_dataloader(TOKEN_DATA_ROOT, batchsize=128, num_workers=0)
test_dataloader = dataloaders["test"]

print("=" * 88)
for name, ckpt_path in CHECKPOINTS.items():
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_dataloader:
            pred_time, labels_time = inference(batch, model, tokenizer, device)  # (B, 6, 12) physical
            all_pred.append(pred_time.cpu())
            all_true.append(labels_time.cpu())
    pred = torch.cat(all_pred, dim=0)[:, :, :11]  # drop Pump Level column, (N, 6, 11)
    true = torch.cat(all_true, dim=0)[:, :, :11]

    pred_norm = (pred - mean[:11]) / std[:11]
    true_norm = (true - mean[:11]) / std[:11]

    mae = (pred_norm - true_norm).abs().mean(dim=(0, 1))
    mse = (pred_norm - true_norm).pow(2).mean(dim=(0, 1))
    dtw_per = torch.stack([
        dtw(pred_norm[:, :, j : j + 1], true_norm[:, :, j : j + 1], reduction="mean") for j in range(11)
    ]).squeeze()
    corr_per = torch.stack([
        pearsoncor(pred_norm[:, :, j : j + 1], true_norm[:, :, j : j + 1], reduction="mean") for j in range(11)
    ]).squeeze()

    print(f"\n{name}  (n={pred.shape[0]}, normalized z-score units)")
    print(f"  {'variable':<14}{'MAE':>8}{'MSE':>8}{'DTW':>8}{'corr':>8}")
    for j in range(11):
        print(f"  {FEATURE_NAMES[j]:<14}{mae[j]:>8.4f}{mse[j]:>8.4f}{dtw_per[j]:>8.4f}{corr_per[j]:>8.4f}")
    print(f"  {'OVERALL':<14}{mae.mean():>8.4f}{mse.mean():>8.4f}{dtw_per.mean():>8.4f}{corr_per.mean():>8.4f}")

    del model
    torch.cuda.empty_cache()
print("=" * 88)

# ── diagnostic: outlier check on SYSTOLIC/PULSAT physical-unit predictions ──
