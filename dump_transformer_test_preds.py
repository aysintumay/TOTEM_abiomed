"""
Save abiomed_env.model.WorldModel (Transformer) test-set forecasts in the same npz format
as the other models (pred/true: (N, 6, 11) z-score, time-major), so the shared
visualize_predictions.py can plot them. Point forecast = mean of 10 dropout
samples, same protocol as eval_transformer_world_model.py.

Run: python dump_transformer_test_preds.py --gpu 4
"""
import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "abiomed_env")
from config import model_kwargs_10min_1hr_full  # noqa: E402
from model import WorldModel  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model_path", default="/public/gormpo/models/10min_1hr_all_data_model.pth")
p.add_argument("--data_path", default="/public/gormpo/10min_1hr_all_data.pkl")
p.add_argument("--num_samples", type=int, default=10)
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--output_npz", default="logs/transformer_test_preds.npz")
args = p.parse_args()

torch.manual_seed(args.seed)
kw = dict(model_kwargs_10min_1hr_full)
kw["device"] = f"cuda:{args.gpu}"
model = WorldModel(**kw)
model.load_data(args.data_path)
model.load_model(args.model_path)
model.model.eval()

H, F = model.forecast_horizon, model.num_features - 1
preds, trues = [], []
with torch.no_grad():
    for x, pl, y in DataLoader(model.data_test, batch_size=256, shuffle=False):
        s = model.model.sample_multiple(x.float().to(model.device), pl.float().to(model.device),
                                        num_samples=args.num_samples)
        preds.append(s.reshape(args.num_samples, -1, H, F).mean(0).cpu().numpy())
        trues.append(y.reshape(-1, H, F).numpy())
pred, true = np.concatenate(preds).astype(np.float32), np.concatenate(trues).astype(np.float32)
np.savez(args.output_npz, pred=pred, true=true, n=len(pred), n_fallback=0)
print(f"saved {args.output_npz}: {pred.shape}; MAE(z)={np.abs(pred - true).mean():.4f}")
