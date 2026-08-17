"""
Evaluate abiomed_env.model.WorldModel (the original Transformer digital twin, trained
on MCS data -- see abiomed_env/README.md and mcs.md's project overview: this is the
model the LLM/Chronos world models are meant to replace/compare against) on the real
MCS test split, using the exact per-channel MAE/MSE/DTW/Pearson-correlation convention
every other eval script in this project uses, so results are directly comparable to
zero-shot Chronos / fine-tuned GORMPO+Chronos / GORMPO-few-shot / medllama.

abiomed_env/evaluate_transformer.py already evaluates this same checkpoint, but with a
different metric set (aggregate MSE/MAE/CRPS + static/dynamic MAE + trend accuracy, no
per-channel breakdown, no DTW/Pearson) -- this script reuses its exact architecture
config (abiomed_env/config.py's model_kwargs_10min_1hr_full) and its
WorldModel.test_output_multiple() protocol (10 stochastic forward passes, dropout-based
uncertainty per WorldModel.sample_multiple), just adds the metrics needed for
comparability with this project's LLM/Chronos world models.

Run with: python eval_transformer_world_model.py --model_path /public/gormpo/models/10min_1hr_all_data_model.pth
"""
import argparse
import sys

import torch

sys.path.insert(0, "abiomed_env")
sys.path.insert(0, "forecasting")
from config import model_kwargs_10min_1hr_full  # noqa: E402
from model import WorldModel  # noqa: E402
from lib.models.metrics import dtw, pearsoncor  # noqa: E402

FEATURE_NAMES = [
    "PumpPressure", "PumpSpeed", "PumpFlow", "LVP", "LVEDP", "SYSTOLIC",
    "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate", "ESE_lv", "Pump Level",
]


def main(args):
    model_kwargs = dict(model_kwargs_10min_1hr_full)
    model_kwargs["device"] = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"

    model = WorldModel(**model_kwargs)
    model.load_data(args.data_path)
    model.load_model(args.model_path)
    model.model.eval()

    print(f"Running {args.num_samples}-sample stochastic inference over the full test set...")
    outputs, ys, _ = model.test_output_multiple(num_samples=args.num_samples)
    # outputs: list of (B, num_samples, horizon, 11) : ys: list of (B, horizon, 11)
    pred = torch.cat(outputs, dim=0).mean(dim=1).cpu()  # (N, horizon, 11) -- mean over samples
    true = torch.cat(ys, dim=0).cpu()                   # (N, horizon, 11)
    print(f"Evaluated on {pred.shape[0]} test episodes (full test set)")

    mae = (pred - true).abs().mean(dim=(0, 1))   # (11,)
    mse = (pred - true).pow(2).mean(dim=(0, 1))
    corr = torch.tensor([
        pearsoncor(pred[:, :, j:j + 1], true[:, :, j:j + 1], reduction="mean").item()
        for j in range(pred.shape[-1])
    ])
    dtw_dist = torch.tensor([
        dtw(pred[:, :, j:j + 1], true[:, :, j:j + 1], reduction="mean").item()
        for j in range(pred.shape[-1])
    ])

    print("\n=== Per-variable MAE / MSE / DTW / Pearson corr (normalized/z-score units, averaged over all samples) ===")
    for j in range(11):
        print(f"  {FEATURE_NAMES[j]:<14} MAE={mae[j]:7.4f}  MSE={mse[j]:7.4f}  DTW={dtw_dist[j]:7.4f}  corr={corr[j]:7.4f}")
    print(
        f"\nOverall (mean across 11 forecast variables): "
        f"MAE={mae.mean():.4f}  MSE={mse.mean():.4f}  DTW={dtw_dist.mean():.4f}  corr={corr.mean():.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/public/gormpo/models/10min_1hr_all_data_model.pth")
    parser.add_argument("--data_path", type=str, default="/public/gormpo/10min_1hr_all_data.pkl")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
