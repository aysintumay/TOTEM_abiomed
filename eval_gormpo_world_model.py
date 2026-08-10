"""
Evaluate GormpoFewShotWorldModel (gormpo_world_model.py) on the real MCS test split
using the real medical LLM (m42-health/Llama3-Med42-8B), in-context, zero fine-tuning.

Metrics (MAE, MSE, DTW, Pearson correlation) are computed in normalized (z-score)
space, averaged across all evaluated samples -- reuses lib.models.metrics
(pearsoncor, dtw) from the forecasting/ tree, same functions the rest of this
project's evaluation scripts use, rather than reimplementing them here.

Run with: python eval_gormpo_world_model.py --num_examples 20
"""
import argparse
import contextlib
import io

import numpy as np
import torch

from gormpo_world_model import GormpoFewShotWorldModel, FEATURE_NAMES
from lib.models.metrics import dtw, pearsoncor


def main(args):
    model = GormpoFewShotWorldModel(
        tokenizer_path=args.tokenizer_path,
        model_id=args.model_id,
        forecast_horizon=6,
        k_shot=args.k_shot,
        num_samples=args.num_samples,
        device=args.device,
        lora_adapter_path=args.lora_adapter_path,
    )
    model.load_data(args.data_path)

    n = min(args.num_examples, len(model.data_test))
    rng = np.random.RandomState(args.random_seed)
    indices = rng.choice(len(model.data_test), size=n, replace=False) if args.random_sample else np.arange(n)
    n_fallback = 0
    all_pred, all_true = [], []

    for progress, i in enumerate(indices):
        x, pl, y = model.data_test[i]
        context_norm = torch.tensor(np.asarray(x), dtype=torch.float32)
        pl_raw = float(np.asarray(pl).mean())
        p_level_int = int(round(pl_raw * model.std[-1].item() + model.mean[-1].item()))

        # y is already z-score normalized (TimeSeriesDataset's own output, same mean/std
        # as context_norm) -- (66,) = (forecast_horizon, 11), no Pump Level column.
        y_norm = np.asarray(y, dtype=np.float32).reshape(model.forecast_horizon, model.num_channels - 1)

        before_fallback_marker = "[GormpoFewShot] All"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mean_norm, std_norm = model.predict(context_norm, p_level_int)
        printed = buf.getvalue()
        print(printed, end="")
        if before_fallback_marker in printed:
            n_fallback += 1

        # mean_norm is already normalized (that's predict()'s own output space); compare
        # directly, no unnorm -- errors here are in the same z-score units for every
        # channel, so they're comparable across channels of wildly different physical
        # scale (e.g. PumpSpeed ~4000 RPM vs. ESE_lv ~0.5 mmHg/mL).
        pred_norm = mean_norm.numpy()[:, :-1]  # drop Pump Level column, (T, 11)

        all_pred.append(pred_norm)
        all_true.append(y_norm)
        print(f"[{progress + 1}/{n}] example done (test idx {i})")

    pred = torch.tensor(np.stack(all_pred), dtype=torch.float32)  # (n, T, 11)
    true = torch.tensor(np.stack(all_true), dtype=torch.float32)  # (n, T, 11)

    mae = (pred - true).abs().mean(dim=(0, 1))          # (11,)
    mse = (pred - true).pow(2).mean(dim=(0, 1))          # (11,)
    # pearsoncor/dtw reduce over the whole (B, S) grid with "sum"/"mean" -- call
    # per-channel (S=1 slice) to keep the per-variable breakdown, averaging each
    # channel's correlation/DTW across all n samples.
    corr = torch.tensor([
        pearsoncor(pred[:, :, j:j + 1], true[:, :, j:j + 1], reduction="mean").item()
        for j in range(pred.shape[-1])
    ])
    dtw_dist = torch.tensor([
        dtw(pred[:, :, j:j + 1], true[:, :, j:j + 1], reduction="mean").item()
        for j in range(pred.shape[-1])
    ])

    print("\n=== Per-variable MAE / MSE / DTW / Pearson corr (normalized/z-score units, averaged over all samples) ===")
    for j in range(model.num_channels - 1):
        print(f"  {FEATURE_NAMES[j]:<14} MAE={mae[j]:7.4f}  MSE={mse[j]:7.4f}  DTW={dtw_dist[j]:7.4f}  corr={corr[j]:7.4f}")
    print(
        f"\nOverall (mean across all 11 variables): "
        f"MAE={mae.mean():.4f}  MSE={mse.mean():.4f}  DTW={dtw_dist.mean():.4f}  corr={corr.mean():.4f}"
    )
    print(f"Fallback rate: {n_fallback}/{n} examples had all samples unparseable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str,
                        default="forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth")
    parser.add_argument("--model_id", type=str, default="m42-health/Llama3-Med42-8B")
    parser.add_argument("--data_path", type=str, default="/public/gormpo/10min_1hr_all_data.pkl")
    parser.add_argument("--k_shot", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--num_examples", type=int, default=20)
    parser.add_argument("--random_sample", action="store_true", help="draw num_examples uniformly at random instead of the first N")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lora_adapter_path", type=str, default=None,
                         help="Optional PEFT adapter dir (from export_gormpo_llm_adapter.py) to merge into the backbone before generation.")
    args = parser.parse_args()
    main(args)
