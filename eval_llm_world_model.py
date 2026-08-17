"""
Evaluate FewShotDigitTokenWorldModel (world_model.py) on the real MCS test split using
a general-purpose (non-medical) LLM, in-context, zero fine-tuning. --k_shot 0 degenerates
cleanly to zero-shot (no retrieval index built, no few-shot messages injected -- same
convention eval_gormpo_world_model.py already uses), so this one script covers both the
zero-shot and k-shot legs of the comparison, only --k_shot differs.

Metrics (MAE, MSE, DTW, Pearson correlation) are computed in normalized (z-score)
space, same convention as every other eval_*_world_model.py script in this project.

Run with: python eval_llm_world_model.py --model_id Qwen/Qwen2.5-7B-Instruct --k_shot 0 --num_examples 20
"""
import argparse
import contextlib
import io
import sys

import numpy as np
import torch

from world_model import FewShotDigitTokenWorldModel, FEATURE_NAMES

sys.path.insert(0, "forecasting")
from lib.models.metrics import dtw, pearsoncor  # noqa: E402


def main(args):
    model = FewShotDigitTokenWorldModel(
        k_shot=args.k_shot,
        model_id=args.model_id,
        num_features=12,
        forecast_horizon=6,
        num_samples=args.num_samples,
        temperature=args.temperature,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
    )
    model.load_data(args.data_path)

    n = min(args.num_examples, len(model.data_test))
    rng = np.random.RandomState(args.random_seed)
    indices = rng.choice(len(model.data_test), size=n, replace=False) if args.random_sample else np.arange(n)
    if args.num_shards > 1:
        # striped, not contiguous: balances any position-correlated timing/difficulty
        # variance evenly across shards rather than handing one shard all the "hard" end.
        indices = indices[args.shard_id :: args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(indices)} of {n} total episodes")
        n = len(indices)
    n_fallback = 0
    all_pred, all_true = [], []
    before_fallback_marker = "] All"

    for chunk_start in range(0, n, args.batch_size):
        chunk = indices[chunk_start : chunk_start + args.batch_size]
        contexts, p_levels, y_norms = [], [], []
        for i in chunk:
            x, pl, y = model.data_test[i]
            contexts.append(torch.tensor(np.asarray(x), dtype=torch.float32))
            pl_raw = float(np.asarray(pl).mean())
            p_levels.append(int(round(pl_raw * model.std[-1].item() + model.mean[-1].item())))
            y_norms.append(np.asarray(y, dtype=np.float32).reshape(model.forecast_horizon, model.num_features - 1))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            batch_results = model.predict_batch(contexts, p_levels)
        printed = buf.getvalue()
        print(printed, end="")
        n_fallback += printed.count(before_fallback_marker)

        for (mean_norm, std_norm), y_norm in zip(batch_results, y_norms):
            all_pred.append(mean_norm.numpy()[:, :-1])  # drop Pump Level column, (T, 11)
            all_true.append(y_norm)

        print(f"[{min(chunk_start + args.batch_size, n)}/{n}] examples done (batch of {len(chunk)})")

    pred = torch.tensor(np.stack(all_pred), dtype=torch.float32)  # (n, T, 11)
    true = torch.tensor(np.stack(all_true), dtype=torch.float32)  # (n, T, 11)

    if args.output_npz:
        # raw per-episode predictions, not pre-averaged metrics: merge_shards.py
        # concatenates these across shards and computes metrics ONCE over the full
        # combined set, so DTW/corr (not simple means) are exact, not a weighted
        # average-of-shard-averages approximation.
        np.savez(args.output_npz, pred=pred.numpy(), true=true.numpy(), n_fallback=n_fallback, n=n)
        print(f"Saved shard predictions to {args.output_npz}")

    mae = (pred - true).abs().mean(dim=(0, 1))
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
    for j in range(model.num_features - 1):
        print(f"  {FEATURE_NAMES[j]:<14} MAE={mae[j]:7.4f}  MSE={mse[j]:7.4f}  DTW={dtw_dist[j]:7.4f}  corr={corr[j]:7.4f}")
    print(
        f"\nOverall (mean across all 11 variables): "
        f"MAE={mae.mean():.4f}  MSE={mse.mean():.4f}  DTW={dtw_dist.mean():.4f}  corr={corr.mean():.4f}"
    )
    print(f"Fallback rate: {n_fallback}/{n} examples had all samples unparseable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str, default="/public/gormpo/10min_1hr_all_data.pkl")
    parser.add_argument("--k_shot", type=int, default=0, help="0 = zero-shot, >0 = k-shot in-context retrieval")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num_examples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8, help="episodes packed into one generate() call")
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--shard_id", type=int, default=0, help="this process's shard index, 0-based")
    parser.add_argument("--num_shards", type=int, default=1, help="total shards splitting --num_examples across parallel processes")
    parser.add_argument("--output_npz", type=str, default="", help="save raw (pred, true) arrays here for merge_shards.py")
    args = parser.parse_args()
    main(args)
