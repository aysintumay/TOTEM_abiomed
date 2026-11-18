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

BEFORE_FALLBACK_MARKER = "[GormpoFewShot] All"


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
    if args.num_shards > 1:
        # striped, not contiguous: balances any position-correlated timing/difficulty
        # variance evenly across shards rather than handing one shard all the "hard" end.
        indices = indices[args.shard_id :: args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(indices)} of {n} total episodes")
        n = len(indices)
    n_fallback = 0
    all_pred, all_true = [], []

    for chunk_start in range(0, n, args.batch_size):
        chunk = indices[chunk_start : chunk_start + args.batch_size]
        contexts, p_levels, y_norms = [], [], []
        for i in chunk:
            x, pl, y = model.data_test[i]
            contexts.append(torch.tensor(np.asarray(x), dtype=torch.float32))
            # pl is the true per-step P-level trajectory over the forecast horizon (the
            # action), not a single value -- previously collapsed via .mean() before ever
            # reaching the prompt, discarding any real within-window P-level change.
            pl_norm = np.asarray(pl, dtype=np.float32).reshape(-1)
            p_levels.append(np.round(pl_norm * model.std[-1].item() + model.mean[-1].item()).astype(int))
            # y is already z-score normalized (TimeSeriesDataset's own output, same mean/std
            # as context_norm) -- (66,) = (forecast_horizon, 11), no Pump Level column. pl is
            # the real realized P-level trajectory for this same window (the action IS the
            # ground truth here, by construction) -- append it as the 12th column, in the
            # same normalized space, since the model now predicts Pump Level like any other
            # channel and its accuracy against the real trajectory is a reportable metric.
            y_norm_11 = np.asarray(y, dtype=np.float32).reshape(model.forecast_horizon, model.num_channels - 1)
            y_norms.append(np.concatenate([y_norm_11, pl_norm.reshape(model.forecast_horizon, 1)], axis=1))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            batch_results = model.predict_batch(contexts, p_levels)
        printed = buf.getvalue()
        print(printed, end="")
        n_fallback += printed.count(BEFORE_FALLBACK_MARKER)

        # Per-feature calls mean 12x more generate() calls per episode than before, each
        # on a slightly different padded sequence length (the token text's digit count
        # varies). Observed in practice: GPU memory allocated by PyTorch grows batch over
        # batch across a long run and eventually OOMs outright (not just reserved-cache
        # fragmentation) at the same ~384-episode mark on two independent runs -- clearing
        # the allocator's cache after every batch keeps usage flat instead of climbing.
        torch.cuda.empty_cache()

        # mean_norm is already normalized (that's predict()'s own output space); compare
        # directly, no unnorm -- errors here are in the same z-score units for every
        # channel, so they're comparable across channels of wildly different physical
        # scale (e.g. PumpSpeed ~4000 RPM vs. ESE_lv ~0.5 mmHg/mL).
        for (mean_norm, std_norm), y_norm in zip(batch_results, y_norms):
            all_pred.append(mean_norm.numpy())  # (T, 12) -- Pump Level is now a genuine prediction, not dropped
            all_true.append(y_norm)

        print(f"[{min(chunk_start + args.batch_size, n)}/{n}] examples done (batch of {len(chunk)})")

    pred = torch.tensor(np.stack(all_pred), dtype=torch.float32)  # (n, T, 12)
    true = torch.tensor(np.stack(all_true), dtype=torch.float32)  # (n, T, 12)

    if args.output_npz:
        # raw per-episode predictions, not pre-averaged metrics: merge_shards.py
        # concatenates these across shards and computes metrics ONCE over the full
        # combined set.
        np.savez(args.output_npz, pred=pred.numpy(), true=true.numpy(), n_fallback=n_fallback, n=n)
        print(f"Saved shard predictions to {args.output_npz}")

    mae = (pred - true).abs().mean(dim=(0, 1))          # (12,)
    mse = (pred - true).pow(2).mean(dim=(0, 1))          # (12,)
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
    for j in range(model.num_channels):
        print(f"  {FEATURE_NAMES[j]:<14} MAE={mae[j]:7.4f}  MSE={mse[j]:7.4f}  DTW={dtw_dist[j]:7.4f}  corr={corr[j]:7.4f}")
    print(
        f"\nOverall (mean across all 12 variables, incl. Pump Level): "
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
    parser.add_argument("--batch_size", type=int, default=8, help="episodes packed into one generate() call")
    parser.add_argument("--random_sample", action="store_true", help="draw num_examples uniformly at random instead of the first N")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lora_adapter_path", type=str, default=None,
                         help="Optional PEFT adapter dir (from export_gormpo_llm_adapter.py) to merge into the backbone before generation.")
    parser.add_argument("--shard_id", type=int, default=0, help="this process's shard index, 0-based")
    parser.add_argument("--num_shards", type=int, default=1, help="total shards splitting --num_examples across parallel processes")
    parser.add_argument("--output_npz", type=str, default="", help="save raw (pred, true) arrays here for merge_shards.py")
    args = parser.parse_args()
    main(args)
