"""
Evaluate HistoryPromptWorldModel (TimesX / CiK-style direct LLM forecasting, zero-shot)
on the MCS test split. Metrics (MAE, MSE, DTW, Pearson corr) in normalized z-score
space over the 11 forecast variables -- same convention as the other eval_*_world_model.py.

The action passed to the model is the full per-step P-level trajectory of the forecast
window (mcs.md interface contract), not its mean.

Run: python eval_history_prompt_world_model.py --num_examples 20 --print_prompt
"""
import argparse

import numpy as np
import torch

from history_prompt_world_model import HistoryPromptWorldModel
from world_model import FEATURE_NAMES

import sys
sys.path.insert(0, "forecasting")
from lib.models.metrics import dtw, pearsoncor  # noqa: E402


def main(args):
    model = HistoryPromptWorldModel(
        model_id=args.model_id,
        num_features=12,
        forecast_horizon=6,
        num_samples=args.num_samples,
        temperature=args.temperature,
        mcs_context=not args.no_mcs_context,
        context_file=args.context_file or None,
        context_sections=args.context_sections,
        pl_prefix=not args.no_pl_prefix,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        max_memory=({i: f"{g}GiB" for i, g in enumerate(args.max_memory_gb)} if args.max_memory_gb else None),
    )
    model.load_data(args.data_path)

    n = min(args.num_examples, len(model.data_test))
    rng = np.random.RandomState(args.random_seed)
    indices = rng.choice(len(model.data_test), size=n, replace=False) if args.random_sample else np.arange(n)
    if args.num_shards > 1:
        indices = indices[args.shard_id :: args.num_shards]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(indices)} of {n} total episodes")
        n = len(indices)

    H, F = model.forecast_horizon, model.num_features - 1
    all_pred, all_true, n_failed_vars = [], [], 0
    for chunk_start in range(0, n, args.batch_size):
        chunk = indices[chunk_start : chunk_start + args.batch_size]
        contexts, pl_seqs, y_norms = [], [], []
        for i in chunk:
            x, pl, y = model.data_test[i]
            contexts.append(torch.tensor(np.asarray(x), dtype=torch.float32))
            pl_phys = np.asarray(pl, dtype=np.float32) * model.std[-1].item() + model.mean[-1].item()
            pl_seqs.append([int(round(p)) for p in pl_phys])
            y_norms.append(np.asarray(y, dtype=np.float32).reshape(H, F))

        if args.print_prompt and chunk_start == 0:
            ctx_phys = model._unnorm_context(contexts[0])
            print("=== SYSTEM ===\n" + model._system_prompt())
            prompt = model._build_prompt(ctx_phys, pl_seqs[0])
            print("=== USER ===\n" + prompt)
            print("=== RAW RESPONSE ===\n" + model._generate([prompt], 1)[0][0])

        for (mean_norm, _, failed), y_norm in zip(model.predict_batch(contexts, pl_seqs, args.prompt_batch_size), y_norms):
            all_pred.append(mean_norm.numpy()[:, :-1])
            all_true.append(y_norm)
            n_failed_vars += failed
        print(f"[{min(chunk_start + args.batch_size, n)}/{n}] episodes done")

    pred = torch.tensor(np.stack(all_pred))
    true = torch.tensor(np.stack(all_true))
    if args.output_npz:
        np.savez(args.output_npz, pred=pred.numpy(), true=true.numpy(), n_fallback=n_failed_vars, n=n,
                 indices=np.asarray(indices))
        print(f"Saved predictions to {args.output_npz}")

    mae = (pred - true).abs().mean(dim=(0, 1))
    mse = (pred - true).pow(2).mean(dim=(0, 1))
    corr = torch.tensor([pearsoncor(pred[:, :, j:j + 1], true[:, :, j:j + 1], reduction="mean").item() for j in range(F)])
    dtw_dist = torch.tensor([dtw(pred[:, :, j:j + 1], true[:, :, j:j + 1], reduction="mean").item() for j in range(F)])

    print("\n=== Per-variable MAE / MSE / DTW / Pearson corr (normalized/z-score units, averaged over all samples) ===")
    for j in range(F):
        print(f"  {FEATURE_NAMES[j]:<14} MAE={mae[j]:7.4f}  MSE={mse[j]:7.4f}  DTW={dtw_dist[j]:7.4f}  corr={corr[j]:7.4f}")
    print(f"\nOverall (mean across all 11 variables): "
          f"MAE={mae.mean():.4f}  MSE={mse.mean():.4f}  DTW={dtw_dist.mean():.4f}  corr={corr.mean():.4f}")
    # same predictions in raw physical units (the LLM itself only ever sees raw values)
    std11 = model.std[model.columns][:F].float()
    mean11 = model.mean[model.columns][:F].float()
    pred_raw, true_raw = pred * std11 + mean11, true * std11 + mean11
    mae_raw = (pred_raw - true_raw).abs().mean(dim=(0, 1))
    rmse_raw = (pred_raw - true_raw).pow(2).mean(dim=(0, 1)).sqrt()
    print("\n=== Per-variable MAE / RMSE in raw physical units ===")
    for j in range(F):
        print(f"  {FEATURE_NAMES[j]:<14} MAE={mae_raw[j]:9.3f}  RMSE={rmse_raw[j]:9.3f}")
    print(f"Unparseable (episode, variable) forecasts -> carry-forward: {n_failed_vars}/{n * F}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="m42-health/Llama3-Med42-8B")
    p.add_argument("--data_path", type=str, default="/public/gormpo/10min_1hr_all_data.pkl")
    p.add_argument("--num_samples", type=int, default=1, help="1 = single direct LLM forecast (no ensemble)")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--num_examples", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32, help="episodes per progress chunk")
    p.add_argument("--prompt_batch_size", type=int, default=8, help="episodes (prompts) per generate() call")
    p.add_argument("--random_sample", action="store_true")
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--max_memory_gb", type=float, nargs="+", default=None)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--output_npz", type=str, default="")
    p.add_argument("--no_mcs_context", action="store_true",
                   help="drop mcs.md variable descriptions/targets and the Context section")
    p.add_argument("--context_file", type=str, default="",
                   help="e.g. mcs_forecast_context.md: replaces the mcs.md Context section and targets")
    p.add_argument("--context_sections", type=str, nargs="+", default=None,
                   help="subset of '## ' section names from --context_file (default: all)")
    p.add_argument("--no_pl_prefix", action="store_true",
                   help='write the planned Pump Level as "(t7, 7)" instead of "(t7, P7)"')
    p.add_argument("--print_prompt", action="store_true", help="print one full prompt + raw response")
    main(p.parse_args())
