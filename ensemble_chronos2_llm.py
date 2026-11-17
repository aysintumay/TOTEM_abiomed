"""
Equal-weight ensemble of a Chronos-2 forecast and a direct-LLM forecast (TimesX-style,
arXiv 2607.06973): ensemble = 0.5 * Chronos-2 + 0.5 * LLM, per variable and timestep.

- Chronos-2: run here, same setup as eval_chronos2_world_model.py (11 physiological
  channels as joint multivariate target, Pump Level as past + known-future covariate,
  point forecast = median quantile).
- LLM: loaded from an eval_history_prompt_world_model.py --output_npz file
  (e.g. MedLlama, no mcs.md context).

The two pipelines read the test split from different sources (data_raw vs. the pkl), so
the script first checks that both ground-truth arrays agree episode-by-episode and
refuses to ensemble if they don't.

Metrics: MAE / MSE / DTW / Pearson corr in normalized (z-score) space, 11 variables,
same convention as the other eval_* scripts.

Run: PYTHONPATH=chronos-forecasting/src python ensemble_chronos2_llm.py \
        --llm_npz logs/med42_history_prompt_nomcs.npz --model_id amazon/chronos-2
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

from chronos import Chronos2Pipeline

sys.path.insert(0, "forecasting")
from train_tokenizer import load_mcs_episode_halves  # noqa: E402
from lib.models.metrics import dtw, pearsoncor  # noqa: E402

FEATURE_NAMES = [
    "PumpPressure", "PumpSpeed", "PumpFlow", "LVP", "LVEDP", "SYSTOLIC",
    "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate", "ESE_lv",
]


def chronos2_predict(args, mean, std):
    """-> pred, true: (n, T, 11) normalized, time-major."""
    if os.path.exists(args.chronos_cache):
        d = np.load(args.chronos_cache)
        print(f"Loaded cached Chronos-2 predictions from {args.chronos_cache}")
        return d["pred"], d["true"]
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=device)
    q_idx = pipeline.quantiles.index(0.5)
    x_half, y_half = load_mcs_episode_halves(args.data_root, "test", args.patch_len)
    x_half, y_half = x_half.numpy(), y_half.numpy()  # (E, 12, T) physical
    preds, trues = [], []
    for start in range(0, x_half.shape[0], args.batch_size):
        idx = range(start, min(start + args.batch_size, x_half.shape[0]))
        inputs = [
            {
                "target": x_half[i, 0:11, :],
                "past_covariates": {"pump_level": x_half[i, 11, :]},
                "future_covariates": {"pump_level": y_half[i, 11, :]},
            }
            for i in idx
        ]
        out = pipeline.predict(inputs, prediction_length=args.patch_len, batch_size=args.batch_size)
        for p, i in zip(out, idx):
            preds.append(p[:, q_idx, :].cpu().numpy())
            trues.append(y_half[i, 0:11, :])
    m, s = mean.reshape(1, 11, 1), std.reshape(1, 11, 1)
    pred = ((np.stack(preds) - m) / s).transpose(0, 2, 1).astype(np.float32)
    true = ((np.stack(trues) - m) / s).transpose(0, 2, 1).astype(np.float32)
    np.savez(args.chronos_cache, pred=pred, true=true)
    print(f"Saved Chronos-2 predictions to {args.chronos_cache}")
    return pred, true


def report(name, pred, true):
    pred, true = torch.tensor(pred), torch.tensor(true)
    mae = (pred - true).abs().mean(dim=(0, 1))
    mse = (pred - true).pow(2).mean(dim=(0, 1))
    F = pred.shape[-1]
    corr = torch.tensor([pearsoncor(pred[:, :, j:j + 1], true[:, :, j:j + 1], reduction="mean").item() for j in range(F)])
    dtw_d = torch.tensor([dtw(pred[:, :, j:j + 1], true[:, :, j:j + 1], reduction="mean").item() for j in range(F)])
    print(f"\n=== {name}: per-variable MAE / MSE / DTW / Pearson corr (z-score units) ===")
    for j in range(F):
        print(f"  {FEATURE_NAMES[j]:<14} MAE={mae[j]:7.4f}  MSE={mse[j]:7.4f}  DTW={dtw_d[j]:7.4f}  corr={corr[j]:7.4f}")
    print(f"Overall: MAE={mae.mean():.4f}  MSE={mse.mean():.4f}  DTW={dtw_d.mean():.4f}  corr={corr.mean():.4f}")
    return mae.mean().item(), mse.mean().item(), dtw_d.mean().item(), corr.mean().item()


def main(args):
    with open(args.mcs_pickle_path, "rb") as f:
        d = pickle.load(f)
    cols = [i for i in range(13) if i != 11]
    mean = np.asarray(d["mean"], dtype=np.float32)[cols][:11]
    std = np.asarray(d["std"], dtype=np.float32)[cols][:11]

    llm = np.load(args.llm_npz)
    llm_pred, llm_true = llm["pred"].astype(np.float32), llm["true"].astype(np.float32)
    c2_pred, c2_true = chronos2_predict(args, mean, std)

    # ── alignment check: both pipelines must be scoring the same episodes ──
    assert llm_true.shape == c2_true.shape, f"shape mismatch {llm_true.shape} vs {c2_true.shape}"
    diff = np.abs(llm_true - c2_true)
    print(f"Ground-truth alignment: max |diff| = {diff.max():.2e}, mean = {diff.mean():.2e} (z-score units)")
    assert diff.max() < 1e-3, "LLM and Chronos-2 test episodes are not aligned -- refusing to ensemble"

    ens = args.w_llm * llm_pred + (1 - args.w_llm) * c2_pred
    rows = [
        ("Chronos-2", report(f"Chronos-2 ({args.model_id})", c2_pred, c2_true)),
        ("LLM", report(f"LLM ({args.llm_npz})", llm_pred, llm_true)),
        (f"Ensemble {args.w_llm:.1f}*LLM + {1 - args.w_llm:.1f}*Chronos-2",
         report("Ensemble", ens, llm_true)),
    ]
    print("\n=== Summary (overall, z-score units) ===")
    print(f"  {'model':<40} {'MAE':>7} {'MSE':>7} {'DTW':>7} {'corr':>7}")
    for name, (a, b, c, e) in rows:
        print(f"  {name:<40} {a:7.4f} {b:7.4f} {c:7.4f} {e:7.4f}")
    if args.output_npz:
        np.savez(args.output_npz, pred=ens, true=llm_true, n=len(ens), n_fallback=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--llm_npz", type=str, default="logs/med42_history_prompt_nomcs.npz")
    p.add_argument("--model_id", type=str, default="amazon/chronos-2",
                   help="Chronos-2 HF id or fine-tuned checkpoint dir")
    p.add_argument("--chronos_cache", type=str, default="logs/chronos2_zeroshot_test_preds.npz",
                   help="where Chronos-2 predictions are cached (reused if it exists)")
    p.add_argument("--w_llm", type=float, default=0.5, help="LLM weight; Chronos-2 gets 1 - w_llm")
    p.add_argument("--data_root", type=str, default="forecasting/data_raw/abiomed")
    p.add_argument("--mcs_pickle_path", type=str, default="/public/gormpo/10min_1hr_all_data.pkl")
    p.add_argument("--patch_len", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_npz", type=str, default="")
    main(p.parse_args())
