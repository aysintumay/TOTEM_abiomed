"""
Compare LLM prompt-context variants on the SAME subset of MCS test episodes.

Subset = the `indices` saved in the variant npz files (eval_history_prompt_world_model.py
--random_sample). Reference rows (no context, original mcs.md context, carry-forward,
Chronos-2 zero-shot) are taken from their existing full-test-set runs, sliced to those
indices -- greedy decoding is deterministic, so this matches rerunning them on the subset
(up to batch-padding numerics).

Run: python compare_context_ablation.py
"""
import argparse
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, "forecasting")
from lib.models.metrics import dtw, pearsoncor  # noqa: E402

NAMES = ["PumpPressure", "PumpSpeed", "PumpFlow", "LVP", "LVEDP", "SYSTOLIC",
         "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate", "ESE_lv"]


def metrics(pred, true):
    pred, true = torch.tensor(pred), torch.tensor(true)
    mae = (pred - true).abs().mean(dim=(0, 1))
    mse = (pred - true).pow(2).mean(dim=(0, 1))
    corr = torch.tensor([pearsoncor(pred[:, :, j:j + 1], true[:, :, j:j + 1], reduction="mean").item() for j in range(11)])
    d = torch.tensor([dtw(pred[:, :, j:j + 1], true[:, :, j:j + 1], reduction="mean").item() for j in range(11)])
    return mae, mse, d, corr


def full_shards(paths, n_total=3876):
    pred = np.zeros((n_total, 6, 11), np.float32)
    true = np.zeros_like(pred)
    for k, p in enumerate(paths):
        z = np.load(p)
        pred[k::len(paths)], true[k::len(paths)] = z["pred"], z["true"]
    return pred, true


def main(args):
    variants = [(n, np.load(p)) for n, p in zip(args.variant_names, args.variants)]
    idx = variants[0][1]["indices"]
    for n, z in variants:
        assert np.array_equal(z["indices"], idx), f"{n} used different episodes"

    d = pickle.load(open(args.mcs_pickle_path, "rb"))
    cols = [i for i in range(13) if i != 11]
    x = ((np.asarray(d["test"]) - np.asarray(d["mean"])) / np.asarray(d["std"]))[:, :, cols].astype(np.float32)
    carry = np.repeat(x[:, 5:6, :11], 6, axis=1)

    nomcs = np.load(args.nomcs_npz)
    mcs_pred, mcs_true = full_shards(args.mcs_shards)
    chronos = np.load(args.chronos_npz)
    true = nomcs["true"][idx]
    rows = [
        ("Carry-forward", carry[idx]),
        ("Chronos-2 zero-shot", chronos["pred"][idx]),
        ("MedLlama, no context", nomcs["pred"][idx]),
        ("MedLlama, original mcs.md", mcs_pred[idx]),
    ] + [(f"MedLlama, {n}", z["pred"]) for n, z in variants]
    for n, z in variants:
        assert np.allclose(z["true"], true, atol=1e-4), f"{n} ground truth mismatch"
    assert np.allclose(mcs_true[idx], true, atol=1e-4)

    res = {n: metrics(p, true) for n, p in rows}
    print(f"\n{len(idx)} test episodes (random, seed {args.seed_note}); z-score units, mean over 11 variables")
    print(f"  {'model':<44}{'MAE':>8}{'MSE':>8}{'DTW':>8}{'corr':>8}")
    for n, (mae, mse, dd, c) in res.items():
        print(f"  {n:<44}{mae.mean():8.4f}{mse.mean():8.4f}{dd.mean():8.4f}{c.mean():8.4f}")
    base = res["MedLlama, no context"][0]
    print("\nPer-variable MAE (lower is better); Δ vs no context in brackets")
    print(f"  {'variable':<13}" + "".join(f"{n.replace('MedLlama, ', '')[:16]:>24}" for n, _ in rows[2:]))
    for j, v in enumerate(NAMES):
        cells = []
        for n, _ in rows[2:]:
            a = res[n][0][j].item()
            cells.append(f"{a:.4f} ({a - base[j].item():+.4f})")
        print(f"  {v:<13}" + "".join(f"{c:>24}" for c in cells))
    for n, z in variants:
        print(f"unparseable (episode, variable) forecasts in {n}: {int(z['n_fallback'])}/{len(idx) * 11}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+",
                   default=["logs/ctxablate_full.npz", "logs/ctxablate_persistence.npz", "logs/ctxablate_pleffects.npz"])
    p.add_argument("--variant_names", nargs="+",
                   default=["revised: all sections", "revised: persistence + typical change",
                            "revised: P-level effects + units"])
    p.add_argument("--nomcs_npz", default="logs/med42_history_prompt_nomcs.npz")
    p.add_argument("--mcs_shards", nargs="+",
                   default=["logs/med42_history_prompt_shard0.npz", "logs/med42_history_prompt_shard1.npz"])
    p.add_argument("--chronos_npz", default="logs/chronos2_zeroshot_test_preds.npz")
    p.add_argument("--mcs_pickle_path", default="/public/gormpo/10min_1hr_all_data.pkl")
    p.add_argument("--seed_note", default="0")
    main(p.parse_args())
