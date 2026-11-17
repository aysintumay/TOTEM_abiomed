"""
Chronos-2 zero-shot, PUMP CHANNELS ONLY: target = the 4 Pump* variables (PumpPressure,
PumpSpeed, PumpFlow, PumpCurrent), Pump Level as past covariate + the next hour's planned
Pump Level as known-future covariate. The 7 physiological channels are not given at all.
Motivation: with all 11 targets, Chronos-2 kept PumpSpeed flat on sample 1240 even though
the Pump Level dropped P7 -> P2 -- does removing the other channels let it use the covariate?

Saves pred/true as (N, 6, 11) z-score (non-pump channels NaN) so it plugs into
visualize_predictions.py, and compares on the pump channels against carry-forward,
Chronos-2 with all 11 targets, and the Transformer.

Run: PYTHONPATH=chronos-forecasting/src python eval_chronos2_pump_only.py --gpu 0
"""
import argparse
import pickle

import numpy as np
from chronos import Chronos2Pipeline

from compare_context_ablation import NAMES, metrics

PUMP = [0, 1, 2, 8]  # PumpPressure, PumpSpeed, PumpFlow, PumpCurrent

p = argparse.ArgumentParser()
p.add_argument("--model_id", default="amazon/chronos-2")
p.add_argument("--mcs_pickle_path", default="/public/gormpo/10min_1hr_all_data.pkl")
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--output_npz", default="logs/chronos2_pumponly_test_preds.npz")
p.add_argument("--chronos_full_npz", default="logs/chronos2_zeroshot_test_preds.npz")
p.add_argument("--transformer_npz", default="logs/transformer_test_preds.npz")
args = p.parse_args()

d = pickle.load(open(args.mcs_pickle_path, "rb"))
cols = [i for i in range(13) if i != 11]
test = np.asarray(d["test"], dtype=np.float32)[:, :, cols]  # (E, 12, 12) physical
m = np.asarray(d["mean"], dtype=np.float32)[cols][:11]
s = np.asarray(d["std"], dtype=np.float32)[cols][:11]

pipe = Chronos2Pipeline.from_pretrained(args.model_id, device_map=f"cuda:{args.gpu}")
q = pipe.quantiles.index(0.5)
inputs = [{"target": ep[:6, PUMP].T,
           "past_covariates": {"pump_level": ep[:6, 11]},
           "future_covariates": {"pump_level": ep[6:, 11]}} for ep in test]
out = pipe.predict(inputs, prediction_length=6, batch_size=args.batch_size)
pump_phys = np.stack([o[:, q, :].cpu().numpy() for o in out]).transpose(0, 2, 1)  # (E, 6, 4)

pred = np.full((len(test), 6, 11), np.nan, np.float32)
pred[:, :, PUMP] = (pump_phys - m[PUMP]) / s[PUMP]
true = ((test[:, 6:, :11] - m) / s).astype(np.float32)
np.savez(args.output_npz, pred=pred, true=true, n=len(pred), n_fallback=0)
print(f"saved {args.output_npz}")

ctx = (test[:, :6, :11] - m) / s
rows = {"Carry-forward": np.repeat(ctx[:, 5:6], 6, 1),
        "Chronos-2, all 11 targets": np.load(args.chronos_full_npz)["pred"],
        "Chronos-2, pump channels only": pred,
        "Transformer WorldModel": np.load(args.transformer_npz)["pred"]}
pl_changed = (test[:, 6:, 11] != test[:, 5:6, 11]).any(1)
for subset, mask in [("all test episodes", np.ones(len(test), bool)),
                     ("Pump Level changes in the next hour", pl_changed),
                     ("Pump Level unchanged", ~pl_changed)]:
    print(f"\n== {subset} (n={mask.sum()}); z-score, pump channels ==")
    print(f"  {'model':<32}" + "".join(f"{NAMES[j]:>13}" for j in PUMP) + f"{'mean MAE':>10}{'MSE':>8}{'corr':>8}")
    for n, pr in rows.items():
        mae, mse, _, c = metrics(pr[mask][:, :, PUMP] if False else np.nan_to_num(pr[mask]), true[mask])
        mae, mse, c = mae[PUMP], mse[PUMP], c[PUMP]
        print(f"  {n:<32}" + "".join(f"{mae[k]:13.4f}" for k in range(4)) +
              f"{mae.mean():10.4f}{mse.mean():8.4f}{c.mean():8.4f}")
for i in [321, 1240, 3740]:
    print(f"\nsample {i}: Pump Level next hour {test[i, 6:, 11]}; PumpSpeed true {test[i, 6:, 1].round()}")
    print(f"   pump-only  {pump_phys[i, :, 1].round()}")
    print(f"   all-11     {(rows['Chronos-2, all 11 targets'][i, :, 1] * s[1] + m[1]).round()}")
