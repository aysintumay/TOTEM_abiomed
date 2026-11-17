"""
Save fine-tuned Chronos-2 (saved_models/chronos2_mcs/finetuned-ckpt) test-set forecasts as
(N, 6, 11) z-score npz for visualize_predictions.py. Same setup as
eval_chronos2_world_model.py: 11 channels as joint target, Pump Level as past + known-future
covariate, median quantile. Also prints pump-channel MAE split by whether Pump Level changes.

Run: PYTHONPATH=chronos-forecasting/src python dump_chronos2_finetuned_preds.py
"""
import argparse
import pickle

import numpy as np
from chronos import Chronos2Pipeline

from compare_context_ablation import NAMES, metrics

p = argparse.ArgumentParser()
p.add_argument("--model_id", default="saved_models/chronos2_mcs/finetuned-ckpt")
p.add_argument("--mcs_pickle_path", default="/public/gormpo/10min_1hr_all_data.pkl")
p.add_argument("--output_npz", default="logs/chronos2_finetuned_test_preds.npz")
args = p.parse_args()

d = pickle.load(open(args.mcs_pickle_path, "rb"))
cols = [i for i in range(13) if i != 11]
test = np.asarray(d["test"], dtype=np.float32)[:, :, cols]
m = np.asarray(d["mean"], dtype=np.float32)[cols][:11]
s = np.asarray(d["std"], dtype=np.float32)[cols][:11]

pipe = Chronos2Pipeline.from_pretrained(args.model_id, device_map="cuda:0")
q = pipe.quantiles.index(0.5)
out = pipe.predict([{"target": ep[:6, :11].T, "past_covariates": {"pump_level": ep[:6, 11]},
                     "future_covariates": {"pump_level": ep[6:, 11]}} for ep in test],
                   prediction_length=6, batch_size=256)
phys = np.stack([o[:, q, :].cpu().numpy() for o in out]).transpose(0, 2, 1)
pred = ((phys - m) / s).astype(np.float32)
true = ((test[:, 6:, :11] - m) / s).astype(np.float32)
np.savez(args.output_npz, pred=pred, true=true, n=len(pred), n_fallback=0)
print(f"saved {args.output_npz}")

ctx = (test[:, :6, :11] - m) / s
rows = {"Carry-forward": np.repeat(ctx[:, 5:6], 6, 1),
        "Chronos-2 zero-shot": np.load("logs/chronos2_zeroshot_test_preds.npz")["pred"],
        "Chronos-2 fine-tuned": pred,
        "Transformer WorldModel": np.load("logs/transformer_test_preds.npz")["pred"]}
chg = (test[:, 6:, 11] != test[:, 5:6, 11]).any(1)
for sub, mask in [("all", np.ones(len(test), bool)), ("Pump Level changes", chg), ("Pump Level unchanged", ~chg)]:
    print(f"\n== {sub} (n={mask.sum()}) z-score MAE ==")
    print(f"  {'model':<24}{'all 11':>8}" + "".join(f"{NAMES[j]:>13}" for j in [0, 1, 2, 8]))
    for n, pr in rows.items():
        mae = metrics(pr[mask], true[mask])[0]
        print(f"  {n:<24}{mae.mean():8.4f}" + "".join(f"{mae[j]:13.4f}" for j in [0, 1, 2, 8]))
for i in [321, 1240, 3740]:
    print(f"sample {i}: PL {test[i, 6:, 11]}  PumpSpeed true {test[i, 6:, 1].round()}  fine-tuned {phys[i, :, 1].round()}")
