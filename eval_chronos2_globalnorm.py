"""
Evaluate Chronos-2 under GLOBAL normalization (chronos2_global_norm.py: train mean/std
z-scoring + identity InstanceNorm) on the MCS test set; saves (N, 6, 11) z-score npz for
visualize_predictions.py and compares against the instance-norm runs.

Run: PYTHONPATH=chronos-forecasting/src python eval_chronos2_globalnorm.py \
        --model_id saved_models/chronos2_globalnorm_mcs/finetuned-ckpt --output_npz logs/x.npz
"""
import argparse

import numpy as np

import chronos2_global_norm as gn  # patches InstanceNorm on import
from chronos import Chronos2Pipeline
from compare_context_ablation import NAMES, metrics

p = argparse.ArgumentParser()
p.add_argument("--model_id", default="saved_models/chronos2_globalnorm_mcs/finetuned-ckpt")
p.add_argument("--mcs_pickle_path", default="/public/gormpo/10min_1hr_all_data.pkl")
p.add_argument("--output_npz", required=True)
p.add_argument("--extra", nargs="*", default=[], help="name=npz pairs to add to the comparison")
args = p.parse_args()

x, mean, std = gn.load_split(args.mcs_pickle_path, "test")
pipe = Chronos2Pipeline.from_pretrained(args.model_id, device_map="cuda:0")
q = pipe.quantiles.index(0.5)
out = pipe.predict(gn.predict_inputs(x), prediction_length=6, batch_size=256)
pred = np.stack([o[:, q, :].cpu().numpy() for o in out]).transpose(0, 2, 1).astype(np.float32)
true = x[:, 6:, :11]
np.savez(args.output_npz, pred=pred, true=true, n=len(pred), n_fallback=0)
print(f"saved {args.output_npz}")

rows = {"Carry-forward": np.repeat(x[:, 5:6, :11], 6, 1),
        "C2 zero-shot, instance norm": np.load("logs/chronos2_zeroshot_test_preds.npz")["pred"],
        "C2 fine-tuned, instance norm": np.load("logs/chronos2_finetuned_test_preds.npz")["pred"]}
rows.update({k: np.load(v)["pred"] for k, v in (e.split("=") for e in args.extra)})
rows[f"THIS: {args.model_id.split('/')[-2] if '/' in args.model_id else args.model_id}"] = pred
rows["Transformer"] = np.load("logs/transformer_test_preds.npz")["pred"]
P = [0, 1, 2, 8]
chg = (x[:, 6:, 11] != x[:, 5:6, 11]).any(1)
for sub, m in [("all", np.ones(len(x), bool)), ("Pump Level changes", chg), ("Pump Level unchanged", ~chg)]:
    print(f"\n== {sub} (n={m.sum()}) z-score ==")
    print(f"  {'model':<34}{'MAE':>8}{'MSE':>8}{'DTW':>8}{'corr':>8}" + "".join(f"{NAMES[j][:11]:>12}" for j in P))
    for n, pr in rows.items():
        mae, mse, dd, c = metrics(pr[m], true[m])
        print(f"  {n:<34}{mae.mean():8.4f}{mse.mean():8.4f}{dd.mean():8.4f}{c.mean():8.4f}" + "".join(f"{mae[j]:12.4f}" for j in P))
ps = pred * std[:11] + mean[:11]
for i in [321, 1240, 3740]:
    print(f"sample {i}: PL {(x[i, 6:, 11] * std[11] + mean[11]).round(1)}  PumpSpeed true {(true[i, :, 1] * std[1] + mean[1]).round()}  pred {ps[i, :, 1].round()}")
