"""
Zero-shot TimesFM 3.0 (google/timesfm-3.0-pytorch) on the MCS test set.

Setup (same framing as the Chronos-2 runs):
- 11 channels as the joint multivariate target, 6-step history.
- Pump Level as past+future covariate (6 history + 6 planned values).
- Horizon 6 (the model decodes one 64-step output patch; the first 6 steps are kept).

Normalization: inputs are z-scored with the TRAIN mean/std of the pkl (all 11 channels + Pump
Level) beforehand, and TimesFM's own normalization is switched OFF:
- per-series RevIN running stats -> mu 0, sigma 1 (util.get_running_stats patched)
- iterative CPM RevIN refinement off (model.use_iterative_cpm_revin = False)
- linear detrending of the context off (model.use_linear_detrending = False)
- use_znorm=False
make_positive=False (it would clip z-scored forecasts at 0). --native_norm restores all of
TimesFM's normalization for comparison.

Saves pred/true (N, 6, 11) z-score npz (same format as the other models) and, with
--probe, prints sample 1240's PumpSpeed forecast under different Pump Level plans.

Run (timesfm3 env): python eval_timesfm3.py --padding_mode none --output_npz logs/timesfm3_padnone_test_preds.npz
"""
import argparse
import pickle

import numpy as np
import torch

from timesfm3.torch import util as tfm_util
from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster

COLS = [i for i in range(13) if i != 11]

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
p.add_argument("--mcs_pickle_path", default="/public/gormpo/10min_1hr_all_data.pkl")
p.add_argument("--padding_mode", choices=["none", "edge"], default="none")
p.add_argument("--no_symmetric_averaging", action="store_true")
p.add_argument("--native_norm", action="store_true", help="keep TimesFM's own normalization")
p.add_argument("--raw_input", action="store_true",
               help="feed raw (un-z-scored) values; forecasts are z-scored afterwards for the npz")
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--output_npz", default="")
p.add_argument("--probe", action="store_true", help="only run the sample-1240 Pump Level probe")
args = p.parse_args()

if not args.native_norm:
    _orig_stats = tfm_util.get_running_stats

    def _identity_stats(values, masks, **kw):
        n, mu, sigma = _orig_stats(values, masks, **kw)
        return n, torch.zeros_like(mu), torch.ones_like(sigma)

    tfm_util.get_running_stats = _identity_stats

d = pickle.load(open(args.mcs_pickle_path, "rb"))
mean = np.asarray(d["mean"], dtype=np.float32)[COLS]
std = np.asarray(d["std"], dtype=np.float32)[COLS]
xr = np.asarray(d["test"], dtype=np.float32)[:, :, COLS]
x = ((xr - mean) / std).astype(np.float32)  # (E,12,12)

fc = TimesFM3Forecaster.from_pretrained(args.checkpoint, device="cuda:0", per_core_batch_size=args.batch_size)
if not args.native_norm:
    fc.model.use_iterative_cpm_revin = False
    fc.model.use_linear_detrending = False
print(f"RevIN identity={not args.native_norm}  cpm_refine={fc.model.use_iterative_cpm_revin}  "
      f"detrend={fc.model.use_linear_detrending}  padding_mode={args.padding_mode}  "
      f"symmetric_avg={not args.no_symmetric_averaging}")


def run(eps, plans=None):
    ctx = [ep[:6, :11].T.copy() for ep in eps]                                   # (11, 6)
    pf = [np.concatenate([ep[:6, 11], ep[6:, 11] if plans is None else plans[i]])[None, :].astype(np.float32)
          for i, ep in enumerate(eps)]                                           # (1, 12)
    outs = fc.predict_batch(contexts=ctx, horizon=6, past_future_covariates=pf, return_quantiles=False,
                            use_symmetric_averaging=not args.no_symmetric_averaging, make_positive=False,
                            use_znorm=False, padding_mode=args.padding_mode)
    return np.stack([o.forecast for o in outs]).transpose(0, 2, 1).astype(np.float32)  # (E, 6, 11)


if args.probe:
    ep = x[1240]
    z = lambda pl: ((np.asarray(pl, np.float32) - mean[11]) / std[11])
    for name, plan in [("true plan", ep[6:, 11]), ("all P2", z([2] * 6)), ("all P9", z([9] * 6))]:
        f = run([ep], [plan])[0]
        print(f"{name:<10} plan {(plan * std[11] + mean[11]).round(1)}  PumpSpeed {(f[:, 1] * std[1] + mean[1]).round()}"
              f"  PumpCurrent {(f[:, 8] * std[8] + mean[8]).round()}")
    print(f"truth      PumpSpeed {(ep[6:, 1] * std[1] + mean[1]).round()}")
else:
    pred = run(list(xr if args.raw_input else x))
    if args.raw_input:
        pred = ((pred - mean[:11]) / std[:11]).astype(np.float32)
    true = x[:, 6:, :11]
    np.savez(args.output_npz, pred=pred, true=true, n=len(pred), n_fallback=0)
    print(f"saved {args.output_npz} {pred.shape}; MAE(z) {np.abs(pred - true).mean():.4f}")
