"""
Fine-tune TimesFM 3.0 (google/timesfm-3.0-pytorch) on the MCS train split with GLOBAL
normalization, mirroring train_chronos2_globalnorm.py.

- Inputs are z-scored with the train mean/std of the pkl (11 channels + Pump Level).
- TimesFM's own normalization is switched off, for training AND inference:
  per-series RevIN running stats -> mu 0, sigma 1 (util.get_running_stats patched),
  iterative CPM RevIN refinement off, linear detrending off. The two model flags are saved
  in the checkpoint's config.json; eval_timesfm3.py (without --native_norm) re-applies the
  RevIN patch when loading it.
- Same framing as the zero-shot eval: 11 channels as the joint target, 6-step history,
  Pump Level as past+future covariate (6 history + 6 planned), horizon 6, unknown future
  covariate steps masked (padding_mode "none").
- Training runs through the model's own decode() (the inference path, gradients enabled),
  so train and test see identical input construction.
- Hyperparameters copied from Chronos2Pipeline.fit as used for the Chronos-2 global-norm
  fine-tune: AdamW (fused, no weight decay), lr 1e-6 linear decay, no warmup, grad-norm clip
  1.0, 1000 steps, batch 256, bf16 autocast, seed 42, quantile (pinball) loss over the
  model's 9 quantiles.

Run (timesfm3 env): CUDA_VISIBLE_DEVICES=0 python train_timesfm3_globalnorm.py \
        --save_path saved_models/timesfm3_globalnorm_mcs
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch

from timesfm3.torch import util as tfm_util
from timesfm3.torch.model import TimesFM3Torch

COLS = [i for i in range(13) if i != 11]

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
p.add_argument("--mcs_pickle_path", default="/public/gormpo/10min_1hr_all_data.pkl")
p.add_argument("--lr", type=float, default=1e-6)
p.add_argument("--num_steps", type=int, default=1000)
p.add_argument("--batch_size", type=int, default=256)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--micro_batch", type=int, default=64,
               help="split each batch into micro-batches (gradient accumulation; same effective batch)")
p.add_argument("--save_path", default="saved_models/timesfm3_globalnorm_mcs")
args = p.parse_args()

_orig_stats = tfm_util.get_running_stats


def _identity_stats(values, masks, **kw):
    n, mu, sigma = _orig_stats(values, masks, **kw)
    return n, torch.zeros_like(mu), torch.ones_like(sigma)


tfm_util.get_running_stats = _identity_stats

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
d = pickle.load(open(args.mcs_pickle_path, "rb"))
mean = np.asarray(d["mean"], dtype=np.float32)[COLS]
std = np.asarray(d["std"], dtype=np.float32)[COLS]
x = torch.from_numpy(((np.asarray(d["train"], dtype=np.float32)[:, :, COLS] - mean) / std).astype(np.float32))
print(f"train episodes {len(x)}")

dev = torch.device("cuda:0")
model = TimesFM3Torch.from_pretrained(args.checkpoint).to(dev)
model.use_iterative_cpm_revin = False
model.use_linear_detrending = False
model.train()
q = torch.tensor(model.quantiles, device=dev)
decode = TimesFM3Torch.decode.__wrapped__  # decode() without its @torch.no_grad

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, fused=True)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / args.num_steps)

t0, run = time.time(), []
for step in range(args.num_steps):
    batch = x[rng.integers(0, len(x), args.batch_size)].to(dev)            # (B, 12, 12)
    opt.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for b in batch.split(args.micro_batch):
        target = b[:, :6, :11].transpose(1, 2)                             # (b, 11, 6)
        pf = b[:, :, 11][:, None, :]                                       # (b, 1, 12)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = decode(model, target=target, past_future_covariates=pf)  # (b, 12, 6, 9)
        pred = out[:, :11].float()
        y = b[:, 6:, :11].transpose(1, 2)[..., None]                       # (b, 11, 6, 1)
        loss = (2 * torch.abs((y - pred) * ((y <= pred).float() - q))).mean() * len(b) / len(batch)
        loss.backward()
        loss_sum += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    run.append(loss_sum)
    if (step + 1) % 100 == 0:
        print(f"step {step + 1:5d}  loss {np.mean(run):.4f}  lr {sched.get_last_lr()[0]:.2e}  "
              f"{time.time() - t0:.0f}s", flush=True)
        run = []

os.makedirs(args.save_path, exist_ok=True)
model.save_pretrained(args.save_path)
print(f"saved {args.save_path}")
