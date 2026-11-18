"""
Fine-tune Chronos-2 on MCS with GLOBAL normalization (see chronos2_global_norm.py) instead
of Chronos-2's per-window instance normalization. Otherwise the same framing as
train_chronos2.py: 11 channels joint target, Pump Level known-future covariate, each
12-step episode one item with the cut fixed at step 6 (min_past=6); best checkpoint on val
loss is kept (Chronos2Pipeline.fit's load_best_model_at_end).

Run: PYTHONPATH=chronos-forecasting/src python train_chronos2_globalnorm.py --lr 3e-5 \
        --save_path saved_models/chronos2_globalnorm_lr3e-5
"""
import argparse
import os

import numpy as np

import chronos2_global_norm as gn  # patches InstanceNorm on import
from chronos import Chronos2Pipeline
from chronos.chronos2 import preprocess


def items(x):
    return [{"target": ep[:, :11].T, "past_covariates": {"pump_level": ep[:, 11]}} for ep in x]


p = argparse.ArgumentParser()
p.add_argument("--model_id", default="amazon/chronos-2")
p.add_argument("--mcs_pickle_path", default="/public/gormpo/10min_1hr_all_data.pkl")
p.add_argument("--save_path", required=True)
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--num_steps", type=int, default=3000)
p.add_argument("--batch_size", type=int, default=256)
args = p.parse_args()

tr, _, _ = gn.load_split(args.mcs_pickle_path, "train")
va, _, _ = gn.load_split(args.mcs_pickle_path, "val")
print(f"train {len(tr)}  val {len(va)}  (globally z-scored)")
kw = dict(prediction_length=6, known_covariates_names=["pump_level"])
pipe = Chronos2Pipeline.from_pretrained(args.model_id, device_map="cuda:0")
os.makedirs(args.save_path, exist_ok=True)
pipe.fit(preprocess.from_list_of_dicts(items(tr), **kw), prediction_length=6,
         validation_inputs=preprocess.from_list_of_dicts(items(va), **kw),
         finetune_mode="full", context_length=6, learning_rate=args.lr, num_steps=args.num_steps,
         batch_size=args.batch_size, min_past=6, output_dir=args.save_path,
         finetuned_ckpt_name="finetuned-ckpt", save_total_limit=2)
print("done")
