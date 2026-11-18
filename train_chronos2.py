"""
Fine-tune Chronos-2 (amazon/chronos-2) on MCS data using its native covariate-
conditioned, multivariate forecasting interface (Chronos2Pipeline.fit()) -- no GORMPO
tokenizer involved; Chronos-2 has no discrete tokenizer to swap (see mcs.md).

Same task framing as eval_chronos2_world_model.py: the 11 physiological channels are
the joint multivariate target; Pump Level (the action) is a known-future covariate,
not a forecast target.

Training-time data shape, per Chronos2Dataset's TRAIN-mode slicing contract (see
chronos2/dataset.py's Chronos2Dataset._construct_slice): each episode's FULL
context+horizon window (x_half + y_half concatenated along time, patch_len*2 steps
total) is given as one item's "target"/"past_covariates" span, with
known_covariates_names=["pump_level"] marking Pump Level as known-into-the-future.
min_past=patch_len forces the dataset's random training-window cut point to land
exactly at the context/horizon boundary every time (randint(patch_len, patch_len+1)
has only one possible value), since each episode here IS the fixed-length window --
not an arbitrarily-long continuous series to sample varying-length windows from, the
way Chronos's own pretraining data looks.

Run with: python train_chronos2.py --save_path saved_models/chronos2_mcs
"""
import argparse
import os

import numpy as np

from chronos import Chronos2Pipeline
from chronos.chronos2 import preprocess
import sys

sys.path.insert(0, "forecasting")
from train_tokenizer import load_mcs_episode_halves  # noqa: E402


def build_finetune_items(data_root, split, patch_len):
    x_half, y_half = load_mcs_episode_halves(data_root, split, patch_len)
    x_half, y_half = x_half.numpy(), y_half.numpy()  # (E, 12, patch_len) channel-major physical

    items = []
    for i in range(x_half.shape[0]):
        full_target = np.concatenate([x_half[i, 0:11, :], y_half[i, 0:11, :]], axis=-1)  # (11, 2*patch_len)
        full_pump_level = np.concatenate([x_half[i, 11, :], y_half[i, 11, :]], axis=-1)   # (2*patch_len,)
        items.append({
            "target": full_target,
            "past_covariates": {"pump_level": full_pump_level},
        })
    return items


def main(args):
    device = f"cuda:{args.gpu}"

    print("Building fine-tune items from MCS train/val splits...")
    train_items = build_finetune_items(args.data_root, "train", args.patch_len)
    val_items = build_finetune_items(args.data_root, "val", args.patch_len)
    print(f"train episodes: {len(train_items)}  val episodes: {len(val_items)}")

    print("Preparing covariate-aware inputs (preprocess.from_list_of_dicts)...")
    train_prepared = preprocess.from_list_of_dicts(
        train_items, prediction_length=args.patch_len, known_covariates_names=["pump_level"],
    )
    val_prepared = preprocess.from_list_of_dicts(
        val_items, prediction_length=args.patch_len, known_covariates_names=["pump_level"],
    )

    print(f"Loading pretrained Chronos-2: {args.model_id}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=device)

    os.makedirs(args.save_path, exist_ok=True)
    finetuned_pipeline = pipeline.fit(
        train_prepared,
        prediction_length=args.patch_len,
        validation_inputs=val_prepared,
        finetune_mode=args.finetune_mode,
        context_length=args.context_length,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        min_past=args.patch_len,
        output_dir=args.save_path,
        finetuned_ckpt_name=args.finetuned_ckpt_name,
    )
    print(f"Fine-tuned checkpoint saved under {os.path.join(args.save_path, '*', args.finetuned_ckpt_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2")
    parser.add_argument("--data_root", type=str, default="forecasting/data_raw/abiomed")
    parser.add_argument("--patch_len", type=int, default=6)
    parser.add_argument("--save_path", type=str, default="saved_models/chronos2_mcs")
    parser.add_argument("--finetuned_ckpt_name", type=str, default="finetuned-ckpt")
    parser.add_argument("--finetune_mode", type=str, default="full", choices=["full", "lora"])
    parser.add_argument("--context_length", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
