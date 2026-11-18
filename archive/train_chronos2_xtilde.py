"""
Fine-tune Chronos-2 on MCS data using GORMPO's cross-channel-attended x_tilde
representation (gormpo_xtilde.py) instead of raw physical values (contrast
train_chronos2.py) or discrete VQ-quantized shape codes (contrast
train_gormpo_chronos2.py). cross_channel_attn is FROZEN here -- see
gormpo_xtilde.py's module docstring and eval_chronos2_xtilde_zeroshot.py's for why:
x_tilde is precomputed ONCE (no gradients needed through the tokenizer), so this
reuses Chronos2Pipeline.fit() exactly as train_chronos2.py does, just fed x_tilde
instead of raw values. Contrast train_chronos2_xtilde_jointtrain.py, which keeps
cross_channel_attn trainable and therefore can't use this same precompute-once +
pipeline.fit() shortcut.

Same task framing as train_chronos2.py: the 11 physiological channels are the joint
multivariate target; Pump Level's own x_tilde (not raw value, for representation
consistency with the targets) is the known-future covariate.

Run with: python train_chronos2_xtilde.py --save_path saved_models/chronos2_xtilde_mcs
"""
import argparse
import os
import sys

import numpy as np
import torch

from chronos import Chronos2Pipeline
from chronos.chronos2 import preprocess
from gormpo_xtilde import compute_x_tilde

sys.path.insert(0, "forecasting")
from train_tokenizer import load_mcs_episode_halves  # noqa: E402


def build_finetune_items(tokenizer, data_root, split, patch_len, device, batch_size=512):
    x_half, y_half = load_mcs_episode_halves(data_root, split, patch_len)  # (E, 12, patch_len) physical

    items = []
    for start in range(0, x_half.shape[0], batch_size):
        x_batch = x_half[start:start + batch_size].to(device)
        y_batch = y_half[start:start + batch_size].to(device)
        x_tilde_ctx, _, _ = compute_x_tilde(tokenizer, x_batch)  # (b, 12, patch_len)
        x_tilde_fut, _, _ = compute_x_tilde(tokenizer, y_batch)
        x_tilde_ctx = x_tilde_ctx.cpu().numpy()
        x_tilde_fut = x_tilde_fut.cpu().numpy()

        for b in range(x_tilde_ctx.shape[0]):
            full_target = np.concatenate([x_tilde_ctx[b, 0:11, :], x_tilde_fut[b, 0:11, :]], axis=-1)  # (11, 2*patch_len)
            full_pump_level = np.concatenate([x_tilde_ctx[b, 11, :], x_tilde_fut[b, 11, :]], axis=-1)   # (2*patch_len,)
            items.append({
                "target": full_target,
                "past_covariates": {"pump_level": full_pump_level},
            })
    return items


def main(args):
    device = f"cuda:{args.gpu}"

    tokenizer = torch.load(args.tokenizer_path, map_location=device, weights_only=False)
    tokenizer.to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)  # frozen -- see gormpo_xtilde.py's module docstring

    print("Building fine-tune items (x_tilde) from MCS train/val splits...")
    train_items = build_finetune_items(tokenizer, args.data_root, "train", args.patch_len, device)
    val_items = build_finetune_items(tokenizer, args.data_root, "val", args.patch_len, device)
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
    pipeline.fit(
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
    print(f"Fine-tuned checkpoint saved under {os.path.join(args.save_path, args.finetuned_ckpt_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2")
    parser.add_argument("--tokenizer_path", type=str,
                         default="forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth")
    parser.add_argument("--data_root", type=str, default="forecasting/data_raw/abiomed")
    parser.add_argument("--patch_len", type=int, default=6)
    parser.add_argument("--save_path", type=str, default="saved_models/chronos2_xtilde_mcs")
    parser.add_argument("--finetuned_ckpt_name", type=str, default="finetuned-ckpt")
    parser.add_argument("--finetune_mode", type=str, default="full", choices=["full", "lora"])
    parser.add_argument("--context_length", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
