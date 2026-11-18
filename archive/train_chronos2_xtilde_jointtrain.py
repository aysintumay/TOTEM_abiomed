"""
Fine-tune Chronos-2 on MCS x_tilde data (gormpo_xtilde.py) JOINTLY with GORMPO's
cross_channel_attn (Eq 10-16) -- contrast train_chronos2_xtilde.py, which keeps
cross_channel_attn frozen. Here, x_tilde is recomputed from raw physical patches on
EVERY training step, with gradients flowing back through cross_channel_attn into the
same optimizer step as Chronos-2's own weights -- cross_channel_attn is no longer
anchored to the tokenizer's original softDTW+MSE reconstruction objective, and is free
to adapt specifically to what makes Chronos-2's forecasting loss lowest instead (see
gormpo_xtilde.py's module docstring for the tradeoff this represents).

Mechanically this can't reuse Chronos2Pipeline.fit() the way train_chronos2.py /
train_chronos2_xtilde.py do: fit()'s Trainer abstraction consumes already-materialized
tensors (via Chronos2Dataset), with no hook for "recompute these through a trainable
module every step." So this calls Chronos2Model.forward() directly instead, replicating
by hand exactly the raw tensor contract Chronos2Dataset._build_batch constructs
(confirmed by reading chronos2/dataset.py directly, not assumed):
  - context: (total_rows, context_length) float -- one row per (episode, channel)
  - future_target: (total_rows, prediction_length) float -- the 11 physiological
    channels' real future x_tilde; a safe dummy (0.0) at the Pump Level covariate row
  - future_covariates: (total_rows, prediction_length) float -- Pump Level's real
    known-future x_tilde; a safe dummy (0.0) at the 11 target rows
  - group_ids: (total_rows,) long, one id per episode, shared by that episode's 12 rows
    (11 targets + Pump Level) -- this is what lets GroupSelfAttention mix exactly the
    channels/covariate of one episode, same mechanism the native (raw-value)
    train_chronos2.py run already relies on.
  - num_output_patches=1, since ceil(prediction_length / output_patch_size) = ceil(6/16)
    = 1 regardless of prediction_length < 16 -- Chronos2Model's own internal
    Patch/padding logic handles a context/prediction window shorter than its native
    patch size; this isn't something the caller needs to pad for by hand (confirmed by
    both train_chronos2.py and train_chronos2_xtilde.py already working correctly at
    context_length=prediction_length=6).

future_target_mask/future_covariates_mask are built and passed EXPLICITLY, not left to
Chronos2Model's own NaN-based auto-detection (contrast Chronos2Dataset, which freely
uses torch.nan as a sentinel, safely, because its future_target/future_covariates are
constant leaf tensors with requires_grad=False). Here they're NOT leaves -- they're
derived from cross_channel_attn's trainable weights, so autograd must backprop INTO
them. A literal NaN inside a tensor that requires_grad can leak NaN through the
backward pass of Chronos2Model's internal `torch.where(mask, future_target, 0.0)`
masking (a well-known PyTorch footgun) even when the forward value comes out finite --
empirically confirmed during development: forward loss was finite on step 0, but
grad_norm was already NaN, corrupting cross_channel_attn's weights via that very first
optimizer step and producing NaN context/predictions from step 1 onward. Using a safe
dummy value (0.0) plus an explicit boolean mask sidesteps this regardless of the exact
internal cause.

Run with: python train_chronos2_xtilde_jointtrain.py --save_path saved_models/chronos2_xtilde_jointtrain_mcs
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

from chronos import Chronos2Pipeline

sys.path.insert(0, "forecasting")
from train_tokenizer import load_mcs_episode_halves  # noqa: E402


def build_batch_tensors(tokenizer, x_batch, y_batch):
    """x_batch/y_batch: (b, 12, patch_len) raw physical, WITH gradients enabled through
    cross_channel_attn (no torch.no_grad() here -- contrast train_chronos2_xtilde.py's
    frozen precompute). Returns (context, future_target, future_target_mask,
    future_covariates, future_covariates_mask, group_ids), row order b*12+n (11 target
    channels then Pump Level, per episode) -- matches Chronos2Dataset._build_batch's
    "targets first, known-future-covariates last" row convention. See module docstring
    for why masks are explicit rather than NaN-sentinel-based here."""
    b, N, patch_len = x_batch.shape
    device = x_batch.device

    x_bar_ctx, _, _ = tokenizer.patch_and_scale(x_batch)
    x_tilde_ctx = tokenizer.cross_channel_attn(x_bar_ctx)  # (b, 12, patch_len) -- gradients flow
    x_bar_fut, _, _ = tokenizer.patch_and_scale(y_batch)
    x_tilde_fut = tokenizer.cross_channel_attn(x_bar_fut)

    context = x_tilde_ctx.reshape(b * N, patch_len)  # (b*12, patch_len)

    target_row = torch.zeros(N, dtype=torch.bool, device=device)
    target_row[0:11] = True  # 11 physiological channels are the forecast target
    target_row_mask = target_row.unsqueeze(0).expand(b, N).unsqueeze(-1).expand(b, N, patch_len)  # (b,12,patch_len)

    zeros_like_fut = torch.zeros_like(x_tilde_fut)
    future_target = torch.where(target_row_mask, x_tilde_fut, zeros_like_fut).reshape(b * N, patch_len)
    # float, not bool: Chronos2Model._compute_loss does `1 - future_covariates_mask`
    # internally, which PyTorch disallows on a bool tensor (RuntimeError: "Subtraction,
    # the `-` operator, with a bool tensor is not supported").
    future_target_mask = target_row_mask.reshape(b * N, patch_len).float()

    future_covariates = torch.where(~target_row_mask, x_tilde_fut, zeros_like_fut).reshape(b * N, patch_len)
    future_covariates_mask = (~target_row_mask).reshape(b * N, patch_len).float()

    group_ids = torch.arange(b, device=device).repeat_interleave(N)  # (b*12,)
    return context, future_target, future_target_mask, future_covariates, future_covariates_mask, group_ids


@torch.no_grad()
def run_validation(model, tokenizer, val_loader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        context, future_target, future_target_mask, future_covariates, future_covariates_mask, group_ids = (
            build_batch_tensors(tokenizer, x_batch, y_batch)
        )
        out = model(
            context=context, future_target=future_target, future_target_mask=future_target_mask,
            future_covariates=future_covariates, future_covariates_mask=future_covariates_mask,
            group_ids=group_ids, num_output_patches=1,
        )
        total_loss += out.loss.item()
        n_batches += 1
    model.train()
    return total_loss / n_batches


def main(args):
    device = f"cuda:{args.gpu}"

    tokenizer = torch.load(args.tokenizer_path, map_location=device, weights_only=False)
    tokenizer.to(device)
    tokenizer.train()  # cross_channel_attn jointly trained; frozen submodules (encoder) stay in eval via GormpoTokenizer.train()'s own override
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    for p in tokenizer.cross_channel_attn.parameters():
        p.requires_grad_(True)  # the ONLY tokenizer params that train here

    print(f"Loading pretrained Chronos-2: {args.model_id}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=device)
    model = pipeline.model
    model.train()
    # Chronos2Model doesn't support HF's standard gradient_checkpointing_enable() (raises
    # ValueError) -- the only lever against GroupSelfAttention's O((batch_size*12)^2)
    # memory cost in this hand-rolled loop is batch_size itself (pipeline.fit()'s
    # internal Trainer handles batch_size=256 fine via some other mechanism we don't
    # have access to here; see --batch_size's default/help text).

    x_train, y_train = load_mcs_episode_halves(args.data_root, "train", args.patch_len)
    x_val, y_val = load_mcs_episode_halves(args.data_root, "val", args.patch_len)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    print(f"train episodes: {len(x_train)}  val episodes: {len(x_val)}")

    trainable_params = list(model.parameters()) + list(tokenizer.cross_channel_attn.parameters())
    print("Total # trainable parameters:", sum(p.numel() for p in trainable_params if p.requires_grad))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    os.makedirs(args.save_path, exist_ok=True)
    best_val_loss = float("inf")
    step = 0
    train_iter = iter(train_loader)
    running_loss, running_n = 0.0, 0

    while step < args.num_steps:
        try:
            x_batch, y_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_batch, y_batch = next(train_iter)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        context, future_target, future_target_mask, future_covariates, future_covariates_mask, group_ids = (
            build_batch_tensors(tokenizer, x_batch, y_batch)
        )
        out = model(
            context=context, future_target=future_target, future_target_mask=future_target_mask,
            future_covariates=future_covariates, future_covariates_mask=future_covariates_mask,
            group_ids=group_ids, num_output_patches=1,
        )
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        running_loss += loss.item()
        running_n += 1
        step += 1

        if step % args.log_every == 0:
            print(f"| step {step:5d}/{args.num_steps} | train loss {running_loss / running_n:.4f}")
            running_loss, running_n = 0.0, 0

        if step % args.eval_every == 0 or step == args.num_steps:
            val_loss = run_validation(model, tokenizer, val_loader, device)
            print(f"| step {step:5d}/{args.num_steps} | val loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving best checkpoint...")
                model.save_pretrained(os.path.join(args.save_path, args.finetuned_ckpt_name))
                torch.save(tokenizer, os.path.join(args.save_path, "tokenizer_with_jointtrained_cross_channel_attn.pth"))

    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved under {os.path.join(args.save_path, args.finetuned_ckpt_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2")
    parser.add_argument("--tokenizer_path", type=str,
                         default="forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth")
    parser.add_argument("--data_root", type=str, default="forecasting/data_raw/abiomed")
    parser.add_argument("--patch_len", type=int, default=6)
    parser.add_argument("--save_path", type=str, default="saved_models/chronos2_xtilde_jointtrain_mcs")
    parser.add_argument("--finetuned_ckpt_name", type=str, default="finetuned-ckpt")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64,
                         help="episodes per batch; GroupSelfAttention's memory cost scales with "
                              "(batch_size * 12 channels)^2, so this is lower than the native "
                              "pipeline.fit()-based scripts default to")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
