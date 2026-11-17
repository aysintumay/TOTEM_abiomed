"""
Fine-tune Chronos-2 on MCS data with GORMPO's cross-channel-attended x_tilde as input
AND Chronos-2's own GroupSelfAttention stripped from the encoder
(chronos2_time_only_encoder.py) -- "version C": channels mix exactly once, in GORMPO's
cross_channel_attn, before Chronos-2 ever sees the data. Contrast:
  - train_chronos2_xtilde.py / train_chronos2_xtilde_jointtrain.py: x_tilde input but
    GroupSelfAttention kept -> channels mix twice.
  - train_chronos2_xbar_notimegroup.py: GroupSelfAttention stripped but x_bar input
    (cross_channel_attn skipped) -> channels never mix.
This script is that one with x_bar swapped for x_tilde (gormpo_xtilde.compute_x_tilde),
cross_channel_attn FROZEN (same rationale as train_chronos2_xtilde.py / gormpo_xtilde.py:
x_tilde is only affine-decodable because cross_channel_attn was trained under the
tokenizer's reconstruction objective). x_tilde is therefore precomputed under
torch.no_grad(), and fit() must still be bypassed for the same reason documented in
train_chronos2_xbar_notimegroup.py (it would rebuild a model with GroupSelfAttention).

Pump Level covariate (past AND future): globally z-scored, (P - mean) / std with the
dataset-wide stats (gormpo_xtilde.load_pump_level_stats) -- NOT its x_tilde row. Two
reasons: (1) the future patch's x_tilde row comes out of cross_channel_attn over the
whole FUTURE patch, i.e. it would leak the very vitals being predicted into the
covariate; (2) per-patch z-scoring erases the absolute P-level and step size. The 11
target rows still use x_tilde (context: past-only attention, no leak; future: labels
only). Pump Level is never predicted -- it's masked out of future_target_mask.

Run with: python train_chronos2_xtilde_notimegroup.py --save_path saved_models/chronos2_xtilde_notimegroup_globalpl_mcs
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

from chronos import Chronos2Pipeline
from chronos2_time_only_encoder import strip_group_self_attention
from gormpo_xtilde import compute_x_tilde, load_pump_level_stats

sys.path.insert(0, "forecasting")
from train_tokenizer import load_mcs_episode_halves  # noqa: E402


def build_batch_tensors(tokenizer, x_batch, y_batch, pl_mean, pl_std):
    """x_batch/y_batch: (b, 12, patch_len) raw physical. cross_channel_attn is frozen,
    so x_tilde is computed under torch.no_grad(). Returns the (context, future_target,
    future_target_mask, future_covariates, future_covariates_mask, group_ids) contract
    Chronos2Model.forward() expects, row order b*12+n (11 target channels then Pump
    Level, per episode). Pump Level rows (context and future covariate) are globally
    z-scored with pl_mean/pl_std, not x_tilde -- see module docstring."""
    b, N, patch_len = x_batch.shape
    device = x_batch.device

    with torch.no_grad():
        x_tilde_ctx, _, _ = compute_x_tilde(tokenizer, x_batch)
        x_tilde_fut, _, _ = compute_x_tilde(tokenizer, y_batch)

    context = x_tilde_ctx.clone()
    context[:, 11, :] = (x_batch[:, 11, :] - pl_mean) / pl_std
    context = context.reshape(b * N, patch_len)

    target_row = torch.zeros(N, dtype=torch.bool, device=device)
    target_row[0:11] = True
    target_row_mask = target_row.unsqueeze(0).expand(b, N).unsqueeze(-1).expand(b, N, patch_len)

    zeros_like_fut = torch.zeros_like(x_tilde_fut)
    future_target = torch.where(target_row_mask, x_tilde_fut, zeros_like_fut).reshape(b * N, patch_len)
    future_target_mask = target_row_mask.reshape(b * N, patch_len).float()

    pl_fut = zeros_like_fut.clone()
    pl_fut[:, 11, :] = (y_batch[:, 11, :] - pl_mean) / pl_std  # never through cross_channel_attn: no future-vitals leak
    future_covariates = torch.where(~target_row_mask, pl_fut, zeros_like_fut).reshape(b * N, patch_len)
    future_covariates_mask = (~target_row_mask).reshape(b * N, patch_len).float()

    group_ids = torch.arange(b, device=device).repeat_interleave(N)
    return context, future_target, future_target_mask, future_covariates, future_covariates_mask, group_ids


@torch.no_grad()
def run_validation(model, tokenizer, val_loader, device, pl_mean, pl_std):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        context, future_target, future_target_mask, future_covariates, future_covariates_mask, group_ids = (
            build_batch_tensors(tokenizer, x_batch, y_batch, pl_mean, pl_std)
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
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)  # cross_channel_attn frozen (see module docstring)
    pl_mean, pl_std = load_pump_level_stats(args.mcs_pickle_path)
    print(f"Pump Level global z-score: mean={pl_mean:.4f} std={pl_std:.4f}")

    print(f"Loading pretrained Chronos-2: {args.model_id}")
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=device)
    model = pipeline.model
    strip_group_self_attention(model)
    print("GroupSelfAttention stripped from the encoder (TimeSelfAttention + FeedForward only).")
    model.train()

    x_train, y_train = load_mcs_episode_halves(args.data_root, "train", args.patch_len)
    x_val, y_val = load_mcs_episode_halves(args.data_root, "val", args.patch_len)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    print(f"train episodes: {len(x_train)}  val episodes: {len(x_val)}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Total # trainable parameters:", sum(p.numel() for p in trainable_params))
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
            build_batch_tensors(tokenizer, x_batch, y_batch, pl_mean, pl_std)
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
            val_loss = run_validation(model, tokenizer, val_loader, device, pl_mean, pl_std)
            print(f"| step {step:5d}/{args.num_steps} | val loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving best checkpoint...")
                model.save_pretrained(os.path.join(args.save_path, args.finetuned_ckpt_name))

    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved under {os.path.join(args.save_path, args.finetuned_ckpt_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2")
    parser.add_argument("--tokenizer_path", type=str,
                         default="forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth")
    parser.add_argument("--data_root", type=str, default="forecasting/data_raw/abiomed")
    parser.add_argument("--mcs_pickle_path", type=str, default="/public/gormpo/10min_1hr_all_data.pkl")
    parser.add_argument("--patch_len", type=int, default=6)
    parser.add_argument("--save_path", type=str, default="saved_models/chronos2_xtilde_notimegroup_globalpl_mcs")
    parser.add_argument("--finetuned_ckpt_name", type=str, default="finetuned-ckpt")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64,
                         help="episodes per batch; even without GroupSelfAttention active, its "
                              "weights are still allocated on the model, so keep this modest")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
