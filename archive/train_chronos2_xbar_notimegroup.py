"""
Fine-tune Chronos-2 on MCS data with BOTH cross-channel-mixing mechanisms removed:
GORMPO's cross_channel_attn is skipped entirely (feeding x_bar -- GORMPO's basic
per-channel z-score normalization, patch_and_scale's output -- instead of x_tilde),
and Chronos-2's own GroupSelfAttention is stripped from the encoder
(chronos2_time_only_encoder.py), the same removal train_gormpo_chronos2.py already
applied to the discrete-token architecture. Contrast train_chronos2_xtilde.py /
train_chronos2_xtilde_jointtrain.py, which keep GroupSelfAttention active natively
while ALSO feeding the cross-channel-attended x_tilde -- two independent
cross-channel-mixing mechanisms stacked, never deliberately reasoned about together.
See chronos2_time_only_encoder.py's module docstring for the concrete diagnostic
evidence (LVEDP-SYSTOLIC entanglement) motivating this ablation.

x_bar has no trainable parameters upstream of it (patch_and_scale is pure arithmetic,
no nn.Parameter) -- so unlike train_chronos2_xtilde_jointtrain.py there's no
joint-training-vs-frozen question, and no autograd-through-a-trainable-preprocessing
-module complexity. x_bar is precomputed once with torch.no_grad().

IMPORTANT: this can NOT reuse Chronos2Pipeline.fit() the way train_chronos2_xtilde.py
does, even though x_bar needs no gradients -- fit() does, internally:
    config = deepcopy(self.model.config)
    model = Chronos2Model(config).to(self.model.device)
    model.load_state_dict(self.model.state_dict())
i.e. it always constructs a BRAND NEW Chronos2Model from config (which has the
NATIVE Chronos2Encoder, attribute name "block", WITH GroupSelfAttention) and copies
weights into it from self.model.state_dict(). TimeOnlyChronos2Encoder renames the
attribute to "blocks" (and drops GroupSelfAttention's role in forward()), so (a) the
state_dict keys don't even match ("encoder.block.*" vs "encoder.blocks.*") and
(b) even if they did, fit() would silently hand back a model with GroupSelfAttention
restored and active -- defeating the whole point of this ablation. So this calls
Chronos2Model.forward() directly instead, exactly like train_chronos2_xtilde_jointtrain.py,
just with precomputed no-grad x_bar in place of trainable-per-step x_tilde (no
tokenizer parameters are optimized here at all).

Run with: python train_chronos2_xbar_notimegroup.py --save_path saved_models/chronos2_xbar_notimegroup_mcs
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

from chronos import Chronos2Pipeline
from chronos2_time_only_encoder import strip_group_self_attention

sys.path.insert(0, "forecasting")
from train_tokenizer import load_mcs_episode_halves  # noqa: E402


def build_batch_tensors(tokenizer, x_batch, y_batch):
    """x_batch/y_batch: (b, 12, patch_len) raw physical. x_bar has no trainable
    parameters upstream (patch_and_scale is pure arithmetic), so this runs fully
    under torch.no_grad() -- contrast train_chronos2_xtilde_jointtrain.py's
    build_batch_tensors, which needs gradients through cross_channel_attn and
    therefore needs the NaN-safe explicit-mask treatment. Returns the same
    (context, future_target, future_target_mask, future_covariates,
    future_covariates_mask, group_ids) contract Chronos2Model.forward() expects,
    row order b*12+n (11 target channels then Pump Level, per episode)."""
    b, N, patch_len = x_batch.shape
    device = x_batch.device

    with torch.no_grad():
        x_bar_ctx, _, _ = tokenizer.patch_and_scale(x_batch)
        x_bar_fut, _, _ = tokenizer.patch_and_scale(y_batch)

    context = x_bar_ctx.reshape(b * N, patch_len)

    target_row = torch.zeros(N, dtype=torch.bool, device=device)
    target_row[0:11] = True
    target_row_mask = target_row.unsqueeze(0).expand(b, N).unsqueeze(-1).expand(b, N, patch_len)

    zeros_like_fut = torch.zeros_like(x_bar_fut)
    future_target = torch.where(target_row_mask, x_bar_fut, zeros_like_fut).reshape(b * N, patch_len)
    future_target_mask = target_row_mask.reshape(b * N, patch_len).float()

    future_covariates = torch.where(~target_row_mask, x_bar_fut, zeros_like_fut).reshape(b * N, patch_len)
    future_covariates_mask = (~target_row_mask).reshape(b * N, patch_len).float()

    group_ids = torch.arange(b, device=device).repeat_interleave(N)
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
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)  # x_bar has no trainable params upstream anyway

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

    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved under {os.path.join(args.save_path, args.finetuned_ckpt_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="amazon/chronos-2")
    parser.add_argument("--tokenizer_path", type=str,
                         default="forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth")
    parser.add_argument("--data_root", type=str, default="forecasting/data_raw/abiomed")
    parser.add_argument("--patch_len", type=int, default=6)
    parser.add_argument("--save_path", type=str, default="saved_models/chronos2_xbar_notimegroup_mcs")
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
