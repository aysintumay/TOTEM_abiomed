"""
Train GormpoTokenLLMForecaster (lib/models/gormpo_llm_forecaster.py) on MCS tokens
extracted by extract_gormpo_tokens.py. Mirrors train_llm_forecaster.py's structure
(LoRA + OneCycle + EarlyStopping), but the loss combines shape-code cross-entropy
with the four scalar-bin cross-entropies, and val/test decode through the trained
GormpoTokenizer (no RevIN anywhere in this path -- affine restoration uses the
model's own *predicted* mu/sigma, per Eq 28).

Run with: python train_gormpo_llm_forecaster.py --data_root <extracted tokens dir> \
    --tokenizer_path <trained tokenizer checkpoint>
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_constant_schedule_with_warmup

from lib.models.gormpo_llm_forecaster import (
    GormpoTokenLLMForecaster,
    flatten_channels_scalar,
    unflatten_channels_scalar,
)
from lib.models.llm_forecaster import build_training_sequence, flatten_channels
from lib.models.metrics import dtw, pearsoncor
from lib.utils.checkpoint import EarlyStopping
from lib.utils.env import seed_all_rng

SCALAR_NAMES = ("mu", "sigma", "min", "max")
# Three separate per-patch reward fields (Eq 24-25), replacing the old single "r" --
# see tokenizer.py's RewardQuantizer and extract_gormpo_tokens.py.
REWARD_NAMES = ("r_map", "r_hr", "r_pulsat")
# Pump Level is channel 11 of 12 (see abiomed_env/model.py's `columns`/FEATURE_NAMES
# ordering) -- the *action*, not something to forecast. It's split out of the
# channel set below and its target-patch mu instead becomes a conditioning input
# (GormpoTokenLLMForecaster.future_action_embedding), matching how the ICL world
# model (gormpo_world_model.py) is given the next-window P-level as a known input
# rather than asked to predict it. Without this split, the model has no way to
# answer "what happens if P-level is set to X" -- it would just forecast its own
# guess of P-level jointly with the 11 physiological channels, not usable as an
# MBRL world model queried with a specific chosen action.
PL_CHANNEL = 11


def split_action_channel(y_orig, x_codes, y_codes, x_mu, x_sigma, x_min, x_max,
                          x_r_map, x_r_hr, x_r_pulsat, y_mu, y_sigma, y_min, y_max):
    """Drops Pump Level (channel PL_CHANNEL) from the 12-channel forecast set (now 11
    physiological channels only) and pulls the given future action from y_orig's own
    raw Pump Level column directly -- a natural small integer (2-10), not routed
    through the generic 32-bin ScalarBinQuantizer (q_mu) built for continuous vitals,
    which would just lossily rebin something already discrete. Kept at full per-
    timestep resolution (B, patch_len), NOT collapsed to the patch's mean level --
    P-level can ramp within a 6-step window, and the ICL world model is given the
    whole per-step sequence in its prompt, not a single summary value; collapsing
    here would make this model's conditioning silently weaker and not a fair
    comparison. All (..., 12) inputs -> (..., 11) plus one extra (B, patch_len)
    zero-indexed action tensor (level - 2)."""
    y_action = y_orig[:, :, PL_CHANNEL].round().clamp(2, 10).long() - 2  # (B, patch_len)
    keep = [i for i in range(x_codes.shape[-1]) if i != PL_CHANNEL]
    sliced = (t[:, :, keep] if t.dim() == 3 else t[:, keep] for t in
              (x_codes, y_codes, x_mu, x_sigma, x_min, x_max, x_r_map, x_r_hr, x_r_pulsat, y_mu, y_sigma, y_min, y_max))
    return (*sliced, y_action)


def create_gormpo_dataloader(datapath, batchsize, num_workers=4):
    dataloaders = {}
    for split in ("train", "val", "test"):
        def _load(name, dtype):
            arr = np.load(os.path.join(datapath, f"{split}_{name}.npy"))
            return torch.from_numpy(arr).to(dtype=dtype)

        x_orig = _load("x_original", torch.float32)
        y_orig = _load("y_original", torch.float32)
        x_codes = _load("x_codes", torch.int64)
        y_codes = _load("y_codes", torch.int64)
        N = x_codes.shape[-1]

        scalars = [_load(f"x_{s}", torch.int64) for s in SCALAR_NAMES]
        # reward is stored per-patch (num_episodes,), not per-channel like the other
        # scalars -- broadcast to (num_episodes, N) so every channel's row sees the
        # same three reward values, matching how embed() sums them onto every row.
        scalars += [_load(f"x_{s}", torch.int64).unsqueeze(1).expand(-1, N).contiguous() for s in REWARD_NAMES]
        scalars += [_load(f"y_{s}", torch.int64) for s in SCALAR_NAMES]

        dataset = TensorDataset(x_orig, y_orig, x_codes, y_codes, *scalars)
        dataloaders[split] = DataLoader(
            dataset, batch_size=batchsize, shuffle=(split == "train"),
            num_workers=num_workers, drop_last=(split == "train"),
        )
    return dataloaders


def _unpack(data, device):
    (x_orig, y_orig, x_codes, y_codes, x_mu, x_sigma, x_min, x_max,
     x_r_map, x_r_hr, x_r_pulsat, y_mu, y_sigma, y_min, y_max) = (t.to(device) for t in data)
    return (x_orig, y_orig, x_codes, y_codes, x_mu, x_sigma, x_min, x_max,
            x_r_map, x_r_hr, x_r_pulsat, y_mu, y_sigma, y_min, y_max)


def train_one_epoch(dataloader, model, optimizer, scheduler, epoch, device, scalar_loss_weight, grad_accum_steps=1):
    running_loss, running_shape_loss, running_scalar_loss, running_acc = 0.0, 0.0, 0.0, 0.0
    log_every = max(len(dataloader) // 3, 3)
    num_batches = len(dataloader)
    optimizer.zero_grad()

    for i, data in enumerate(dataloader):
        (_, y_orig, x_codes, y_codes, x_mu, x_sigma, x_min, x_max,
         x_r_map, x_r_hr, x_r_pulsat, y_mu, y_sigma, y_min, y_max) = _unpack(data, device)
        (x_codes, y_codes, x_mu, x_sigma, x_min, x_max, x_r_map, x_r_hr, x_r_pulsat,
         y_mu, y_sigma, y_min, y_max, y_action) = split_action_channel(
            y_orig, x_codes, y_codes, x_mu, x_sigma, x_min, x_max, x_r_map, x_r_hr, x_r_pulsat,
            y_mu, y_sigma, y_min, y_max,
        )
        B, TCin, N = x_codes.shape  # N=11 physiological channels now (Pump Level split out)

        x_ids = flatten_channels(x_codes)  # (B*N, TCin)
        y_ids = flatten_channels(y_codes)  # (B*N, TCout)
        x_mu_f, x_sigma_f, x_min_f, x_max_f, x_r_map_f, x_r_hr_f, x_r_pulsat_f = (
            flatten_channels_scalar(t) for t in (x_mu, x_sigma, x_min, x_max, x_r_map, x_r_hr, x_r_pulsat)
        )
        y_scalars_f = {name: flatten_channels_scalar(t) for name, t in zip(SCALAR_NAMES, (y_mu, y_sigma, y_min, y_max))}
        # y_action is (B, patch_len) -- broadcast across the N=11 channel rows (same
        # convention as the per-patch reward fields), keeping the per-timestep axis.
        y_action_f = y_action.unsqueeze(1).expand(-1, N, -1).reshape(-1, y_action.shape[-1])

        inputs, labels = build_training_sequence(x_ids, y_ids)
        shape_logits, hidden = model(
            inputs, x_mu_f, x_sigma_f, x_min_f, x_max_f, x_r_map_f, x_r_hr_f, x_r_pulsat_f,
            y_action_f, TCin=TCin,
        )
        # shape_logits/hidden are (B*N, action_len + TCin+TCout-1, ...) -- the leading
        # action_len positions are the given action (its own sequence positions now, see
        # GormpoTokenLLMForecaster.embed()), never a prediction target; drop them before
        # comparing against labels, which only cover the original TCin+TCout-1 codes.
        shape_logits = shape_logits[:, model.action_len:, :]
        loss_shape = F.cross_entropy(
            shape_logits.reshape(-1, shape_logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )

        boundary_hidden = hidden[:, model.action_len + TCin - 1, :]
        scalars_pred = model.predict_scalars(boundary_hidden)
        loss_scalar = sum(F.cross_entropy(scalars_pred[name], y_scalars_f[name]) for name in SCALAR_NAMES)

        loss = loss_shape + scalar_loss_weight * loss_scalar

        # Gradient accumulation: physical batch size stays whatever fits in memory
        # (GPU 1 is shared and its free memory fluctuates with other users' jobs --
        # see train run history), but summing gradients over grad_accum_steps
        # micro-batches before each optimizer step simulates a larger *effective*
        # batch (less noisy gradient estimate) at the same peak memory footprint as
        # a single micro-batch. Divide by grad_accum_steps so accumulated grads
        # average correctly instead of summing to grad_accum_steps times their scale.
        (loss / grad_accum_steps).backward()
        is_last_batch = (i + 1) == num_batches
        if (i + 1) % grad_accum_steps == 0 or is_last_batch:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            mask = labels != -100
            preds = shape_logits.argmax(dim=-1)
            acc = (preds[mask] == labels[mask]).float().mean().item()

        running_loss += loss.item()
        running_shape_loss += loss_shape.item()
        running_scalar_loss += loss_scalar.item()
        running_acc += acc
        if i % log_every == log_every - 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"| epoch {epoch:3d} | {i + 1:5d}/{len(dataloader):5d} batches | lr {lr:.6f} | "
                f"loss {running_loss / log_every:5.4f} | shape_loss {running_shape_loss / log_every:5.4f} | "
                f"scalar_loss {running_scalar_loss / log_every:5.4f} | code_acc {running_acc / log_every:5.4f}"
            )
            running_loss, running_shape_loss, running_scalar_loss, running_acc = 0.0, 0.0, 0.0, 0.0


@torch.no_grad()
def inference(data, model, tokenizer, device):
    """Returns (pred_time, labels_time), both (B, patch_len, N) raw physiological units,
    N=11 -- Pump Level is excluded from both: it's given as the conditioning action
    (from the real episode, same counterfactual-eval convention as the ICL path), not
    forecasted, so there's nothing of the model's own to score for that channel."""
    (x_orig, y_orig, x_codes, y_codes, x_mu, x_sigma, x_min, x_max,
     x_r_map, x_r_hr, x_r_pulsat, y_mu, y_sigma, y_min, y_max) = _unpack(data, device)
    (x_codes, y_codes, x_mu, x_sigma, x_min, x_max, x_r_map, x_r_hr, x_r_pulsat,
     y_mu, y_sigma, y_min, y_max, y_action) = split_action_channel(
        y_orig, x_codes, y_codes, x_mu, x_sigma, x_min, x_max, x_r_map, x_r_hr, x_r_pulsat,
        y_mu, y_sigma, y_min, y_max,
    )
    B, TCin, N = x_codes.shape  # N=11
    TCout = y_codes.shape[1]

    x_ids = flatten_channels(x_codes)
    x_mu_f, x_sigma_f, x_min_f, x_max_f, x_r_map_f, x_r_hr_f, x_r_pulsat_f = (
        flatten_channels_scalar(t) for t in (x_mu, x_sigma, x_min, x_max, x_r_map, x_r_hr, x_r_pulsat)
    )
    y_action_f = y_action.unsqueeze(1).expand(-1, N, -1).reshape(-1, y_action.shape[-1])

    y_ids_pred, scalars_pred = model.generate_codes(
        x_ids, TCout, x_mu_f, x_sigma_f, x_min_f, x_max_f, x_r_map_f, x_r_hr_f, x_r_pulsat_f, y_action_f
    )

    mu_idx = unflatten_channels_scalar(scalars_pred["mu"].argmax(dim=-1), B, N)
    sigma_idx = unflatten_channels_scalar(scalars_pred["sigma"].argmax(dim=-1), B, N)
    mu_hat = tokenizer.q_mu.dequantize(mu_idx)
    sigma_hat = tokenizer.q_sigma.dequantize(sigma_idx)

    num_code_words, code_dim = tokenizer.vq._embedding.weight.shape
    one_hot = F.one_hot(y_ids_pred, num_code_words).to(tokenizer.vq._embedding.weight.dtype)  # (B*N, TCout, vocab)
    quantized = torch.matmul(one_hot, tokenizer.vq._embedding.weight)  # (B*N, TCout, code_dim)
    quantized = quantized.transpose(1, 2)  # (B*N, code_dim, TCout)

    x_hat, _ = tokenizer.decode(quantized, mu_hat, sigma_hat)  # (B, N, patch_len)
    pred_time = x_hat.transpose(1, 2)  # (B, patch_len, N)
    keep = [i for i in range(y_orig.shape[-1]) if i != PL_CHANNEL]
    labels_time = y_orig[:, :, keep]  # (B, patch_len, 11) -- drop Pump Level to match pred_time
    return pred_time, labels_time


def run_eval(dataloader, model, tokenizer, device, tag, save_file):
    model.eval()
    running_mse, running_mae, running_cor, running_dtw = 0.0, 0.0, 0.0, 0.0
    total_num, total_num_c = 0.0, 0.0
    for data in dataloader:
        pred_time, labels_time = inference(data, model, tokenizer, device)
        running_mse += F.mse_loss(pred_time, labels_time, reduction="sum")
        running_mae += (pred_time - labels_time).abs().sum()
        running_cor += pearsoncor(pred_time, labels_time, reduction="sum")
        running_dtw += dtw(pred_time, labels_time, reduction="sum")
        total_num += labels_time.numel()
        total_num_c += labels_time.shape[0] * labels_time.shape[2]
    mse, mae = running_mse / total_num, running_mae / total_num
    cor, dt = running_cor / total_num_c, running_dtw / total_num_c
    msg = f"| [{tag}] mse {mse:5.4f} mae {mae:5.4f} corr {cor:5.4f} dtw {dt:5.4f}"
    print(msg)
    save_file.write(msg + "\n")
    return mse, mae


def train(args):
    os.makedirs(args.file_save_path, exist_ok=True)
    save_file = open(os.path.join(args.file_save_path, f"gormpo_llm_{args.seed}.txt"), "w+")

    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_all_rng(None if args.seed < 0 else args.seed)

    if args.checkpoint:
        os.makedirs(args.checkpoint_path, exist_ok=True)
    early_stopping = EarlyStopping(patience=args.patience, path=args.checkpoint_path)

    tokenizer = torch.load(args.tokenizer_path, map_location=device, weights_only=False)
    tokenizer.to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    dataloaders = create_gormpo_dataloader(args.data_root, args.batchsize, args.num_workers)
    train_dataloader, val_dataloader, test_dataloader = (
        dataloaders["train"], dataloaders["val"], dataloaders["test"]
    )

    model = GormpoTokenLLMForecaster(
        llm_name=args.llm_name, num_embeddings=args.codebook_size, num_bins=args.num_bins,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules.split(","),
        load_in_4bit=args.load_in_4bit, device=args.cuda_id,
    )
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Total # trainable parameters:", sum(p.numel() for p in trainable_params),
          " / total params:", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(trainable_params, lr=args.baselr)
    # OneCycleLR needs epochs * steps_per_epoch up front to shape its warmup/anneal
    # curve, but EarlyStopping can (and did, in an earlier run: stopped at epoch 7 of
    # a schedule sized for 50) cut training off long before that horizon -- stranding
    # LR mid-ramp, far below its configured peak, for the entire run. A warmup-then-
    # constant schedule doesn't need to know the eventual stopping point, so it can't
    # be orphaned by early stopping the same way.
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    for epoch in range(args.epochs):
        model.train()
        train_one_epoch(
            train_dataloader, model, optimizer, scheduler, epoch, device,
            args.scalar_loss_weight, args.grad_accum_steps,
        )

        val_mse, val_mae = run_eval(val_dataloader, model, tokenizer, device, "Val", save_file)
        early_stopping_counter = early_stopping(val_mse, val_mae, {"gormpo_llm_forecaster": model})
        _ = run_eval(test_dataloader, model, tokenizer, device, "Test", save_file)
        save_file.write(f"Early stopping counter is: {early_stopping_counter}\n")

        if early_stopping.early_stop:
            print("Early stopping....")
            save_file.write("Early stopping....\n")
            break

    save_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_id", default=0, type=int)
    parser.add_argument("--seed", default=2021, type=int)
    parser.add_argument("--data_root", type=str, default="data/gormpo_tokens_mcs")
    parser.add_argument("--tokenizer_path", type=str, default="saved_models/gormpo_tokenizer_mcs/checkpoints/final_tokenizer.pth")
    parser.add_argument("--codebook_size", default=256, type=int)
    parser.add_argument("--num_bins", default=32, type=int)
    parser.add_argument("--scalar_loss_weight", default=1.0, type=float)
    parser.add_argument("--batchsize", default=128, type=int)
    parser.add_argument("--grad_accum_steps", default=1, type=int,
                         help="Accumulate gradients over this many micro-batches per optimizer step, "
                              "simulating a larger effective batch (batchsize * grad_accum_steps) at the "
                              "same peak memory as --batchsize alone.")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--checkpoint", action="store_true", default=True)
    parser.add_argument("--checkpoint_path", default="saved_models/gormpo_llm_mcs/checkpoints/", type=str)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--baselr", default=1e-4, type=float)
    parser.add_argument("--warmup_steps", default=200, type=int,
                         help="Linear LR warmup steps before holding constant at --baselr; unlike OneCycleLR "
                              "this does not need a total-steps horizon, so early stopping can't strand it mid-ramp.")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--llm_name", default="Qwen/Qwen2.5-0.5B", type=str)
    parser.add_argument("--load_in_4bit", action="store_true", default=False,
                         help="QLoRA: load the base LLM in 4-bit (bitsandbytes) to fit larger backbones on a shared GPU.")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument(
        "--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", type=str,
    )
    parser.add_argument("--file_save_path", default="results_llm/gormpo_mcs/", type=str)
    args = parser.parse_args()

    train(args)
