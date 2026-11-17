"""
Fine-tune GormpoChronos2Model (gormpo_chronos2_model.py) on MCS tokens extracted by
forecasting/extract_gormpo_tokens.py -- same data source every other GORMPO-tokenizer
model in this project uses (the Qwen2.5-7B+GORMPO LoRA fine-tune, the earlier
GORMPO+Chronos-1 swap), reused here via forecasting/train_gormpo_llm_forecaster.py's
own create_gormpo_dataloader, unmodified.

Loss: sum of cross-entropy over the 3 shape-code heads + scalar_loss_weight * sum of
cross-entropy over the 4 scale-bin heads, supervised ONLY at the 11 physiological
target rows (never at the Pump Level covariate row) -- same split
train_gormpo_llm_forecaster.py's loss already uses, just non-autoregressive here (one
forward pass predicts all 7 facets at once, no token-by-token generate_codes loop,
since GormpoChronos2Model's encoder is bidirectional/non-causal).

Validation/test decode through the trained GormpoTokenizer (no RevIN anywhere in this
path -- affine restoration uses the model's own *predicted* mu/sigma, per Eq 28),
clipped to the training-percentile bounds every other GORMPO-tokenizer eval in this
project needs (the decoder has no inherent output-range guarantee -- see
eval_gormpo_chronos.py's compute_channel_bounds for the SYSTOLIC/PULSAT decode-
explosion case this guards against), and reports per-epoch physical-unit MSE/MAE plus
a final normalized-space per-channel summary (MAE/MSE/DTW/corr) for direct
comparability with every other model in this project's comparison.

Run with: python train_gormpo_chronos2.py --tokenizer_path <GORMPO tokenizer checkpoint>
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from gormpo_chronos2_model import GormpoChronos2Model

sys.path.insert(0, "forecasting")
from train_gormpo_llm_forecaster import create_gormpo_dataloader  # noqa: E402
from lib.models.llm_forecaster import flatten_channels  # noqa: E402
from lib.models.gormpo_llm_forecaster import flatten_channels_scalar  # noqa: E402
from lib.models.metrics import dtw, pearsoncor  # noqa: E402
from lib.utils.checkpoint import EarlyStopping  # noqa: E402
from lib.utils.env import seed_all_rng  # noqa: E402

SCALAR_NAMES = ("mu", "sigma", "min", "max")


def compute_channel_bounds(data_root, low_pct=1, high_pct=99):
    """Per-channel (low, high) physical-unit clipping bounds from the train split's
    raw values (11 physiological channels only) -- same role/reasoning as
    eval_gormpo_chronos.py's compute_channel_bounds."""
    x_orig = np.load(f"{data_root}/train_x_original.npy")  # (E, patch_len, 12)
    flat = x_orig[:, :, :11].reshape(-1, 11)
    low = np.percentile(flat, low_pct, axis=0)
    high = np.percentile(flat, high_pct, axis=0)
    return torch.tensor(low, dtype=torch.float32), torch.tensor(high, dtype=torch.float32)


def flatten_batch(data, device, num_channels):
    """Unpacks one create_gormpo_dataloader batch (same 15-tuple order as
    train_gormpo_llm_forecaster.py's _unpack) and flattens every tensor to (B*N, ...)
    row order b*N+n, matching flatten_channels/flatten_channels_scalar's convention.
    Returns a dict of flat tensors plus a (B*N,) pump_level_row_mask (True at the last
    of every N rows, matching FEATURE_NAMES' Pump-Level-last channel ordering)."""
    (x_orig, y_orig, x_codes, y_codes, x_mu, x_sigma, x_min, x_max,
     x_r_map, x_r_hr, x_r_pulsat, y_mu, y_sigma, y_min, y_max) = (t.to(device) for t in data)

    B = x_codes.shape[0]
    ctx_shape_ids = flatten_channels(x_codes)   # (B*N, L)
    y_shape_ids = flatten_channels(y_codes)     # (B*N, L)
    ctx_mu_f, ctx_sigma_f, ctx_min_f, ctx_max_f = (flatten_channels_scalar(t) for t in (x_mu, x_sigma, x_min, x_max))
    ctx_r_map_f, ctx_r_hr_f, ctx_r_pulsat_f = (flatten_channels_scalar(t) for t in (x_r_map, x_r_hr, x_r_pulsat))
    y_mu_f, y_sigma_f, y_min_f, y_max_f = (flatten_channels_scalar(t) for t in (y_mu, y_sigma, y_min, y_max))

    rows = ctx_shape_ids.shape[0]
    pump_level_row_mask = (torch.arange(rows, device=device) % num_channels) == (num_channels - 1)

    return dict(
        B=B, x_orig=x_orig, y_orig=y_orig,
        ctx_shape_ids=ctx_shape_ids, ctx_mu=ctx_mu_f, ctx_sigma=ctx_sigma_f, ctx_min=ctx_min_f, ctx_max=ctx_max_f,
        ctx_r_map=ctx_r_map_f, ctx_r_hr=ctx_r_hr_f, ctx_r_pulsat=ctx_r_pulsat_f,
        pump_level_row_mask=pump_level_row_mask,
        y_shape_ids=y_shape_ids, y_mu=y_mu_f, y_sigma=y_sigma_f, y_min=y_min_f, y_max=y_max_f,
    )


def forward_and_loss(model, batch, scalar_loss_weight):
    """Pump Level's row uses its OWN y_* values as the model's known-future input
    (the same tensors used as loss labels elsewhere) -- its row's "ground truth future"
    and "known future action fed to the model" are the literal same data, just read at
    different rows depending on role; see gormpo_chronos2_model.py's module docstring."""
    shape_logits, scalar_logits = model(
        batch["ctx_shape_ids"], batch["ctx_mu"], batch["ctx_sigma"], batch["ctx_min"], batch["ctx_max"],
        batch["ctx_r_map"], batch["ctx_r_hr"], batch["ctx_r_pulsat"], batch["pump_level_row_mask"],
        future_shape_ids=batch["y_shape_ids"], future_mu=batch["y_mu"], future_sigma=batch["y_sigma"],
        future_min=batch["y_min"], future_max=batch["y_max"],
    )
    target_row_mask = ~batch["pump_level_row_mask"]
    c0, c1, c2 = shape_logits
    loss_shape = (
        F.cross_entropy(c0[target_row_mask], batch["y_shape_ids"][target_row_mask, 0])
        + F.cross_entropy(c1[target_row_mask], batch["y_shape_ids"][target_row_mask, 1])
        + F.cross_entropy(c2[target_row_mask], batch["y_shape_ids"][target_row_mask, 2])
    )
    loss_scalar = sum(
        F.cross_entropy(scalar_logits[name][target_row_mask], batch[f"y_{name}"][target_row_mask])
        for name in SCALAR_NAMES
    )
    loss = loss_shape + scalar_loss_weight * loss_scalar

    with torch.no_grad():
        preds = c0.argmax(dim=-1)
        acc = (preds[target_row_mask] == batch["y_shape_ids"][target_row_mask, 0]).float().mean().item()

    return loss, loss_shape.item(), loss_scalar.item(), acc, shape_logits, scalar_logits


@torch.no_grad()
def decode_predictions(shape_logits, scalar_logits, batch, tokenizer, bound_low, bound_high):
    """argmax each head -> decode through the trained GormpoTokenizer -> physical
    units, exactly mirroring gormpo_chronos_tokenizer.py's output_transform /
    GormpoTokenLLMForecaster's inference(). Returns (B, 11, patch_len) physical-unit
    predictions, time-major-compatible with y_orig's layout after a transpose.

    Row-layout note: target_row_mask strips exactly channel index (N-1) [Pump Level]
    from every consecutive block of N rows, leaving a REGULAR (B, 11) pattern in
    channel order 0..10 with no gaps or reordering -- so a plain reshape after masking
    already lines a flat (B*11,) tensor up as (B, 11) in true channel order. This is
    simpler than gormpo_chronos_tokenizer.py's _dequantize_per_channel (which needed an
    explicit permute) only because there's no extra num_samples axis interleaved here;
    the underlying hazard it exists to avoid is identical: ScalarBinQuantizer.dequantize
    indexes its PER-CHANNEL bin edges by the id tensor's last dimension, so channel
    identity must land there correctly, or every channel silently dequantizes against
    the wrong edges."""
    target_row_mask = ~batch["pump_level_row_mask"]
    B = batch["B"]
    N_target = 11

    c0, c1, c2 = shape_logits
    shape_ids = torch.stack([c0.argmax(-1), c1.argmax(-1), c2.argmax(-1)], dim=-1)[target_row_mask]  # (B*11, 3)
    mu_ids = scalar_logits["mu"].argmax(-1)[target_row_mask].view(B, N_target)      # (B, 11) -- channel order 0..10
    sigma_ids = scalar_logits["sigma"].argmax(-1)[target_row_mask].view(B, N_target)

    mu_hat = tokenizer.q_mu.dequantize(mu_ids).reshape(B * N_target, 1)       # (B*11, 1)
    sigma_hat = tokenizer.q_sigma.dequantize(sigma_ids).reshape(B * N_target, 1)

    codebook = tokenizer.vq._embedding.weight  # (num_embeddings, embedding_dim)
    L = shape_ids.shape[-1]
    one_hot = F.one_hot(shape_ids.reshape(-1), codebook.shape[0]).to(codebook.dtype)
    quantized = torch.matmul(one_hot, codebook).view(B * N_target, L, -1).transpose(1, 2)  # (B*11, emb_dim, L)

    # decode()'s own (B', N') grouping is just a reshape convenience for its output,
    # unrelated to real channel identity (already resolved above) -- B'=B*11, N'=1.
    x_hat, _ = tokenizer.decode(quantized, mu_hat, sigma_hat)  # (B*11, 1, patch_len)
    pred_phys = x_hat.squeeze(1).view(B, N_target, -1)         # (B, 11, patch_len)
    pred_phys = torch.clamp(pred_phys, bound_low.to(pred_phys.device).view(1, N_target, 1),
                             bound_high.to(pred_phys.device).view(1, N_target, 1))
    return pred_phys


@torch.no_grad()
def run_eval(dataloader, model, tokenizer, device, num_channels, bound_low, bound_high, tag, save_file):
    model.eval()
    running_mse, running_mae, running_cor, running_dtw = 0.0, 0.0, 0.0, 0.0
    total_num, total_num_c = 0.0, 0.0
    for data in dataloader:
        batch = flatten_batch(data, device, num_channels)
        _, _, _, _, shape_logits, scalar_logits = forward_and_loss(model, batch, scalar_loss_weight=1.0)
        pred_phys = decode_predictions(shape_logits, scalar_logits, batch, tokenizer, bound_low, bound_high)
        pred_time = pred_phys.transpose(1, 2)              # (B, patch_len, 11)
        labels_time = batch["y_orig"][:, :, :11]            # (B, patch_len, 11) physical, target channels only

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
    save_file = open(os.path.join(args.file_save_path, f"gormpo_chronos2_{args.seed}.txt"), "w+")

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
    num_channels = tokenizer.num_channels

    bound_low, bound_high = compute_channel_bounds(args.data_root)

    dataloaders = create_gormpo_dataloader(args.data_root, args.batchsize, args.num_workers)
    train_dataloader, val_dataloader, test_dataloader = (
        dataloaders["train"], dataloaders["val"], dataloaders["test"]
    )

    model = GormpoChronos2Model(
        chronos2_model_id=args.chronos2_model_id,
        num_embeddings=args.codebook_size, num_bins=args.num_bins,
        freeze_backbone=args.freeze_backbone, device=device,
    )
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Total # trainable parameters:", sum(p.numel() for p in trainable_params),
          " / total params:", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(trainable_params, lr=args.baselr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=args.baselr, steps_per_epoch=len(train_dataloader),
        epochs=args.epochs, pct_start=0.2,
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss, running_shape_loss, running_scalar_loss, running_acc = 0.0, 0.0, 0.0, 0.0
        log_every = max(len(train_dataloader) // 3, 3)
        for i, data in enumerate(train_dataloader):
            batch = flatten_batch(data, device, num_channels)
            loss, loss_shape, loss_scalar, acc, _, _ = forward_and_loss(model, batch, args.scalar_loss_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_shape_loss += loss_shape
            running_scalar_loss += loss_scalar
            running_acc += acc
            if i % log_every == log_every - 1:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"| epoch {epoch:3d} | {i + 1:5d}/{len(train_dataloader):5d} batches | lr {lr:.6f} | "
                    f"loss {running_loss / log_every:5.4f} | shape_loss {running_shape_loss / log_every:5.4f} | "
                    f"scalar_loss {running_scalar_loss / log_every:5.4f} | code_acc {running_acc / log_every:5.4f}"
                )
                running_loss, running_shape_loss, running_scalar_loss, running_acc = 0.0, 0.0, 0.0, 0.0

        val_mse, val_mae = run_eval(val_dataloader, model, tokenizer, device, num_channels,
                                     bound_low, bound_high, "Val", save_file)
        early_stopping_counter = early_stopping(val_mse, val_mae, {"gormpo_chronos2": model})
        _ = run_eval(test_dataloader, model, tokenizer, device, num_channels,
                      bound_low, bound_high, "Test", save_file)
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
    parser.add_argument("--data_root", type=str, default="forecasting/data/gormpo_tokens_mcs_scratch_long2000")
    parser.add_argument("--tokenizer_path", type=str,
                         default="forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth")
    parser.add_argument("--chronos2_model_id", type=str, default="amazon/chronos-2")
    parser.add_argument("--codebook_size", default=256, type=int)
    parser.add_argument("--num_bins", default=32, type=int)
    parser.add_argument("--scalar_loss_weight", default=1.0, type=float)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--batchsize", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--checkpoint", action="store_true", default=True)
    parser.add_argument("--checkpoint_path", default="saved_models/gormpo_chronos2_mcs/checkpoints/", type=str)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--baselr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--file_save_path", default="results_llm/gormpo_chronos2_mcs/", type=str)
    args = parser.parse_args()

    train(args)
