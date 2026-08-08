"""
Train GormpoTransformerForecaster (lib/models/gormpo_transformer_forecaster.py) on
MCS tokens extracted by extract_gormpo_tokens.py -- the non-LLM counterpart to
train_gormpo_llm_forecaster.py. No teacher-forcing/masking and no autoregressive
generation loop: one forward pass over the context predicts all TCout target shape
codes + the target patch's scale bins directly.

Reuses create_gormpo_dataloader/_unpack/SCALAR_NAMES from
train_gormpo_llm_forecaster.py (same data schema, same flatten-channels convention).

Run with: python train_gormpo_transformer_forecaster.py --data_root <extracted tokens dir> \
    --tokenizer_path <trained tokenizer checkpoint>
"""
import argparse
import os

import torch
import torch.nn.functional as F

from lib.models.gormpo_llm_forecaster import flatten_channels_scalar, unflatten_channels_scalar
from lib.models.gormpo_transformer_forecaster import GormpoTransformerForecaster
from lib.models.llm_forecaster import flatten_channels
from lib.models.metrics import dtw, pearsoncor
from lib.utils.checkpoint import EarlyStopping
from lib.utils.env import seed_all_rng
from train_gormpo_llm_forecaster import SCALAR_NAMES, _unpack, create_gormpo_dataloader


def train_one_epoch(dataloader, model, optimizer, epoch, device, scalar_loss_weight):
    running_loss, running_shape_loss, running_scalar_loss, running_acc = 0.0, 0.0, 0.0, 0.0
    log_every = max(len(dataloader) // 3, 3)

    for i, data in enumerate(dataloader):
        (_, _, x_codes, y_codes, x_mu, x_sigma, x_min, x_max, x_r, y_mu, y_sigma, y_min, y_max) = _unpack(data, device)

        x_ids = flatten_channels(x_codes)  # (B*N, TCin)
        y_ids = flatten_channels(y_codes)  # (B*N, TCout)
        x_mu_f, x_sigma_f, x_min_f, x_max_f, x_r_f = (
            flatten_channels_scalar(t) for t in (x_mu, x_sigma, x_min, x_max, x_r)
        )
        y_scalars_f = {name: flatten_channels_scalar(t) for name, t in zip(SCALAR_NAMES, (y_mu, y_sigma, y_min, y_max))}

        shape_logits, scalars_pred = model(x_ids, x_mu_f, x_sigma_f, x_min_f, x_max_f, x_r_f)
        loss_shape = F.cross_entropy(shape_logits.reshape(-1, shape_logits.shape[-1]), y_ids.reshape(-1))
        loss_scalar = sum(F.cross_entropy(scalars_pred[name], y_scalars_f[name]) for name in SCALAR_NAMES)
        loss = loss_shape + scalar_loss_weight * loss_scalar

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = (shape_logits.argmax(dim=-1) == y_ids).float().mean().item()

        running_loss += loss.item()
        running_shape_loss += loss_shape.item()
        running_scalar_loss += loss_scalar.item()
        running_acc += acc
        if i % log_every == log_every - 1:
            print(
                f"| epoch {epoch:3d} | {i + 1:5d}/{len(dataloader):5d} batches | "
                f"loss {running_loss / log_every:5.4f} | shape_loss {running_shape_loss / log_every:5.4f} | "
                f"scalar_loss {running_scalar_loss / log_every:5.4f} | code_acc {running_acc / log_every:5.4f}"
            )
            running_loss, running_shape_loss, running_scalar_loss, running_acc = 0.0, 0.0, 0.0, 0.0


@torch.no_grad()
def inference(data, model, tokenizer, device):
    """Returns (pred_time, labels_time), both (B, patch_len, N) raw physiological units."""
    (x_orig, y_orig, x_codes, y_codes, x_mu, x_sigma, x_min, x_max, x_r, _, _, _, _) = _unpack(data, device)
    B, TCin, N = x_codes.shape

    x_ids = flatten_channels(x_codes)
    x_mu_f, x_sigma_f, x_min_f, x_max_f, x_r_f = (
        flatten_channels_scalar(t) for t in (x_mu, x_sigma, x_min, x_max, x_r)
    )

    shape_logits, scalars_pred = model(x_ids, x_mu_f, x_sigma_f, x_min_f, x_max_f, x_r_f)
    y_ids_pred = shape_logits.argmax(dim=-1)  # (B*N, TCout), one forward pass, no rollout

    mu_idx = unflatten_channels_scalar(scalars_pred["mu"].argmax(dim=-1), B, N)
    sigma_idx = unflatten_channels_scalar(scalars_pred["sigma"].argmax(dim=-1), B, N)
    mu_hat = tokenizer.q_mu.dequantize(mu_idx)
    sigma_hat = tokenizer.q_sigma.dequantize(sigma_idx)

    num_code_words, code_dim = tokenizer.vq._embedding.weight.shape
    one_hot = F.one_hot(y_ids_pred, num_code_words).to(tokenizer.vq._embedding.weight.dtype)
    quantized = torch.matmul(one_hot, tokenizer.vq._embedding.weight).transpose(1, 2)

    x_hat, _ = tokenizer.decode(quantized, mu_hat, sigma_hat)
    pred_time = x_hat.transpose(1, 2)
    return pred_time, y_orig


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
    save_file = open(os.path.join(args.file_save_path, f"gormpo_transformer_{args.seed}.txt"), "w+")

    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_all_rng(None if args.seed < 0 else args.seed)

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

    sample_x_codes = next(iter(train_dataloader))[2]
    sample_y_codes = next(iter(train_dataloader))[3]
    TCin, TCout = sample_x_codes.shape[1], sample_y_codes.shape[1]

    model = GormpoTransformerForecaster(
        num_embeddings=args.codebook_size, num_bins=args.num_bins, TCin=TCin, TCout=TCout,
        d_model=args.d_model, nhead=args.nhead, d_hid=args.d_hid, nlayers=args.nlayers,
        dropout=args.dropout,
    )
    model.to(device)
    print("Total trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = model.configure_optimizers(lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        train_one_epoch(train_dataloader, model, optimizer, epoch, device, args.scalar_loss_weight)

        val_mse, val_mae = run_eval(val_dataloader, model, tokenizer, device, "Val", save_file)
        early_stopping_counter = early_stopping(val_mse, val_mae, {"gormpo_transformer_forecaster": model})
        run_eval(test_dataloader, model, tokenizer, device, "Test", save_file)
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
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--checkpoint_path", default="saved_models/gormpo_transformer_mcs/checkpoints/", type=str)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--d_model", default=128, type=int)
    parser.add_argument("--nhead", default=4, type=int)
    parser.add_argument("--d_hid", default=256, type=int)
    parser.add_argument("--nlayers", default=4, type=int)
    parser.add_argument("--file_save_path", default="results_llm/gormpo_transformer_mcs/", type=str)
    args = parser.parse_args()

    train(args)
