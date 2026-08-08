"""
Finetune the GORMPO tokenizer (forecasting/lib/models/tokenizer.py) on MCS data:
frozen pretrained TOTEM VQ-VAE encoder + trainable cross-channel attention/codebook/
decoder, per Section 2.1 of the tokenizer proposal.

Run with: python train_tokenizer.py --pretrained_vqvae_path <path> --save_path <dir>
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lib.models.tokenizer import GormpoTokenizer
from lib.utils.env import seed_all_rng


def load_mcs_episode_halves(data_root, split, patch_len):
    """Like load_mcs_patches, but keeps each episode's x-half/y-half paired (needed
    for forecasting: the LLM must know which target patch follows which context
    patch, unlike tokenizer training where patches are reconstructed independently).
    Returns (x_patches, y_patches), each (num_episodes, N, k)."""
    episodes = np.load(os.path.join(data_root, f"{split}_data.npy"))
    assert episodes.shape[1] == 2 * patch_len, (
        f"expected episodes of length {2 * patch_len}, got {episodes.shape[1]}"
    )
    x_half = np.transpose(episodes[:, :patch_len, :], (0, 2, 1))  # (num_episodes, N, k)
    y_half = np.transpose(episodes[:, patch_len:, :], (0, 2, 1))
    return torch.from_numpy(x_half).float(), torch.from_numpy(y_half).float()


def load_mcs_patches(data_root, split, patch_len):
    """data_raw/abiomed/{split}_data.npy: (num_episodes, 2*patch_len, num_channels)
    raw physiological units. Splits each episode into its two patch_len-step halves
    and transposes to channel-major (N, k), matching the tokenizer's (B, N, k) input."""
    x_half, y_half = load_mcs_episode_halves(data_root, split, patch_len)
    return torch.cat([x_half, y_half], dim=0)  # (2 * num_episodes, N, k)


@torch.no_grad()
def fit_bin_quantizers(model, train_loader, device):
    """Calibrate the fixed scalar-statistic bin edges (Eq 20-23) once, from the
    whole train split, using the tokenizer's current (freshly-initialized)
    cross-channel attention for the shape-proxy min/max (Eq 17). These bins then
    stay fixed for the rest of training, per the paper's fixed-bin design."""
    model.eval()
    all_mu, all_sigma, all_min, all_max = [], [], [], []
    for (batch_x,) in train_loader:
        batch_x = batch_x.to(device)
        x_bar, mu, sigma = model.patch_and_scale(batch_x)
        x_tilde = model.cross_channel_attn(x_bar)
        all_mu.append(mu.cpu())
        all_sigma.append(sigma.cpu())
        all_min.append(x_tilde.min(dim=-1).values.cpu())
        all_max.append(x_tilde.max(dim=-1).values.cpu())

    model.q_mu.fit(torch.cat(all_mu, dim=0))
    model.q_sigma.fit(torch.cat(all_sigma, dim=0))
    model.q_min.fit(torch.cat(all_min, dim=0))
    model.q_max.fit(torch.cat(all_max, dim=0))


def train(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    seed_all_rng(args.seed)

    os.makedirs(os.path.join(args.save_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "configs"), exist_ok=True)

    train_patches = load_mcs_patches(args.data_root, "train", args.patch_len)
    val_patches = load_mcs_patches(args.data_root, "val", args.patch_len)
    print(f"train patches: {tuple(train_patches.shape)}, val patches: {tuple(val_patches.shape)}")

    train_loader = DataLoader(
        TensorDataset(train_patches), batch_size=args.batchsize, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_patches), batch_size=args.batchsize, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )

    num_channels = train_patches.shape[1]
    tokenizer_kwargs = dict(
        d_model=args.d_model, num_bins=args.num_bins,
        softdtw_gamma=args.softdtw_gamma, lambda_sdtw=args.lambda_sdtw,
        use_reward=args.use_reward,
    )
    if args.pretrained_vqvae_path:
        model = GormpoTokenizer.from_pretrained_vqvae(
            args.pretrained_vqvae_path, patch_len=args.patch_len, num_channels=num_channels,
            **tokenizer_kwargs,
        )
    else:
        # No checkpoint: a pretrained encoder's filters were tuned for plain
        # (non-cross-attended) patches, not the cross-attended ones this tokenizer
        # actually feeds it, so train the whole VQ-VAE (encoder+codebook+decoder)
        # from random init jointly with cross-channel attention instead.
        vqvae_config = dict(
            block_hidden_size=args.block_hidden_size, num_residual_layers=args.num_residual_layers,
            res_hidden_size=args.res_hidden_size, embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings, commitment_cost=args.commitment_cost,
            compression_factor=args.compression_factor,
        )
        model = GormpoTokenizer.from_scratch(
            vqvae_config, patch_len=args.patch_len, num_channels=num_channels, **tokenizer_kwargs,
        )
    model.to(device)

    if not args.pretrained_vqvae_path:
        print("Calibrating codebook init from real encoder outputs (from-scratch run)...")
        calib_batch = next(iter(train_loader))[0].to(device)
        model.calibrate_codebook(calib_batch)

    print("Fitting fixed scalar bin quantizers on the train split...")
    fit_bin_quantizers(model, train_loader, device)

    optimizer = model.configure_optimizers(lr=args.lr, encoder_lr=args.encoder_lr, vq_lr=args.vq_lr)
    total_params = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    print("Total trainable parameters:", total_params)
    for g in optimizer.param_groups:
        print(f"  group lr={g['lr']:.2e}: {sum(p.numel() for p in g['params'])} params")

    comet_logger = None
    if args.comet_log:
        import comet_ml
        comet_logger = comet_ml.Experiment(project_name=args.comet_project)
        comet_logger.log_parameters(vars(args))

    start_time = time.time()
    for epoch in range(args.num_epochs):
        model.train()
        running_loss, running_recon, running_vq, running_perplexity, n_batches = 0.0, 0.0, 0.0, 0.0, 0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            loss, out = model.shared_eval(batch_x, optimizer, "train", comet_logger=comet_logger)
            running_loss += loss.item()
            running_recon += torch.nn.functional.mse_loss(out["x_hat"], batch_x).item()
            running_vq += out["vq_loss"].item()
            running_perplexity += out["perplexity"].item()
            n_batches += 1

        print(
            f"| epoch {epoch:3d} | train loss {running_loss / n_batches:.4f} "
            f"| recon {running_recon / n_batches:.4f} | vq {running_vq / n_batches:.4f} "
            f"| perplexity {running_perplexity / n_batches:.2f}"
        )

        if epoch % args.val_every == 0 or epoch == args.num_epochs - 1:
            model.eval()
            val_loss, val_recon, val_perplexity, n_val = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for (batch_x,) in val_loader:
                    batch_x = batch_x.to(device)
                    loss, out = model.shared_eval(batch_x, optimizer, "val", comet_logger=comet_logger)
                    val_loss += loss.item()
                    val_recon += torch.nn.functional.mse_loss(out["x_hat"], batch_x).item()
                    val_perplexity += out["perplexity"].item()
                    n_val += 1
            print(
                f"| [Val] epoch {epoch:3d} | loss {val_loss / n_val:.4f} | recon {val_recon / n_val:.4f} "
                f"| perplexity {val_perplexity / n_val:.2f}"
            )

        if epoch % args.save_every == 0 or epoch == args.num_epochs - 1:
            torch.save(model, os.path.join(args.save_path, "checkpoints", f"tokenizer_epoch_{epoch}.pth"))

    torch.save(model, os.path.join(args.save_path, "checkpoints", "final_tokenizer.pth"))

    bin_edges = {
        name: getattr(model, name).edges.tolist()
        for name in ("q_mu", "q_sigma", "q_min", "q_max")
    }
    with open(os.path.join(args.save_path, "configs", "bin_edges.json"), "w") as f:
        json.dump(bin_edges, f, indent=2)
    with open(os.path.join(args.save_path, "configs", "config_file.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Total training time: {time.time() - start_time:.1f}s")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_vqvae_path", type=str, default="",
                        help="if given, warm-starts (or freezes, see --freeze_encoder) the "
                             "encoder/codebook/decoder from this checkpoint. If omitted "
                             "(default), the whole VQ-VAE is trained from random init "
                             "jointly with cross-channel attention -- see vqvae architecture "
                             "args below.")
    parser.add_argument("--data_root", type=str, default="data_raw/abiomed",
                        help="folder with {train,val,test}_data.npy, (episodes, 2*patch_len, num_channels)")
    parser.add_argument("--save_path", type=str, default="saved_models/gormpo_tokenizer")
    parser.add_argument("--patch_len", type=int, default=6, help="k in the paper")
    parser.add_argument("--block_hidden_size", type=int, default=128, help="vqvae arch (only used if training from scratch)")
    parser.add_argument("--num_residual_layers", type=int, default=2, help="vqvae arch (only used if training from scratch)")
    parser.add_argument("--res_hidden_size", type=int, default=64, help="vqvae arch (only used if training from scratch)")
    parser.add_argument("--embedding_dim", type=int, default=64, help="vqvae arch (only used if training from scratch)")
    parser.add_argument("--num_embeddings", type=int, default=256, help="codebook size (only used if training from scratch)")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="VQ commitment cost (only used if training from scratch)")
    parser.add_argument("--compression_factor", type=int, default=2, help="vqvae compression factor (only used if training from scratch)")
    parser.add_argument("--d_model", type=int, default=32, help="cross-channel attention hidden dim")
    parser.add_argument("--num_bins", type=int, default=32, help="bins per scalar quantizer (Eq 20-23)")
    parser.add_argument("--lambda_sdtw", type=float, default=1.0, help="lambda in L_recon (Eq 31)")
    parser.add_argument("--softdtw_gamma", type=float, default=0.1)
    parser.add_argument("--no_reward", dest="use_reward", action="store_false",
                        help="disable reward tokens (Eq 24) for datasets without MAP/HR/Pulsatility channels")
    parser.add_argument("--lr", type=float, default=1e-3, help="LR for cross_channel_attn + decoder")
    parser.add_argument("--encoder_lr", type=float, default=None, help="LR for the encoder (defaults to --lr); only used if training from scratch")
    parser.add_argument("--vq_lr", type=float, default=None, help="LR for the VQ codebook (defaults to --lr)")
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--comet_log", action="store_true")
    parser.add_argument("--comet_project", type=str, default="gormpo-tokenizer")
    args = parser.parse_args()

    train(args)
