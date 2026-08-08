"""
Evaluate a finetuned GORMPO tokenizer checkpoint (see train_tokenizer.py) on the MCS
test split:
  1. Reconstruction fidelity (Axiom 5): ||D(T(x)) - x||_2, mean and max, per channel,
     compared against a plain-TOTEM ablation (same frozen encoder/decoder, no
     cross-channel attention, no scalar/reward split) -- the "TOTEM" row of Table 1.
  2. Codebook utilization / perplexity.
  3. A cheap Axiom-3 (phase awareness) probe: shift each patch by tau != 0 timesteps
     and check that the shape code changes.

Run with: python eval_tokenizer.py --tokenizer_path <checkpoint> --pretrained_vqvae_path <checkpoint>
"""
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from train_tokenizer import load_mcs_patches


@torch.no_grad()
def eval_tokenizer(model, loader, device):
    total_sq_err = None  # (N,) running sum of squared error per channel
    max_err = None  # (N,) running max L2 error per channel
    total_n = 0
    perplexities = []

    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        out = model.forward(batch_x)
        err = out["x_hat"] - batch_x  # (B, N, k)
        sq_err = (err ** 2).sum(dim=-1)  # (B, N)
        l2_err = sq_err.sqrt()  # (B, N)

        batch_sum = sq_err.sum(dim=0)
        batch_max = l2_err.max(dim=0).values
        total_sq_err = batch_sum if total_sq_err is None else total_sq_err + batch_sum
        max_err = batch_max if max_err is None else torch.maximum(max_err, batch_max)
        total_n += batch_x.shape[0]
        perplexities.append(out["perplexity"].item())

    mean_l2_per_channel = (total_sq_err / total_n).sqrt()
    return {
        "mean_l2_per_channel": mean_l2_per_channel.cpu(),
        "max_l2_per_channel": max_err.cpu(),
        "mean_l2_overall": mean_l2_per_channel.mean().item(),
        "max_l2_overall": max_err.max().item(),
        "perplexity": sum(perplexities) / len(perplexities),
    }


@torch.no_grad()
def eval_plain_totem_baseline(pretrained_vqvae, loader, device, patch_len):
    """Ablation: normalize per-channel-patch, encode/quantize/decode with the frozen
    generalist VQ-VAE directly (no cross-channel attention, no scalar/reward split),
    then affine-restore. Matches the TOTEM row of Table 1."""
    pretrained_vqvae = pretrained_vqvae.to(device)
    pretrained_vqvae.eval()

    total_sq_err, max_err, total_n = None, None, 0
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        B, N, k = batch_x.shape
        mu = batch_x.mean(dim=-1, keepdim=True)
        sigma = torch.sqrt(batch_x.var(dim=-1, keepdim=True, unbiased=False) + 1e-5)
        x_bar = (batch_x - mu) / sigma

        flat = x_bar.reshape(B * N, k)
        z = pretrained_vqvae.encoder(flat, pretrained_vqvae.compression_factor)
        _, quantized, _, _, _, _ = pretrained_vqvae.vq(z)
        x_hat_norm = pretrained_vqvae.decoder(quantized, pretrained_vqvae.compression_factor).view(B, N, k)
        x_hat = sigma.squeeze(-1).unsqueeze(-1) * x_hat_norm + mu

        err = x_hat - batch_x
        sq_err = (err ** 2).sum(dim=-1)
        l2_err = sq_err.sqrt()
        total_sq_err = sq_err.sum(dim=0) if total_sq_err is None else total_sq_err + sq_err.sum(dim=0)
        max_err = l2_err.max(dim=0).values if max_err is None else torch.maximum(max_err, l2_err.max(dim=0).values)
        total_n += B

    mean_l2_per_channel = (total_sq_err / total_n).sqrt()
    return {
        "mean_l2_per_channel": mean_l2_per_channel.cpu(),
        "max_l2_per_channel": max_err.cpu(),
        "mean_l2_overall": mean_l2_per_channel.mean().item(),
        "max_l2_overall": max_err.max().item(),
    }


@torch.no_grad()
def phase_awareness_probe(model, loader, device, tau=1):
    """Axiom 3: shifting a patch by tau != 0 must change its shape code."""
    changed, total = 0, 0
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        shifted = torch.roll(batch_x, shifts=tau, dims=-1)

        q_shape = model.encode(batch_x)["q_shape"]
        q_shape_shifted = model.encode(shifted)["q_shape"]

        changed += (q_shape != q_shape_shifted).any(dim=-1).sum().item()
        total += q_shape.shape[0] * q_shape.shape[1]
    return changed / total


def main(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"

    model = torch.load(args.tokenizer_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    test_patches = load_mcs_patches(args.data_root, "test", model.patch_len)
    loader = DataLoader(TensorDataset(test_patches), batch_size=args.batchsize, shuffle=False)

    print("Evaluating GORMPO tokenizer on test split...")
    gormpo_results = eval_tokenizer(model, loader, device)
    print(f"  mean L2/channel:    {gormpo_results['mean_l2_per_channel']}")
    print(f"  max L2/channel:     {gormpo_results['max_l2_per_channel']}")
    print(f"  overall mean/max L2: {gormpo_results['mean_l2_overall']:.4f} / {gormpo_results['max_l2_overall']:.4f}")
    print(f"  perplexity: {gormpo_results['perplexity']:.2f}")

    if args.pretrained_vqvae_path:
        print("\nEvaluating plain-TOTEM ablation baseline on the same patches...")
        pretrained_vqvae = torch.load(args.pretrained_vqvae_path, map_location=device, weights_only=False)
        totem_results = eval_plain_totem_baseline(pretrained_vqvae, loader, device, model.patch_len)
        print(f"  mean L2/channel:    {totem_results['mean_l2_per_channel']}")
        print(f"  overall mean/max L2: {totem_results['mean_l2_overall']:.4f} / {totem_results['max_l2_overall']:.4f}")

    print("\nAxiom-3 phase-awareness probe (fraction of channel-patches whose shape code changes under a shift)...")
    for tau in (1, 2, 3):
        frac = phase_awareness_probe(model, loader, device, tau=tau)
        print(f"  tau={tau}: {frac:.4f} changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="saved_models/gormpo_tokenizer/checkpoints/final_tokenizer.pth")
    parser.add_argument("--pretrained_vqvae_path", type=str,
                        default="saved_models/all_mcs/CD64_CW256_CF2_BS4096_ITR15000/checkpoints/final_model.pth")
    parser.add_argument("--data_root", type=str, default="data_raw/abiomed")
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
