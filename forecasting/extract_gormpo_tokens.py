"""
Run a trained GORMPO tokenizer (see train_tokenizer.py) over the MCS train/val/test
splits and save the full per-channel token schema, paired by episode (x-half =
context, y-half = target), for train_gormpo_llm_forecaster.py to consume.

Mirrors extract_forecasting_data.py's role in the existing TOTEM+LLM pipeline, but
uses GormpoTokenizer.encode() (cross-channel attention + scale/reward tokens)
instead of RevIN + a plain per-channel vqvae.

Run with: python extract_gormpo_tokens.py --tokenizer_path <checkpoint> --save_path <dir>
"""
import argparse
import os

import numpy as np
import torch

from train_tokenizer import load_mcs_episode_halves

# Three per-patch reward fields (Eq 24-25), replacing the old single per-channel-masked
# q_r -- context-only, never a target/prediction label (same reasoning as before, now
# applied to all three).
REWARD_KEYS = ("q_r_map", "q_r_hr", "q_r_pulsat")


@torch.no_grad()
def encode_split(model, data_root, split, patch_len, device, batch_size=512):
    x_patches, y_patches = load_mcs_episode_halves(data_root, split, patch_len)
    num_episodes = x_patches.shape[0]

    keys = ["q_shape", "q_mu", "q_sigma", "q_min", "q_max", *REWARD_KEYS]
    x_acc = {k: [] for k in keys}
    y_acc = {k: [] for k in keys}
    x_orig, y_orig = [], []

    for start in range(0, num_episodes, batch_size):
        x_batch = x_patches[start:start + batch_size].to(device)
        y_batch = y_patches[start:start + batch_size].to(device)

        x_out = model.encode(x_batch)
        y_out = model.encode(y_batch)

        x_orig.append(x_batch.cpu().numpy())
        y_orig.append(y_batch.cpu().numpy())
        for k in keys:
            x_acc[k].append(x_out[k].cpu().numpy())
            y_acc[k].append(y_out[k].cpu().numpy())

    result = {
        # (num_episodes, patch_len, N) time-major, matching the repo-wide x/y_original convention
        "x_original": np.concatenate(x_orig, axis=0).transpose(0, 2, 1),
        "y_original": np.concatenate(y_orig, axis=0).transpose(0, 2, 1),
    }
    for k in keys:
        x_full = np.concatenate(x_acc[k], axis=0)
        y_full = np.concatenate(y_acc[k], axis=0)
        if k == "q_shape":
            # (num_episodes, N, L) -> (num_episodes, L, N), matching x_codes/y_codes convention
            x_full = x_full.transpose(0, 2, 1)
            y_full = y_full.transpose(0, 2, 1)
        # the three reward keys are (num_episodes,) -- one value per patch, not per
        # channel like q_shape/q_mu/etc -- nothing to transpose, saved as-is.
        result[f"x_{k}"] = x_full
        if k not in REWARD_KEYS:  # reward is context-only, never a target/prediction label
            result[f"y_{k}"] = y_full
    return result


def main(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    model = torch.load(args.tokenizer_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    os.makedirs(args.save_path, exist_ok=True)

    name_map = {
        "q_shape": "codes", "q_mu": "mu", "q_sigma": "sigma", "q_min": "min", "q_max": "max",
        "q_r_map": "r_map", "q_r_hr": "r_hr", "q_r_pulsat": "r_pulsat",
    }
    for split in ("train", "val", "test"):
        print(f"Extracting {split}...")
        out = encode_split(model, args.data_root, split, model.patch_len, device, args.batch_size)
        np.save(os.path.join(args.save_path, f"{split}_x_original.npy"), out["x_original"])
        np.save(os.path.join(args.save_path, f"{split}_y_original.npy"), out["y_original"])
        for q_name, short in name_map.items():
            np.save(os.path.join(args.save_path, f"{split}_x_{short}.npy"), out[f"x_{q_name}"])
            if q_name not in REWARD_KEYS:
                np.save(os.path.join(args.save_path, f"{split}_y_{short}.npy"), out[f"y_{q_name}"])
        print(f"  x_original {out['x_original'].shape}, x_codes {out['x_q_shape'].shape}, "
              f"x_mu {out['x_q_mu'].shape}, x_r_map {out['x_q_r_map'].shape}")

    codebook = model.vq._embedding.weight.detach().cpu().numpy()
    np.save(os.path.join(args.save_path, "codebook.npy"), codebook)
    print(f"codebook {codebook.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="saved_models/gormpo_tokenizer_mcs/checkpoints/final_tokenizer.pth")
    parser.add_argument("--data_root", type=str, default="data_raw/abiomed")
    parser.add_argument("--save_path", type=str, default="data/gormpo_tokens_mcs")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
