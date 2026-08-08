"""
Generate raw (unscaled) ETTh1 episode windows for the GORMPO tokenizer's ETTh1
specialist run, matching the format train_tokenizer.py's load_mcs_episode_halves
expects: {split}_data.npy of shape (num_windows, 2*patch_len, num_channels),
physiological/original units, x-half then y-half back to back.

Mirrors extract_abiomed_data.py's role for MCS, but pulls straight from
Dataset_ETT_hour (data_provider/data_loader.py) with scale=False -- unlike
save_revin_data.py/extract_forecasting_data.py, this must NOT go through
Dataset_ETT_hour's default StandardScaler, since the tokenizer computes its own
per-patch normalization (Eq 7-9) and needs true raw units as input, exactly as
data_raw/abiomed/{split}_data.npy already is.

Uses seq_len=pred_len=patch_len, label_len=0, so each window's x-half and y-half
are back-to-back non-overlapping halves of one 2*patch_len episode, matching the
MCS convention. patch_len=96 matches the ETTh1 specialist VQVAE checkpoint
(saved_models/ETTh1/CD64_CW256_CF4_BS4096_ITR15000, compression_factor=4).

Run with: python extract_ett_data.py --save_path data_raw/ETTh1_specialist --patch_len 96
"""
import argparse
import os

import numpy as np
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_ETT_hour


def extract_split(root_path, data_path, split, patch_len):
    dataset = Dataset_ETT_hour(
        root_path=root_path, data_path=data_path, flag=split,
        size=[patch_len, 0, patch_len], features="M", scale=False,
    )
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=2)
    episodes = []
    for seq_x, seq_y, _, _ in loader:
        episodes.append(np.concatenate([seq_x.numpy(), seq_y.numpy()], axis=1))  # (b, 2*patch_len, N)
    return np.concatenate(episodes, axis=0)


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    for split in ("train", "val", "test"):
        episodes = extract_split(args.root_path, args.data_path, split, args.patch_len)
        out_file = os.path.join(args.save_path, f"{split}_data.npy")
        np.save(out_file, episodes.astype(np.float32))
        print(f"{split}: saved {episodes.shape} to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="data_raw/ETTh1")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--save_path", type=str, default="data_raw/ETTh1_specialist")
    parser.add_argument("--patch_len", type=int, default=96)
    args = parser.parse_args()
    main(args)
