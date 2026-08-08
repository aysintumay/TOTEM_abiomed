"""
Generate train/val/test .npy arrays for the abiomed dataset, in the format
Dataset_Neuro expects (num_episodes, T, num_channels), original physiological units.

Rather than re-deriving normalization/column-selection logic by hand, this goes
through abiomed_env's own WorldModel: it loads the real trained checkpoint and the
real data pkl, which gives us exactly the same column selection (self.columns) and
normalization stats (mean/std) the world model was trained with. We then reconstruct
each 12-step episode (6 in + 6 out, matching forecast_horizon=6) from the resulting
TimeSeriesDataset splits and undo the normalization, since Dataset_Neuro fits its own
StandardScaler on whatever we hand it (consistent with how every other TOTEM dataset
is prepared).

Run with: python forecasting/extract_abiomed_data.py
"""
import argparse
import os
import sys

import numpy as np
import torch

ABIOMED_ENV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ABIOMED_ENV_ROOT not in sys.path:
    sys.path.insert(0, ABIOMED_ENV_ROOT)

from abiomed_env.config import model_configs  # noqa: E402
from abiomed_env.model import WorldModel  # noqa: E402


def reconstruct_episodes(dataset):
    """Undo TimeSeriesDataset.prep_transformer_world: rebuild (N, Tin+Tout, C) windows
    from its (x, pl, labels) split, in the same normalized space World Model uses."""
    x = dataset.data  # (N, Tin, C)
    pl = dataset.pl  # (N, Tout)
    labels = dataset.labels  # (N, Tout * (C - 1))

    n, tout = pl.shape
    c = x.shape[-1]
    y = labels.reshape(n, tout, c - 1)
    y_full = torch.cat([y, pl.unsqueeze(-1)], dim=-1)  # (N, Tout, C)

    return torch.cat([x, y_full], dim=1)  # (N, Tin+Tout, C)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="10min_1hr_all_data")
    parser.add_argument("--model_path", type=str, default="/public/gormpo/models/10min_1hr_all_data_model.pth")
    parser.add_argument("--data_path", type=str, default="/public/gormpo/10min_1hr_all_data.pkl")
    parser.add_argument("--save_path", type=str, default="forecasting/data_raw/abiomed")
    args = parser.parse_args()

    cfg = model_configs[args.model_name]
    world_model = WorldModel(**cfg)
    world_model.load_model(args.model_path)
    world_model.load_data(args.data_path)

    mean_sel = world_model.mean[world_model.columns]  # (C,)
    std_sel = world_model.std[world_model.columns]  # (C,)

    os.makedirs(args.save_path, exist_ok=True)

    for split, dataset in [
        ("train", world_model.data_train),
        ("val", world_model.data_val),
        ("test", world_model.data_test),
    ]:
        episodes_normalized = reconstruct_episodes(dataset)
        episodes_original = episodes_normalized * std_sel + mean_sel
        episodes_original = episodes_original.cpu().numpy().astype(np.float32)

        out_file = os.path.join(args.save_path, f"{split}_data.npy")
        np.save(out_file, episodes_original)
        print(f"{split}: saved {episodes_original.shape} to {out_file}")


if __name__ == "__main__":
    main()
