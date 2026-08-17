"""
Convert MCS data into GluonTS JSON-Lines format for chronos-forecasting's own
scripts/training/train.py pipeline (FileDataset + MeanScaleUniformBins + HF Trainer),
so we can fine-tune Chronos with its native tokenizer as a comparison point against
train_gormpo_chronos.py's GormpoChronosTokenizer swap.

Uses the exact same episodes/split as train_gormpo_chronos.py (forecasting/
train_tokenizer.py's load_mcs_episode_halves) and the same "all 12 channels including
Pump Level, one channel = one independent series" convention, so the two Chronos
fine-tunes differ only in tokenizer + training loop implementation, not in what data
they saw.

Each (episode, channel) pair becomes one 12-value univariate series (6 raw context
steps + 6 raw label steps, un-normalized physical units -- MeanScaleUniformBins does
its own per-series mean scaling, so it doesn't need pre-normalized input the way our
z-score-based scripts do). "start" is a dummy shared timestamp; only within-series
order matters here, not any real calendar alignment across items.

Run with: python prepare_chronos_native_data.py
"""
import json
import os
import sys

sys.path.insert(0, "forecasting")
from train_tokenizer import load_mcs_episode_halves  # noqa: E402

OUT_DIR = "chronos_gluonts_data"
DATA_ROOT = "forecasting/data_raw/abiomed"
PATCH_LEN = 6
FREQ = "10min"
START = "2020-01-01 00:00:00"


def write_split(split):
    x_half, y_half = load_mcs_episode_halves(DATA_ROOT, split, PATCH_LEN)  # (E, N=12, k=6) each
    episodes = x_half.shape[0]
    num_channels = x_half.shape[1]

    split_dir = os.path.join(OUT_DIR, split)
    os.makedirs(split_dir, exist_ok=True)
    path = os.path.join(split_dir, "data.jsonl")

    with open(path, "w") as f:
        for e in range(episodes):
            for c in range(num_channels):
                target = x_half[e, c].tolist() + y_half[e, c].tolist()  # 12 raw physical values
                f.write(json.dumps({"start": START, "target": target}) + "\n")

    print(f"{split}: {episodes} episodes x {num_channels} channels = {episodes * num_channels} series -> {path}")


if __name__ == "__main__":
    write_split("train")
    write_split("val")
