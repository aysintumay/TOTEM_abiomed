"""
Evaluate the fine-tuned GORMPO+Chronos model (train_gormpo_chronos.py) on the real MCS
test split. Metrics (MAE, MSE, DTW, Pearson correlation) are computed in normalized
(z-score) space using /public/gormpo/10min_1hr_all_data.pkl's mean/std -- same
convention as eval_gormpo_world_model.py / eval_chronos_world_model.py, so results are
directly comparable across every world-model variant evaluated in this project
(medllama zero-shot, medllama few-shot, GORMPO few-shot, zero-shot Chronos, and this
fine-tuned GORMPO+Chronos model).

The model predicts all 12 channels, including Pump Level (see train_gormpo_chronos.py's
docstring for why), but the headline per-variable table and "Overall" average only
cover the 11 forecast variables, matching every other eval script in this project;
Pump Level's own row is reported separately as a secondary diagnostic, not folded into
the comparable "Overall" number.

Run with: python eval_gormpo_chronos.py --num_examples 100
"""
import argparse
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSeq2SeqLM

sys.path.insert(0, "forecasting")
from train_tokenizer import load_mcs_episode_halves  # noqa: E402
from lib.models.metrics import dtw, pearsoncor  # noqa: E402

from chronos import ChronosModel  # noqa: E402
from train_gormpo_chronos import build_tokenizer  # noqa: E402


FEATURE_NAMES = [
    "PumpPressure", "PumpSpeed", "PumpFlow", "LVP", "LVEDP", "SYSTOLIC",
    "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate", "ESE_lv", "Pump Level",
]


def compute_channel_bounds(data_root, patch_len, low_pct=1, high_pct=99):
    """Per-channel (low, high) physical-unit clipping bounds from the train split's
    raw values -- same role as gormpo_world_model.py's compute_variable_ylims, needed
    here for the same reason: nothing in the tokenizer's decoder inherently bounds its
    output range, and a bad shape-code/scale-bin combination can decode to a physically
    impossible value (observed: SYSTOLIC to roughly -830..1090 mmHg against a true
    range of 65..113) that dominates an MSE average otherwise. Model-independent --
    only depends on the training data distribution."""
    x_train, _ = load_mcs_episode_halves(data_root, "train", patch_len)  # (E, N, k)
    flat = x_train.permute(0, 2, 1).reshape(-1, x_train.shape[1])  # (E*k, N) -- N last, percentile per channel
    low = torch.quantile(flat.float(), low_pct / 100, dim=0)
    high = torch.quantile(flat.float(), high_pct / 100, dim=0)
    return low, high  # (N,), (N,)


def load_mean_std(pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    mean, std = data["mean"], data["std"]
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)
    columns = [i for i in range(13) if i != 11]  # matches FEATURE_NAMES order project-wide
    return mean[columns], std[columns]  # (12,) each


def main(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"

    tokenizer, chronos_config = build_tokenizer(args.tokenizer_path, device)
    print(
        f"L={tokenizer.L}  label_len={tokenizer.label_len}  context_len={tokenizer.context_len}  "
        f"num_channels={tokenizer.gormpo_tokenizer.num_channels}  combined vocab={tokenizer.vocab_size}"
    )
    chronos_config.num_samples = args.num_samples
    chronos_config.temperature = args.temperature

    print(f"Loading fine-tuned checkpoint: {args.checkpoint_dir}")
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_dir).to(device)
    hf_model.eval()
    chronos_model = ChronosModel(config=chronos_config, model=hf_model)

    mean, std = load_mean_std(args.mcs_pickle_path)  # (12,), (12,) -- FEATURE_NAMES order
    bound_low, bound_high = compute_channel_bounds(args.data_root, args.patch_len)
    bound_low, bound_high = bound_low.to(device), bound_high.to(device)

    x_test, y_test = load_mcs_episode_halves(args.data_root, "test", args.patch_len)
    n = min(args.num_examples, x_test.shape[0])
    rng = np.random.RandomState(args.random_seed)
    indices = rng.choice(x_test.shape[0], size=n, replace=False) if args.random_sample else np.arange(n)
    x_sel, y_sel = x_test[indices], y_test[indices]
    print(f"Evaluating on {n} test episodes (of {x_test.shape[0]})")

    loader = DataLoader(TensorDataset(x_sel, y_sel), batch_size=args.batch_size, shuffle=False)

    m = mean.to(device)
    s = std.to(device)
    all_pred_norm, all_true_norm = [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # (b, N, k) each
            b, N, k = x_batch.shape

            context_ids, context_mask, state = tokenizer.context_input_transform(x_batch)  # (b*N, context_len)
            samples = chronos_model(
                input_ids=context_ids, attention_mask=context_mask.long(),
                prediction_length=tokenizer.label_len, num_samples=args.num_samples,
            )  # (b*N, num_samples, label_len)
            decoded = tokenizer.output_transform(samples, state)  # (b*N, num_samples, k) physical
            # row order is b*N with channel = row % N (context_input_transform's B,N -> B*N
            # flatten is row-major, b varies slower) -- tile the per-channel bounds to match.
            row_low = bound_low.repeat(b).view(b * N, 1, 1)
            row_high = bound_high.repeat(b).view(b * N, 1, 1)
            decoded = torch.clamp(decoded, row_low, row_high)
            pred_phys = decoded.mean(dim=1).view(b, N, k)  # (b, N, k) physical, mean over samples

            pred_norm = (pred_phys - m.view(1, N, 1)) / s.view(1, N, 1)
            true_norm = (y_batch - m.view(1, N, 1)) / s.view(1, N, 1)

            all_pred_norm.append(pred_norm.cpu())
            all_true_norm.append(true_norm.cpu())

    pred = torch.cat(all_pred_norm, dim=0).permute(0, 2, 1)  # (n, k, N) -- time-major, matching other eval scripts
    true = torch.cat(all_true_norm, dim=0).permute(0, 2, 1)

    mae = (pred - true).abs().mean(dim=(0, 1))   # (N,)
    mse = (pred - true).pow(2).mean(dim=(0, 1))
    corr = torch.tensor([
        pearsoncor(pred[:, :, j:j + 1], true[:, :, j:j + 1], reduction="mean").item()
        for j in range(pred.shape[-1])
    ])
    dtw_dist = torch.tensor([
        dtw(pred[:, :, j:j + 1], true[:, :, j:j + 1], reduction="mean").item()
        for j in range(pred.shape[-1])
    ])

    print("\n=== Per-variable MAE / MSE / DTW / Pearson corr (normalized/z-score units, averaged over all samples) ===")
    for j in range(11):  # 11 forecast variables, matching every other eval script's headline table
        print(f"  {FEATURE_NAMES[j]:<14} MAE={mae[j]:7.4f}  MSE={mse[j]:7.4f}  DTW={dtw_dist[j]:7.4f}  corr={corr[j]:7.4f}")
    print(
        f"\nOverall (mean across 11 forecast variables): "
        f"MAE={mae[:11].mean():.4f}  MSE={mse[:11].mean():.4f}  DTW={dtw_dist[:11].mean():.4f}  corr={corr[:11].mean():.4f}"
    )
    print(
        f"\n[secondary] Pump Level (channel 12, predicted but not a forecast target -- see module docstring): "
        f"MAE={mae[11]:7.4f}  MSE={mse[11]:7.4f}  DTW={dtw_dist[11]:7.4f}  corr={corr[11]:7.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="saved_models/gormpo_chronos/checkpoints/epoch_60",
                         help="epoch_60 is the lowest-val-loss checkpoint (2.3836) from the 80-epoch run, "
                              "before val loss started rising past epoch ~62")
    parser.add_argument(
        "--tokenizer_path", type=str,
        default="forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth",
    )
    parser.add_argument("--data_root", type=str, default="forecasting/data_raw/abiomed")
    parser.add_argument("--mcs_pickle_path", type=str, default="/public/gormpo/10min_1hr_all_data.pkl")
    parser.add_argument("--patch_len", type=int, default=6)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--random_sample", action="store_true", help="draw num_examples uniformly at random instead of the first N")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
