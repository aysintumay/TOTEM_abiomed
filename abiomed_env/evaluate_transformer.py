'''
python abiomed_env/evaluate_transformer.py
Evaluates the Transformer world model on the test set with 10 forward passes,
computing MSE, MAE, and CRPS (matching the LLM evaluation protocol).
'''
import sys
import os
import torch
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from model import WorldModel
from config import model_kwargs_10min_1hr_full

DATA_PATH  = "/public/gormpo/10min_1hr_all_data.pkl"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "10min_1hr_all_data_model.pth")
DEVICE     = "cuda:0"
NUM_SAMPLES = 10


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, obs: np.ndarray) -> float:
    sigma = np.maximum(sigma, 1e-8)
    z = (obs - mu) / sigma
    return float((sigma * (z * (2 * stats.norm.cdf(z) - 1)
                           + 2 * stats.norm.pdf(z)
                           - 1 / np.sqrt(np.pi))).mean())


def get_transformer_confidence(model_path=None):
    model_kwargs = dict(model_kwargs_10min_1hr_full)
    model_kwargs["device"] = DEVICE
    model = WorldModel(**model_kwargs)
    model.load_data(DATA_PATH)
    model.load_model(model_path or MODEL_PATH)
    model.model.eval()
    # print(f"Test set size: {len(model.data_test)}")

    # ── run 10-sample inference ───────────────────────────────────────────────────

    # print(f"Running test_output_multiple(num_samples={NUM_SAMPLES}) ...")
    outputs, ys, pls_list = model.test_output_multiple(num_samples=NUM_SAMPLES)

    # outputs : list of (B, num_samples, horizon, 11) tensors
    # ys      : list of (B, horizon, 11) tensors
    # pls_list: list of (B, horizon) tensors — normalized P-level in the output window
    all_preds = torch.cat(outputs,   dim=0).cpu().numpy()   # (N, 10, horizon, 11)
    all_ys    = torch.cat(ys,        dim=0).cpu().numpy()   # (N, horizon, 11)
    all_pls   = torch.cat(pls_list,  dim=0).cpu().numpy()   # (N, horizon)

    print(f"Predictions shape: {all_preds.shape}")
    print(f"Ground truth shape: {all_ys.shape}")

    mean_pred = all_preds.mean(axis=1)   # (N, horizon, 11)
    std_pred  = all_preds.std(axis=1)    # (N, horizon, 11)

    map_std = model.std[model.columns[0]].item()   # MAP std in mmHg

    return mean_pred, std_pred, all_ys, all_pls, map_std

def get_tranformer_metrics(mean_pred, std_pred, all_ys):

    mse_per  = ((mean_pred - all_ys) ** 2).mean(axis=(1, 2))   # (N,)
    mae_per  = np.abs(mean_pred - all_ys).mean(axis=(1, 2))    # (N,)
    crps_per = np.array([
        crps_gaussian(mean_pred[[i]], std_pred[[i]], all_ys[[i]])
        for i in range(len(all_ys))
    ])                                                           # (N,)

    print(f"\n=== Transformer test set metrics (n={len(all_ys)}, normalized space) ===")
    print(f"MSE:  {mse_per.mean():.6f} ± {mse_per.std():.6f}")
    print(f"MAE:  {mae_per.mean():.6f} ± {mae_per.std():.6f}")
    print(f"CRPS: {crps_per.mean():.6f} ± {crps_per.std():.6f}")


def _classify_pl(all_pls):
    """True = static (P-level constant across the 1-hr output window)."""
    return np.all(all_pls == all_pls[:, :1], axis=1)  # (N,)


def _map_slope(map_seq):
    """Closed-form per-sample linear regression slope. map_seq: (N, H)."""
    H = map_seq.shape[1]
    t = np.arange(H, dtype=float)
    t -= t.mean()
    denom = (t ** 2).sum()
    return ((map_seq - map_seq.mean(axis=1, keepdims=True)) * t).sum(axis=1) / denom


def _trend_label(slopes, threshold=2.0):
    labels = np.full(len(slopes), "flat", dtype=object)
    labels[slopes >=  threshold] = "up"
    labels[slopes <= -threshold] = "down"
    return labels


def get_transformer_extended_metrics(mean_pred, all_ys, all_pls, map_std=1.0, trend_threshold=2.0):
    """
    mean_pred  : (N, horizon, 11) — mean prediction in normalized space
    all_ys     : (N, horizon, 11) — ground truth in normalized space
    all_pls    : (N, horizon)     — normalized P-level in the output window
    map_std    : MAP standard deviation in mmHg — used to unnormalize before slope computation
    trend_threshold : slope threshold in mmHg/step (default 2.0)
    """
    is_static = _classify_pl(all_pls)
    n_static  = is_static.sum()
    n_dynamic = (~is_static).sum()

    # MAE over all 11 columns
    mae_per = np.abs(mean_pred - all_ys).mean(axis=(1, 2))   # (N,)
    mae_static_per  = mae_per[is_static]
    mae_dynamic_per = mae_per[~is_static]
    mae_static       = mae_static_per.mean()  if n_static  > 0 else float("nan")
    mae_static_std   = mae_static_per.std()   if n_static  > 0 else float("nan")
    mae_dynamic      = mae_dynamic_per.mean() if n_dynamic > 0 else float("nan")
    mae_dynamic_std  = mae_dynamic_per.std()  if n_dynamic > 0 else float("nan")

    # Trend accuracy on MAP (col 0), unnormalized to mmHg/step
    pred_slope = _map_slope(mean_pred[:, :, 0] * map_std)
    true_slope = _map_slope(all_ys[:, :, 0]    * map_std)
    pred_trend = _trend_label(pred_slope, trend_threshold)
    true_trend = _trend_label(true_slope, trend_threshold)
    correct     = (pred_trend == true_trend).astype(float)
    trend_acc   = correct.mean()
    trend_acc_std = correct.std()

    print(f"\n=== Transformer extended metrics (MAE normalized, MAP slope in mmHg/step, threshold=±{trend_threshold}) ===")
    print(f"MAE Static  (n={n_static:4d}): {mae_static:.6f} ± {mae_static_std:.6f}")
    print(f"MAE Dynamic (n={n_dynamic:4d}): {mae_dynamic:.6f} ± {mae_dynamic_std:.6f}")
    print(f"Trend Acc   (MAP)          : {trend_acc:.4f} ± {trend_acc_std:.4f}")

    # per-class recall breakdown
    for cls in ("up", "flat", "down"):
        mask = true_trend == cls
        acc_cls = (pred_trend[mask] == cls).mean() if mask.any() else float("nan")
        print(f"  {cls:5s} (n={mask.sum():4d}) recall={acc_cls:.3f}")

    return {
        "mae_static":      mae_static,
        "mae_static_std":  mae_static_std,
        "mae_dynamic":     mae_dynamic,
        "mae_dynamic_std": mae_dynamic_std,
        "trend_acc":       trend_acc,
        "trend_acc_std":   trend_acc_std,
        "n_static":        int(n_static),
        "n_dynamic":       int(n_dynamic),
    }
