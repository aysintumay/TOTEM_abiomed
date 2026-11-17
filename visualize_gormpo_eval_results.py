"""
python visualize_gormpo_eval_results.py

Visualizes and reports metrics for the saved GORMPO ICL eval results (Med42-8B /
Qwen2.5-7B-Instruct, 0-shot / 3-shot -- forecasting/results_llm/gormpo_mcs_scratch_long2000/
{med42,qwen}_{0,3}shot_full_test.npz), without reloading any LLM -- only needs the raw
data pickle (for mean/std and sample contexts) and each run's saved (pred, true) arrays.
Runs whose .npz doesn't exist yet are skipped, so this can be re-run as each finishes.
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from gormpo_world_model import FEATURE_NAMES, compute_variable_ylims  # noqa: E402 (sets up sys.path)
from model import TimeSeriesDataset  # noqa: E402
from lib.models.metrics import dtw, pearsoncor  # noqa: E402

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
RESULTS_DIR = "forecasting/results_llm/gormpo_mcs_scratch_long2000"
OUT_DIR = "figures"
SAMPLE_IDXS = [0, 1, 2, 3]
COLUMNS = [i for i in range(0, 13) if i != 11]  # matches GormpoFewShotWorldModel.columns

# color = model identity, linestyle = k-shot count -- two independent dimensions encoded
# separately instead of 4 arbitrary hues, so "which model" and "how many shots" are each
# visually consistent across every plot.
MED42_COLOR = "#4C72B0"
QWEN_COLOR = "#DD8452"
ALL_RUNS = {
    "Med42 0-shot": {"npz": f"{RESULTS_DIR}/med42_0shot_full_test.npz", "color": MED42_COLOR, "style": "-"},
    "Qwen 0-shot": {"npz": f"{RESULTS_DIR}/qwen_0shot_full_test.npz", "color": QWEN_COLOR, "style": "-"},
    "Med42 3-shot": {"npz": f"{RESULTS_DIR}/med42_3shot_full_test.npz", "color": MED42_COLOR, "style": "--"},
    "Qwen 3-shot": {"npz": f"{RESULTS_DIR}/qwen_3shot_full_test.npz", "color": QWEN_COLOR, "style": "--"},
}
RUNS = {name: cfg for name, cfg in ALL_RUNS.items() if os.path.exists(cfg["npz"])}
missing = [name for name in ALL_RUNS if name not in RUNS]
if missing:
    print(f"Skipping (not finished yet): {missing}")

os.makedirs(OUT_DIR, exist_ok=True)

# ── load data (no LLM) ───────────────────────────────────────────────────────
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)
mean = torch.tensor(data["mean"], dtype=torch.float32)
std = torch.tensor(data["std"], dtype=torch.float32)
norm_test = (data["test"] - data["mean"]) / data["std"]
data_test = TimeSeriesDataset(norm_test[:, :, COLUMNS], 6, 6)

ylims = compute_variable_ylims(DATA_PATH)

for run in RUNS.values():
    d = np.load(run["npz"])
    run["pred"] = d["pred"]  # (N, 6, 11) normalized
    run["true"] = d["true"]
    run["n_fallback"] = int(d["n_fallback"])
    run["n"] = int(d["n"])

# ── average metrics table (recomputed directly from the saved arrays) ───────
print("=" * 88)
for name, run in RUNS.items():
    pred = torch.tensor(run["pred"], dtype=torch.float32)
    true = torch.tensor(run["true"], dtype=torch.float32)
    mae = (pred - true).abs().mean(dim=(0, 1))
    mse = (pred - true).pow(2).mean(dim=(0, 1))
    dtw_per = torch.stack([
        dtw(pred[:, :, j : j + 1], true[:, :, j : j + 1], reduction="mean") for j in range(11)
    ]).squeeze()
    corr_per = torch.stack([
        pearsoncor(pred[:, :, j : j + 1], true[:, :, j : j + 1], reduction="mean") for j in range(11)
    ]).squeeze()

    print(f"\n{name}  (n={run['n']}, fallback={run['n_fallback']}/{run['n']} "
          f"= {100 * run['n_fallback'] / run['n']:.1f}%)")
    print(f"  MAE={mae.mean():.4f}  MSE={mse.mean():.4f}  DTW={dtw_per.mean():.4f}  corr={corr_per.mean():.4f}"
          "  (each averaged over all 11 variables)")
    run["mae_per_var"] = mae.numpy()
print("=" * 88)

# ── sample-episode trajectory plots: context + ground truth + every available run ──
first_run = next(iter(RUNS.values()))
for idx in SAMPLE_IDXS:
    x, pl, y = data_test[idx]
    ctx_norm = torch.as_tensor(np.asarray(x)).float()          # (6, 12)
    ctx_phys = ctx_norm * std[COLUMNS] + mean[COLUMNS]

    t_ctx = np.arange(-5, 1) * 10
    t_pred = np.arange(1, 7) * 10

    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    axes = axes.flatten()
    fig.suptitle(f"Test sample {idx}: ground truth vs. {', '.join(RUNS.keys())} (GORMPO ICL)", fontsize=12)

    for j in range(11):
        ax = axes[j]
        ax.plot(t_ctx, ctx_phys[:, j].numpy(), color="#555555", marker="o", markersize=4, label="context")

        true_norm = torch.as_tensor(first_run["true"][idx, :, j])
        true_phys = true_norm * std[COLUMNS[j]] + mean[COLUMNS[j]]
        ax.plot(t_pred, true_phys.numpy(), color="black", marker="o", markersize=4,
                linestyle=":", label="ground truth")

        for name, run in RUNS.items():
            pred_norm = torch.as_tensor(run["pred"][idx, :, j])
            pred_phys = pred_norm * std[COLUMNS[j]] + mean[COLUMNS[j]]
            ax.plot(t_pred, pred_phys.numpy(), color=run["color"], linestyle=run["style"],
                     marker="o", markersize=4, label=name)

        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylim(*ylims[FEATURE_NAMES[j]])
        ax.set_title(FEATURE_NAMES[j], fontsize=9)
        ax.set_xlabel("time (min)", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

    for j in range(11, 12):
        axes[j].axis("off")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"gormpo_eval_sample_{idx:03d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")

# ── summary bar chart: per-variable MAE across every available run ──────────
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(11)
n_runs = len(RUNS)
width = 0.8 / n_runs
for i, (name, run) in enumerate(RUNS.items()):
    offset = (i - (n_runs - 1) / 2) * width
    ax.bar(x + offset, run["mae_per_var"], width, label=name, color=run["color"],
           hatch="//" if run["style"] == "--" else None, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(FEATURE_NAMES[:11], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("MAE (normalized z-score units)")
ax.set_title(f"Per-variable MAE: {', '.join(RUNS.keys())} (GORMPO ICL, full test set n=3876)")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
path = os.path.join(OUT_DIR, "gormpo_eval_mae_summary.png")
plt.savefig(path, dpi=120)
plt.close()
print(f"Saved {path}")

print("Done.")
