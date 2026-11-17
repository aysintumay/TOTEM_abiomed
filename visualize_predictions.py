"""
Plot ground truth vs. any set of models' forecasts for selected MCS test episodes, in raw
physical units (one 3x4 figure per episode: 11 forecast variables + the Pump Level plan).

Every model is read from a saved prediction file -- an .npz with `pred` and `true` arrays
(N_test, 6, 11) in z-score units (train mean/std of the pkl), test-set order -- as written by
the eval/dump scripts (eval_*.py --output_npz, dump_*.py, ensemble_chronos2_llm.py). To plot
a new model, dump its test predictions in that format and add it to MODELS or pass
--npz key=path.

Each model's ground truth is checked against the pkl's test episode, so a misaligned file
fails loudly instead of plotting the wrong episode.

y-limits: one fixed set for every figure, read from --ylim_json (mcs_fixed_ylims.json, the
limits all figures since Nov 2026 use). --recompute_ylims rebuilds them from the train
1st/99th percentile widened to cover every plotted episode and forecast (+5% pad), with
Pump Level fixed at (1, 11) and USER_YLIMS winning, and writes them to --ylim_json.

Run:
  python visualize_predictions.py --models c2_gn transformer timesfm --out_dir figures/timesfm3_vs_best
  python visualize_predictions.py --models c2_gn --npz c2_gn=logs/other_run.npz --out_dir figures/x
"""
import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FEATURES = [
    ("PumpPressure", "mmHg"), ("PumpSpeed", "RPM"), ("PumpFlow", "device units"),
    ("LVP", "mmHg"), ("LVEDP", "mmHg"), ("SYSTOLIC", "mmHg"), ("DIASTOLIC", "mmHg"),
    ("PULSAT", "mmHg"), ("PumpCurrent", "device units"), ("Heart Rate", "bpm"),
    ("ESE_lv", "mmHg/mL"), ("Pump Level", "P-level"),
]
COLS = [i for i in range(13) if i != 11]
# dataviz reference palette; truth in neutral ink
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
# key: (legend label, prediction npz -- or a list of striped shards -- , color).
# The 8 categorical slots are exhausted, so some colors repeat; don't draw those pairs together:
# llm_mcs/c2_gn (aqua), llm_nomcs/timesfm (orange).
MODELS = {
    "chronos": ("Chronos-2 (zero-shot)", "logs/chronos2_zeroshot_test_preds.npz", "#2a78d6"),
    "llm_nomcs": ("MedLlama 8B, no mcs.md", "logs/med42_history_prompt_nomcs.npz", "#eb6834"),
    "llm_mcs": ("MedLlama 8B, with mcs.md",
                ["logs/med42_history_prompt_shard0.npz", "logs/med42_history_prompt_shard1.npz"], "#1baf7a"),
    "ens": ("Ensemble 0.5 MedLlama (no mcs.md) + 0.5 Chronos-2", "logs/ensemble_med42nomcs_chronos2zs.npz",
            "#eda100"),
    "llm_rev": ("MedLlama 8B, revised context", "logs/med42_revctx_full.npz", "#e87ba4"),
    "ens_rev": ("Ensemble 0.5 MedLlama (revised) + 0.5 Chronos-2", "logs/ensemble_med42revctx_chronos2zs.npz",
                "#008300"),
    "transformer": ("Transformer WorldModel", "logs/transformer_test_preds.npz", "#6250d6"),
    "c2_ft": ("Chronos-2 fine-tuned", "logs/chronos2_finetuned_test_preds.npz", "#e34948"),
    "c2_pump": ("Chronos-2, pump channels only", "logs/chronos2_pumponly_test_preds.npz", "#6b6a64"),
    "c2_gn": ("Chronos-2 fine-tuned, global norm", "logs/chronos2_globalnorm_finetuned_test_preds.npz",
              "#1baf7a"),
    "timesfm": ("TimesFM 3.0 (zero-shot)", "logs/timesfm3_native_padedge_test_preds.npz", "#eb6834"),
}
# User-specified y-limits (clinical display ranges); override the computed ones.
# MAP -> PumpPressure (the MAP reward is computed from PumpPressure), HR -> Heart Rate.
USER_YLIMS = {"PumpPressure": (10, 150), "Heart Rate": (40, 160), "PULSAT": (10, 80)}
T = 6
T_CTX = np.arange(-5, 1) * 10   # -50 .. 0 min
T_PRED = np.arange(1, 7) * 10   # +10 .. +60 min
XLIM = (-55.5, 65.5)            # matplotlib's default 5% margin on [-50, 60]


def load_npz(src, n_total):
    """-> pred, true (N, 6, 11) z-score. A list = striped shards (shard k holds episodes k, k+S, ...)."""
    if isinstance(src, str):
        z = np.load(src)
        return z["pred"], z["true"]
    pred = np.zeros((n_total, T, 11), np.float32)
    true = np.zeros_like(pred)
    for k, p in enumerate(src):
        z = np.load(p)
        pred[k::len(src)], true[k::len(src)] = z["pred"], z["true"]
    return pred, true


def compute_ylims(d, test, preds, samples):
    train = np.asarray(d["train"], dtype=np.float32)[:, :, COLS].reshape(-1, 12)
    lo_all, hi_all = np.percentile(train, 1, axis=0), np.percentile(train, 99, axis=0)
    ylims = {}
    for j, (name, _) in enumerate(FEATURES[:11]):
        lo, hi = lo_all[j], hi_all[j]
        for idx in samples:
            vals = [test[idx][:, j]] + [p[idx, :, j] for p in preds.values() if not np.isnan(p[idx, :, j]).any()]
            lo, hi = min(lo, *(v.min() for v in vals)), max(hi, *(v.max() for v in vals))
        pad = 0.05 * (hi - lo)
        ylims[name] = (float(lo - pad), float(hi + pad))
    ylims["Pump Level"] = (1.0, 11.0)
    ylims.update({k: tuple(v) for k, v in USER_YLIMS.items()})
    return ylims


def main(args):
    for kv in args.npz:
        k, path = kv.split("=", 1)
        assert k in MODELS, f"unknown model key {k}"
        MODELS[k] = (MODELS[k][0], path, MODELS[k][2])
    unknown = set(args.models) - set(MODELS)
    assert not unknown, f"unknown model keys {unknown}; choose from {list(MODELS)}"

    with open(args.mcs_pickle_path, "rb") as f:
        d = pickle.load(f)
    test = np.asarray(d["test"], dtype=np.float32)[:, :, COLS]  # (E, 12 steps, 12 features) physical
    m = np.asarray(d["mean"], dtype=np.float32)[COLS][:11]
    s = np.asarray(d["std"], dtype=np.float32)[COLS][:11]

    preds = {}
    for k in args.models:
        pred, true = load_npz(MODELS[k][1], len(test))
        for idx in args.samples:  # every model must be scored on this exact episode
            assert np.allclose(test[idx, T:, :11], true[idx] * s + m, atol=1e-2), f"episode {idx} misaligned in {k}"
        preds[k] = pred * s + m

    if os.path.exists(args.ylim_json) and not args.recompute_ylims:
        with open(args.ylim_json) as f:
            ylims = {k: tuple(v) for k, v in json.load(f).items()}
        print(f"using fixed y-limits from {args.ylim_json}")
    else:
        ylims = compute_ylims(d, test, preds, args.samples)
        with open(args.ylim_json, "w") as f:
            json.dump(ylims, f, indent=1)
        print(f"saved fixed y-limits to {args.ylim_json}")

    os.makedirs(args.out_dir, exist_ok=True)
    for idx in args.samples:
        ep = test[idx]
        fig, axes = plt.subplots(3, 4, figsize=(19, 12), facecolor=SURFACE)
        for j, ax in enumerate(axes.flat):
            name, unit = FEATURES[j]
            ax.set_facecolor(SURFACE)
            ax.axvspan(5, XLIM[1], color=GRID, alpha=0.35, lw=0)
            ax.plot(np.r_[T_CTX, T_PRED], ep[:, j], color=INK, lw=2, marker="o", ms=4,
                    label="Ground truth", zorder=3)
            if j < 11:
                for k in args.models:
                    label, _, color = MODELS[k]
                    ax.plot(T_PRED, preds[k][idx, :, j], color=color, ls="-", lw=1.5,
                            marker="o", ms=3, label=label, zorder=4)
            ax.set_ylim(*ylims[name])
            ax.set_xlim(*XLIM)
            ax.set_title(f"{name} ({unit})", fontsize=15, color=INK, loc="left")
            ax.set_xticks(range(-40, 61, 20))
            ax.set_xlabel("time (min)", fontsize=12, color=MUTED)
            ax.tick_params(colors=MUTED, labelsize=12, length=0)
            ax.grid(axis="y", color=GRID, lw=0.8)
            for sp in ax.spines.values():
                sp.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=14,
                   bbox_to_anchor=(0.5, 0.97), labelcolor=INK)
        pl_fut = ", ".join(f"P{p:g}" for p in np.round(ep[T:, 11], 1))
        fig.suptitle(f"MCS test episode {idx}: 1 h history (−50 to 0 min) → 1 h forecast (+10 to +60 min, "
                     f"shaded); planned Pump Level {pl_fut}", fontsize=17, color=INK, x=0.5, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.945 if len(handles) <= 4 else 0.915))  # 1 vs 2 legend rows
        out = os.path.join(args.out_dir, f"episode_{idx}.png")
        fig.savefig(out, dpi=130, facecolor=SURFACE)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help=f"keys from: {' '.join(MODELS)}")
    p.add_argument("--npz", nargs="*", default=[], metavar="KEY=PATH", help="override a model's prediction file")
    p.add_argument("--samples", type=int, nargs="+", default=[321, 1240, 3740])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mcs_pickle_path", default="/public/gormpo/10min_1hr_all_data.pkl")
    p.add_argument("--ylim_json", default="mcs_fixed_ylims.json")
    p.add_argument("--recompute_ylims", action="store_true")
    main(p.parse_args())
