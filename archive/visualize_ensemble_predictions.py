"""
Plot ground truth vs. Chronos-2 (zero-shot), MedLlama 8B (no mcs.md context), MedLlama 8B
(with mcs.md context) and the 0.5/0.5 MedLlama(no mcs.md) + Chronos-2 ensemble for selected
MCS test episodes, in raw physical units.

Axis limits follow visualize_predictions_chronos2_xtilde_notimegroup.py's figures
(e.g. figures/chronos2_xtilde_notimegroup_globalpl_sample_1240.png), so panels can be
compared side by side:
  - x: time in minutes, context -50..0, forecast +10..+60, matplotlib's default 5% margin
  - y: train 1st/99th percentile (compute_variable_ylims), widened to cover context, ground
    truth and THAT model's q05/q95 band, + 5% pad; Pump Level fixed at (1, 11).
One FIXED set of y-limits is used for every figure: per variable, the widest such range
across all --samples (also covering every forecast drawn), saved to --ylim_json and reused.
The reference model's q05/q95 band is needed only to reproduce those limits, so it is
computed once for the requested samples and cached (--ref_cache); it is not drawn.

Run (Chronos-2 needs the local chronos-forecasting/src on PYTHONPATH):
  PYTHONPATH=chronos-forecasting/src python visualize_ensemble_predictions.py --samples 321 1240 3740
"""
import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gormpo_world_model import compute_variable_ylims  # noqa: E402

FEATURES = [
    ("PumpPressure", "mmHg"), ("PumpSpeed", "RPM"), ("PumpFlow", "device units"),
    ("LVP", "mmHg"), ("LVEDP", "mmHg"), ("SYSTOLIC", "mmHg"), ("DIASTOLIC", "mmHg"),
    ("PULSAT", "mmHg"), ("PumpCurrent", "device units"), ("Heart Rate", "bpm"),
    ("ESE_lv", "mmHg/mL"), ("Pump Level", "P-level"),
]
# dataviz reference palette, categorical slots 1-4 in fixed order; truth in neutral ink
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
MODELS = [  # (label, key, color)
    ("Chronos-2 (zero-shot)", "chronos", "#2a78d6"),
    ("MedLlama 8B, no mcs.md", "llm_nomcs", "#eb6834"),
    ("MedLlama 8B, with mcs.md", "llm_mcs", "#1baf7a"),
    ("Ensemble 0.5 MedLlama (no mcs.md) + 0.5 Chronos-2", "ens", "#eda100"),
    ("MedLlama 8B, revised context", "llm_rev", "#e87ba4"),
    ("Ensemble 0.5 MedLlama (revised) + 0.5 Chronos-2", "ens_rev", "#008300"),
    ("Transformer WorldModel", "transformer", "#6250d6"),
    ("Chronos-2 fine-tuned", "c2_ft", "#e34948"),
    ("Chronos-2, pump channels only", "c2_pump", "#6b6a64"),  # palette exhausted: neutral gray
    # palette exhausted: reuses slot 3 (aqua); never drawn in the same figure as llm_mcs
    ("Chronos-2 fine-tuned, global norm", "c2_gn", "#1baf7a"),
    # palette exhausted: reuses slot 2 (orange); never drawn in the same figure as llm_nomcs
    ("TimesFM 3.0 (zero-shot)", "timesfm", "#eb6834"),
]
# User-specified y-limits (clinical display ranges); override the computed ones.
# MAP -> PumpPressure (mcs.md: the MAP reward is computed from PumpPressure), HR -> Heart Rate.
USER_YLIMS = {"PumpPressure": (10, 150), "Heart Rate": (40, 160), "PULSAT": (10, 80)}
T = 6
T_CTX = np.arange(-5, 1) * 10   # -50 .. 0 min
T_PRED = np.arange(1, 7) * 10   # +10 .. +60 min
XLIM = (-55.5, 65.5)            # matplotlib's default 5% margin on [-50, 60], as in the reference


def reference_bands(samples, args):
    """q05/q95 (physical, (11, 6)) of the reference x_tilde/no-group Chronos-2 model, per sample."""
    cache = dict(np.load(args.ref_cache)) if os.path.exists(args.ref_cache) else {}
    missing = [i for i in samples if f"q05_{i}" not in cache]
    if missing:
        import torch
        from chronos import Chronos2Pipeline
        from chronos2_time_only_encoder import strip_group_self_attention
        from gormpo_xtilde import affine_decode, compute_x_tilde, load_pump_level_stats
        sys.path.insert(0, "forecasting")
        from train_tokenizer import load_mcs_episode_halves

        dev = args.device
        pl_mean, pl_std = load_pump_level_stats(args.mcs_pickle_path)
        tok = torch.load(args.ref_tokenizer, map_location=dev, weights_only=False).to(dev).eval()
        pipe = Chronos2Pipeline.from_pretrained(args.ref_model, device_map=dev)
        strip_group_self_attention(pipe.model)
        qi = [pipe.quantiles.index(q) for q in (0.05, 0.95)]
        x_half, y_half = load_mcs_episode_halves("forecasting/data_raw/abiomed", "test", 6)
        for i in missing:
            x_t, y_t = x_half[i:i + 1].to(dev), y_half[i:i + 1].to(dev)
            with torch.no_grad():
                xt, mu, sig = compute_x_tilde(tok, x_t)
            out = pipe.predict([{
                "target": xt[0, 0:11, :].cpu().numpy(),
                "past_covariates": {"pump_level": ((x_t[0, 11, :] - pl_mean) / pl_std).cpu().numpy()},
                "future_covariates": {"pump_level": ((y_t[0, 11, :] - pl_mean) / pl_std).cpu().numpy()},
            }], prediction_length=6, batch_size=1)[0].to(dev)
            for name, q in zip(("q05", "q95"), qi):
                cache[f"{name}_{i}"] = affine_decode(out[:, q, :], mu[0, 0:11], sig[0, 0:11]).cpu().numpy()
        np.savez(args.ref_cache, **cache)
        print(f"cached reference q05/q95 for samples {missing} -> {args.ref_cache}")
    return {i: (cache[f"q05_{i}"], cache[f"q95_{i}"]) for i in samples}


def load_sharded(paths, n_total):
    """Reassemble striped shards (shard k holds episodes k, k+S, k+2S, ...) into test order."""
    S = len(paths)
    pred = np.zeros((n_total, T, 11), np.float32)
    true = np.zeros_like(pred)
    for k, p in enumerate(paths):
        z = np.load(p)
        pred[k::S], true[k::S] = z["pred"], z["true"]
    return pred, true


def main(args):
    global MODELS
    if args.models:  # plot only a subset of the forecasts (keys from MODELS), keeping their colors
        keep = set(args.models)
        assert keep <= {k for _, k, _ in MODELS}, f"unknown model keys {keep - {k for _, k, _ in MODELS}}"
        MODELS = [mdl for mdl in MODELS if mdl[1] in keep]
    with open(args.mcs_pickle_path, "rb") as f:
        d = pickle.load(f)
    cols = [i for i in range(13) if i != 11]
    test = np.asarray(d["test"], dtype=np.float32)[:, :, cols]  # (E, 12 steps, 12 features) physical
    m = np.asarray(d["mean"], dtype=np.float32)[cols][:11]
    s = np.asarray(d["std"], dtype=np.float32)[cols][:11]

    preds = {k: np.load(p)["pred"] for k, p in
             [("chronos", args.chronos_npz), ("llm_nomcs", args.llm_nomcs_npz), ("ens", args.ens_npz),
              ("llm_rev", args.llm_rev_npz), ("ens_rev", args.ens_rev_npz),
              ("transformer", args.transformer_npz), ("c2_pump", args.c2_pump_npz),
              ("c2_ft", args.c2_ft_npz), ("c2_gn", args.c2_gn_npz), ("timesfm", args.timesfm_npz)]}
    preds["llm_mcs"], true_mcs = load_sharded(args.llm_mcs_shards, len(test))
    trues = {"chronos": np.load(args.chronos_npz)["true"], "llm_nomcs": np.load(args.llm_nomcs_npz)["true"],
             "llm_mcs": true_mcs, "ens": np.load(args.ens_npz)["true"],
             "llm_rev": np.load(args.llm_rev_npz)["true"], "ens_rev": np.load(args.ens_rev_npz)["true"],
             "transformer": np.load(args.transformer_npz)["true"],
             "c2_pump": np.load(args.c2_pump_npz)["true"],
             "c2_ft": np.load(args.c2_ft_npz)["true"],
             "c2_gn": np.load(args.c2_gn_npz)["true"],
             "timesfm": np.load(args.timesfm_npz)["true"]}
    preds = {k: v * s + m for k, v in preds.items()}

    ylims = compute_variable_ylims(args.mcs_pickle_path)
    bands = reference_bands(args.samples, args)

    # ── one fixed set of y-limits for every figure: per variable, the WIDEST range over all
    # samples of the reference formula (train 1/99 pct widened by context, truth and the
    # reference model's q05/q95), also widened to cover every forecast drawn here, + 5% pad.
    ylim_fixed = {}
    for j, (name, _) in enumerate(FEATURES[:11]):
        lo, hi = ylims[name]
        for idx in args.samples:
            q05, q95 = bands[idx]
            vals = [test[idx][:, j]] + [preds[k][idx, :, j] for _, k, _ in MODELS
                                         if not np.isnan(preds[k][idx, :, j]).any()]
            if args.ylim_mode == "reference":  # also cover the reference model's q05/q95 band
                vals += [q05[j], q95[j]]
            lo = min(lo, *(v.min() for v in vals))
            hi = max(hi, *(v.max() for v in vals))
        pad = 0.05 * (hi - lo)
        ylim_fixed[name] = (float(lo - pad), float(hi + pad))
    ylim_fixed["Pump Level"] = tuple(ylims["Pump Level"])
    ylim_fixed.update({k: tuple(v) for k, v in USER_YLIMS.items()})  # user-specified limits win
    if os.path.exists(args.ylim_json) and not args.recompute_ylims:
        with open(args.ylim_json) as f:  # reuse the saved limits so every figure here matches
            ylim_fixed = {k: tuple(v) for k, v in json.load(f).items()}
        print(f"using fixed y-limits from {args.ylim_json}")
    else:
        with open(args.ylim_json, "w") as f:
            json.dump(ylim_fixed, f, indent=1)
        print(f"saved fixed y-limits to {args.ylim_json}")
    for name, (lo, hi) in ylim_fixed.items():
        print(f"  {name:<14} ({lo:9.3f}, {hi:9.3f})")

    os.makedirs(args.out_dir, exist_ok=True)
    for idx in args.samples:
        ep = test[idx]
        for k, t in trues.items():  # every model must be scored on this exact episode
            assert np.allclose(ep[T:, :11], t[idx] * s + m, atol=1e-2), f"episode {idx} misaligned in {k}"
        fig, axes = plt.subplots(3, 4, figsize=(19, 12), facecolor=SURFACE)
        for j, ax in enumerate(axes.flat):
            name, unit = FEATURES[j]
            ax.set_facecolor(SURFACE)
            ax.axvspan(5, XLIM[1], color=GRID, alpha=0.35, lw=0)
            ax.plot(np.r_[T_CTX, T_PRED], ep[:, j], color=INK, lw=2, marker="o", ms=4,
                    label="Ground truth", zorder=3)
            if j < 11:
                for label, key, color in MODELS:
                    # same line style as the reference figures' prediction: solid, small markers,
                    # forecast steps only (+10..+60 min)
                    ax.plot(T_PRED, preds[key][idx, :, j], color=color, ls="-", lw=1.5,
                            marker="o", ms=3, label=label, zorder=4)
            ax.set_ylim(*ylim_fixed[name])
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
    p.add_argument("--samples", type=int, nargs="+", default=[321, 1240, 3740])
    p.add_argument("--chronos_npz", default="logs/chronos2_zeroshot_test_preds.npz")
    p.add_argument("--llm_nomcs_npz", default="logs/med42_history_prompt_nomcs.npz")
    p.add_argument("--llm_mcs_shards", nargs="+",
                   default=["logs/med42_history_prompt_shard0.npz", "logs/med42_history_prompt_shard1.npz"])
    p.add_argument("--ens_npz", default="logs/ensemble_med42nomcs_chronos2zs.npz")
    p.add_argument("--llm_rev_npz", default="logs/med42_revctx_full.npz")
    p.add_argument("--ens_rev_npz", default="logs/ensemble_med42revctx_chronos2zs.npz")
    p.add_argument("--c2_gn_npz", default="logs/chronos2_globalnorm_finetuned_test_preds.npz")
    p.add_argument("--c2_ft_npz", default="logs/chronos2_finetuned_test_preds.npz")
    p.add_argument("--c2_pump_npz", default="logs/chronos2_pumponly_test_preds.npz")
    p.add_argument("--transformer_npz", default="logs/transformer_test_preds.npz")
    p.add_argument("--timesfm_npz", default="logs/timesfm3_native_padedge_test_preds.npz")
    p.add_argument("--mcs_pickle_path", default="/public/gormpo/10min_1hr_all_data.pkl")
    p.add_argument("--out_dir", default="figures/ensemble_med42nomcs_chronos2")
    # reference model used only to reproduce the reference figures' y-limits
    p.add_argument("--ref_model", default="saved_models/chronos2_xtilde_notimegroup_globalpl_mcs/finetuned-ckpt")
    p.add_argument("--ref_tokenizer",
                   default="forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth")
    p.add_argument("--ref_cache", default="logs/xtilde_notimegroup_globalpl_q05_q95_cache.npz")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ylim_json", default="figures/ensemble_med42nomcs_chronos2/fixed_ylims.json",
                   help="fixed per-variable y-limits shared by every figure; created on first run")
    p.add_argument("--recompute_ylims", action="store_true")
    p.add_argument("--models", nargs="+", default=None,
                   help="subset of: chronos llm_nomcs llm_mcs ens llm_rev ens_rev (default: all)")
    p.add_argument("--ylim_mode", choices=["data", "reference"], default="data",
                   help="data: train 1/99 pct + plotted values; reference: also the reference model's q05/q95")
    main(p.parse_args())
