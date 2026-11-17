"""
python filtered_mu_sigma_check.py

Outlier check scoped to SYSTOLIC and PULSAT only (the two affected channels).
For each 6-timestep patch (context half 0:6, target half 6:12, per training episode),
flags any timestep where SYSTOLIC or PULSAT exceeds a physiologically-plausible bound,
and recomputes that patch's mu/sigma for BOTH channels with the flagged timestep(s)
excluded (discarded across the whole timestep, not just the offending channel) --
matching patch_and_scale()'s per-patch mean/var, just timestep-masked.
"""
import pickle

import numpy as np

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
SYSTOLIC, PULSAT = 5, 7
CHECK_CHANNELS = [SYSTOLIC, PULSAT]  # outlier DETECTION scoped to just these two
NAMES = {SYSTOLIC: "SYSTOLIC", PULSAT: "PULSAT"}

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)
train = np.asarray(data["train"])  # (N, 12, 13)
N, T, C = train.shape

# per-channel plausibility bounds, robust median/MAD (unaffected by the outliers themselves)
bounds = {}
for ch in CHECK_CHANNELS:
    vals = train[:, :, ch]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    K = 15
    bounds[ch] = (med - K * mad * 1.4826, med + K * mad * 1.4826)
    print(f"{NAMES[ch]} bounds: [{bounds[ch][0]:.2f}, {bounds[ch][1]:.2f}]  (median={med:.2f}, MAD={mad:.3f})")

# per-(episode, raw timestep) validity: True unless SYSTOLIC or PULSAT is out of bounds there
valid_timestep = np.ones((N, T), dtype=bool)
for ch in CHECK_CHANNELS:
    lo, hi = bounds[ch]
    valid_timestep &= (train[:, :, ch] >= lo) & (train[:, :, ch] <= hi)

n_bad = int((~valid_timestep).sum())
print(f"\nFlagged timesteps: {n_bad} / {N*T} ({100*n_bad/(N*T):.4f}%)")
print(f"Episodes containing >=1 flagged timestep: {int((~valid_timestep).any(axis=1).sum())} / {N}")

# ---- BEFORE vs AFTER per-patch mu/sigma, for the two affected channels ----
def patch_mu_sigma(patch_vals, patch_valid, ch_vals_all_patches=None):
    """patch_vals: (n_patches, 6) raw values for one channel. patch_valid: (n_patches, 6) bool mask.
    Returns masked mean/std per patch (falls back to unmasked if all 6 flagged, which never happens here)."""
    n_valid = patch_valid.sum(axis=1)
    safe_valid = np.where(n_valid[:, None] > 0, patch_valid, True)  # guard against all-False rows
    masked = np.where(safe_valid, patch_vals, np.nan)
    mu = np.nanmean(masked, axis=1)
    sigma = np.nanstd(masked, axis=1)
    return mu, sigma

for ch in CHECK_CHANNELS:
    ctx_vals, tgt_vals = train[:, 0:6, ch], train[:, 6:12, ch]
    ctx_valid, tgt_valid = valid_timestep[:, 0:6], valid_timestep[:, 6:12]

    mu_before = np.concatenate([ctx_vals.mean(axis=1), tgt_vals.mean(axis=1)])
    sigma_before = np.concatenate([ctx_vals.std(axis=1), tgt_vals.std(axis=1)])

    mu_ctx_after, sigma_ctx_after = patch_mu_sigma(ctx_vals, ctx_valid)
    mu_tgt_after, sigma_tgt_after = patch_mu_sigma(tgt_vals, tgt_valid)
    mu_after = np.concatenate([mu_ctx_after, mu_tgt_after])
    sigma_after = np.concatenate([sigma_ctx_after, sigma_tgt_after])

    print(f"\n=== {NAMES[ch]}: per-patch sigma, before vs after filtering ===")
    print(f"  BEFORE: max sigma = {sigma_before.max():.2f}   (top-3: {np.sort(sigma_before)[-3:][::-1].round(2).tolist()})")
    print(f"  AFTER:  max sigma = {sigma_after.max():.2f}   (top-3: {np.sort(sigma_after)[-3:][::-1].round(2).tolist()})")

    # refit a 32-bin quantile quantizer on the corrected sigma distribution, same scheme as
    # ScalarBinQuantizer.fit(), to see the new top-bin edge (what actually gets used to decode)
    qs = np.linspace(0, 1, 33)
    edges_before = np.quantile(sigma_before, qs)
    edges_after = np.quantile(sigma_after, qs)
    print(f"  top-bin edge (used for dequantize center) BEFORE: [{edges_before[-2]:.2f}, {edges_before[-1]:.2f}]"
          f" -> center={ (edges_before[-2]+edges_before[-1])/2 :.2f}")
    print(f"  top-bin edge (used for dequantize center) AFTER:  [{edges_after[-2]:.2f}, {edges_after[-1]:.2f}]"
          f" -> center={ (edges_after[-2]+edges_after[-1])/2 :.2f}")
