"""
python impute_outliers_and_save.py

Data-cleaning step: imputes the 2 known sensor-glitch timesteps (episode 2608 t=11,
episode 10214 t=0 -- SYSTOLIC/PULSAT readings >400, physically impossible) with the
patch's own mean (mu, computed from the other 5 valid timesteps of the SAME 6-step
half the glitch sits in), directly in the raw data. Everything downstream (tokenizer
patch_and_scale, VQ-VAE, LoRA training, token extraction) then needs zero special-
casing anywhere -- it all just sees clean data.

Only the two specific (episode, timestep, channel) cells are touched. val/test are
untouched (already confirmed clean). mean/std are left as originally computed (2
episodes out of 12,051 has negligible effect on a global mean/std, unlike the
per-quantile-bin fit which is far more sensitive to single outliers).

Saves a new pickle (repo-local, not overwriting the external /public/gormpo source).
"""
import pickle

import numpy as np
import torch

SRC_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
OUT_PATH = "forecasting/data_raw/10min_1hr_all_data_imputed.pkl"
SYSTOLIC, DIASTOLIC, PULSAT = 5, 6, 7
BAD_TIMESTEPS = [(2608, 11), (10214, 0)]

with open(SRC_PATH, "rb") as f:
    data = pickle.load(f)

orig_dtype = data["train"].dtype  # torch.float64 -- preserve type/dtype, downstream code (e.g.
# abiomed_env/model.py's load_data) does tensor arithmetic against mean/std and breaks on a
# plain ndarray here even though numpy indexing/assignment below works fine on either.
train = np.asarray(data["train"]).copy()

for episode, t in BAD_TIMESTEPS:
    half_start = 0 if t < 6 else 6
    valid = np.ones(6, dtype=bool)
    valid[t - half_start] = False

    systolic_half = train[episode, half_start:half_start + 6, SYSTOLIC]
    systolic_mu = systolic_half[valid].mean()
    before_systolic = train[episode, t, SYSTOLIC]
    train[episode, t, SYSTOLIC] = systolic_mu
    print(f"episode={episode} t={t} SYSTOLIC: {before_systolic:.2f} -> {systolic_mu:.2f}"
          "  (mean of other 5 valid timesteps)")

    # PULSAT is mechanically SYSTOLIC - DIASTOLIC (corr=0.99999... across the whole
    # dataset) -- derive it from the now-corrected SYSTOLIC and the untouched (never
    # flagged) DIASTOLIC at this exact timestep, rather than imputing independently,
    # or the exact identity breaks for these 2 cells.
    before_pulsat = train[episode, t, PULSAT]
    diastolic = train[episode, t, DIASTOLIC]
    pulsat_derived = systolic_mu - diastolic
    train[episode, t, PULSAT] = pulsat_derived
    print(f"episode={episode} t={t} PULSAT: {before_pulsat:.2f} -> {pulsat_derived:.2f}"
          f"  (= corrected SYSTOLIC {systolic_mu:.2f} - DIASTOLIC {diastolic:.2f}, preserves identity)")

data["train"] = torch.from_numpy(train).to(orig_dtype)

with open(OUT_PATH, "wb") as f:
    pickle.dump(data, f)
print(f"\nSaved imputed dataset to {OUT_PATH}")

# ---- verify: re-run the same outlier scan, should now be zero ----
FEATURE_NAMES = ["PumpPressure", "PumpSpeed", "PumpFlow", "LVP", "LVEDP", "SYSTOLIC",
                  "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate", "ESE_lv"]
flagged = np.zeros(train.shape[:2], dtype=bool)
for ch in range(11):
    vals = train[:, :, ch]
    med, mad = np.median(vals), np.median(np.abs(vals - np.median(vals)))
    lo, hi = med - 15 * mad * 1.4826, med + 15 * mad * 1.4826
    flagged |= (vals < lo) | (vals > hi)
print(f"post-imputation outlier scan: {int(flagged.sum())} flagged timesteps (should be 0)")
