import pickle
import numpy as np

with open("/public/gormpo/10min_1hr_all_data.pkl", "rb") as f:
    data = pickle.load(f)
train = np.asarray(data["train"])  # (N, 12, 13)
N, T, C = train.shape
FEATURE_NAMES = ["PumpPressure", "PumpSpeed", "PumpFlow", "LVP", "LVEDP", "SYSTOLIC",
                  "DIASTOLIC", "PULSAT", "PumpCurrent", "Heart Rate", "ESE_lv"]
PHYS_CHANNELS = list(range(11))  # raw indices 0-10 (11=dropped col, 12=Pump Level)
K = 15  # generous -- only catches truly extreme/impossible spikes, not rare-but-real physiology

bounds = {}
for ch in PHYS_CHANNELS:
    vals = train[:, :, ch]
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    lo, hi = med - K * mad * 1.4826, med + K * mad * 1.4826
    bounds[ch] = (lo, hi)

flagged_any = np.zeros((N, T), dtype=bool)
header = "channel".ljust(14) + "median".rjust(10) + "MAD".rjust(10) + "lo_bound".rjust(12) + "hi_bound".rjust(12) + "n_flagged".rjust(10)
print(header)
for ch in PHYS_CHANNELS:
    vals = train[:, :, ch]
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    lo, hi = bounds[ch]
    bad = (vals < lo) | (vals > hi)
    flagged_any |= bad
    name = FEATURE_NAMES[ch]
    print(name.ljust(14) + f"{med:>10.2f}{mad:>10.3f}{lo:>12.2f}{hi:>12.2f}{int(bad.sum()):>10d}")

print()
n_bad_timesteps = int(flagged_any.sum())
n_bad_episodes = int(flagged_any.any(axis=1).sum())
print(f"TOTAL bad timesteps (any channel flagged): {n_bad_timesteps} out of {N*T} ({100*n_bad_timesteps/(N*T):.4f}%)")
print(f"TOTAL episodes containing >=1 bad timestep: {n_bad_episodes} out of {N} ({100*n_bad_episodes/N:.4f}%)")

ep_idx, t_idx = np.where(flagged_any)
print()
print("all flagged (episode, timestep) pairs:")
for e, t in zip(ep_idx, t_idx):
    row = train[e, t]
    bad_channels = [FEATURE_NAMES[c] for c in PHYS_CHANNELS if (row[c] < bounds[c][0] or row[c] > bounds[c][1])]
    print(f"  episode={e} t={t}  flagged_channels={bad_channels}  values={[round(row[c],2) for c in PHYS_CHANNELS]}")
