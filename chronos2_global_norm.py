"""
Global normalization for Chronos-2 on MCS.

Chronos-2's InstanceNorm standardizes every series by the mean/std of its OWN context window
and reuses that loc/scale for the future covariates and to un-scale predictions. With a
6-step context, flat series (Pump Level constant, so PumpSpeed constant) get std = 0 ->
eps = 1e-5: a planned P7 -> P2 change becomes -5e5 and target outputs are scaled by 1e-5,
so the model cannot respond to Pump Level changes (verified on test sample 1240).

Fix: feed data already z-scored with the TRAIN mean/std of the pkl (all 11 channels + Pump
Level) and make InstanceNorm the identity (loc 0, scale 1) when it computes its own stats.
Import this module (it patches on import) in BOTH training and inference.
"""
import pickle

import numpy as np
import torch

from chronos import chronos_bolt

COLS = [i for i in range(13) if i != 11]  # pkl column 11 is all-NaN; Pump Level is last


def _identity_forward(self, x, loc_scale=None):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if loc_scale is None:
        shape = (*x.shape[:-1], 1)
        loc_scale = (torch.zeros(shape, device=x.device), torch.ones(shape, device=x.device))
    loc, scale = loc_scale
    y = (x - loc) / scale
    if self.use_arcsinh:
        y = torch.arcsinh(y)
    return y.to(orig_dtype), (loc, scale)


chronos_bolt.InstanceNorm.forward = _identity_forward


def load_split(pkl_path, split):
    """-> z-scored (E, 12 steps, 12 features) float32 (11 channels + Pump Level), mean, std (12,)."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    mean = np.asarray(d["mean"], dtype=np.float32)[COLS]
    std = np.asarray(d["std"], dtype=np.float32)[COLS]
    x = (np.asarray(d[split], dtype=np.float32)[:, :, COLS] - mean) / std
    return x.astype(np.float32), mean, std


def predict_inputs(x):
    """z-scored episodes (E, 12, 12) -> Chronos-2 predict() inputs (history + planned Pump Level)."""
    return [{"target": ep[:6, :11].T, "past_covariates": {"pump_level": ep[:6, 11]},
             "future_covariates": {"pump_level": ep[6:, 11]}} for ep in x]
