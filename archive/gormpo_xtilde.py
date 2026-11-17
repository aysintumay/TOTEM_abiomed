"""
Shared helper: compute GORMPO's cross-channel-attended, per-channel-normalized patch
representation (x_tilde, Eq 10-16 of forecasting/lib/models/tokenizer.py) WITHOUT
going through the frozen conv encoder / VQ codebook / decoder at all -- the
"skip discrete tokenization entirely, feed the continuous cross-attended features
straight into Chronos-2" design. See train_gormpo_chronos2.py's module docstring for
why the existing tokenizer.decode() path can't be reused here (it expects
post-VQ `quantized` tensors, a different shape/space than x_tilde): decoding a
predicted x_tilde back to physical units instead uses a direct affine restoration,
`physical ~= sigma * x_tilde + mu`, treating x_tilde as already being in roughly the
same normalized "shape" space as x_bar (true up to cross_channel_attn's residual
attention + LayerNorm, not exact, but grounded by the tokenizer's own training).

cross_channel_attn is used FROZEN here (not fine-tuned): x_tilde's "this is
decodable back to physical units via the affine formula above" property only holds
because cross_channel_attn was trained (as part of the tokenizer's softDTW+MSE
reconstruction objective, train_tokenizer.py) to keep its output shape-faithful.
Letting it drift under a pure forecasting objective, with no reconstruction anchor,
would untether that guarantee -- deferred, per the "incorporate properly later" plan
mu/sigma are also on.
"""
import pickle

import torch


@torch.no_grad()
def compute_x_tilde(tokenizer, x_phys):
    """tokenizer: a loaded (eval-mode) GormpoTokenizer checkpoint.
    x_phys: (B, N, k) raw physical-unit patch, N=12 channels including Pump Level.
    Returns (x_tilde, mu, sigma): x_tilde (B, N, k) cross-channel-attended +
    per-channel-normalized; mu, sigma (B, N) each -- the per-channel patch mean/std
    needed for the affine decode back to physical units."""
    x_bar, mu, sigma = tokenizer.patch_and_scale(x_phys)
    x_tilde = tokenizer.cross_channel_attn(x_bar)
    return x_tilde, mu, sigma


def affine_decode(x_tilde_pred, mu, sigma):
    """Inverse of patch_and_scale's affine step, applied directly to a predicted
    x_tilde (approximation -- see module docstring for why this isn't an exact
    inverse, and why it's the right tradeoff here anyway).
    x_tilde_pred: (..., k). mu, sigma: (...,) broadcastable to x_tilde_pred's leading
    dims. Returns physical-unit values, same shape as x_tilde_pred."""
    return sigma.unsqueeze(-1) * x_tilde_pred + mu.unsqueeze(-1)


def load_pump_level_stats(pickle_path="/public/gormpo/10min_1hr_all_data.pkl"):
    """Dataset-wide Pump Level mean/std (raw column 12 of the MCS pickle -- the same
    mean/std every eval script normalizes with). Used to z-score the Pump Level
    covariate GLOBALLY, (P - mean) / std, instead of per patch: per-patch z-scoring
    erases the absolute level (constant P2 and constant P8 both become all zeros) and
    the size of a step change (2->9 looks like 7->8)."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return float(data["mean"][12]), float(data["std"][12])
