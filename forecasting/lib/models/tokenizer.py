"""
GORMPO tokenizer (Section 2.1 of "First Principles of Time Series Tokenizers").

Reuses a frozen, pretrained TOTEM VQ-VAE (lib.models.vqvae.vqvae) as the encoder E:
its convolutional Encoder stays frozen throughout, while its VectorQuantizer
(codebook) and Decoder are warm-started from the checkpoint but kept trainable, and
are jointly finetuned alongside a new cross-channel attention block on MCS data.

Pipeline per patch (Eq 7-28): raw multivariate patch -> per-channel scale
extraction + normalization -> cross-channel attention -> frozen encoder (applied
per-channel) -> vector quantization (trainable codebook) -> decoder (trainable) ->
affine restoration. Scalar patch statistics (mu, sigma, min, max) and the clinical
reward channels (MAP/HR/Pulsatility) are quantized separately via fixed bins,
bypassing the codebook entirely, and assembled into the per-channel token tuple.

Out of scope here: the downstream LLM world model (Eq 26-28) and MBRL/RL wiring.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.core import BaseModel
from lib.models.metrics import soft_dtw
from lib.models.vqvae import vqvae

_ABIOMED_ENV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ABIOMED_ENV_ROOT not in sys.path:
    sys.path.insert(0, _ABIOMED_ENV_ROOT)
from abiomed_env.cost_func import HR_IDX, MAP_IDX, PULSATILITY_IDX  # noqa: E402


class CrossChannelAttention(nn.Module):
    """Eq 10-16: single-head attention over the N channel rows of a normalized
    multivariate patch, shared projections, applied independently per time patch."""

    def __init__(self, patch_len, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_Q = nn.Linear(patch_len, d_model, bias=False)
        self.W_K = nn.Linear(patch_len, d_model, bias=False)
        self.W_V = nn.Linear(patch_len, d_model, bias=False)
        self.W_O = nn.Linear(d_model, patch_len, bias=False)
        self.layer_norm = nn.LayerNorm(patch_len)

    def forward(self, x_bar):
        """x_bar: (B, N, k) normalized patch matrix -> (B, N, k) cross-attended."""
        q = self.W_Q(x_bar)  # (B, N, d)
        k = self.W_K(x_bar)  # (B, N, d)
        v = self.W_V(x_bar)  # (B, N, d)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_model)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, N, N)
        h = torch.matmul(attn_weights, v)  # (B, N, d)
        h_tilde = self.W_O(h)  # (B, N, k)
        return self.layer_norm(x_bar + h_tilde)


class ScalarBinQuantizer:
    """Fixed per-channel quantile-bin quantizer for a scalar patch statistic
    (Eq 20-23). Bin edges are computed once via fit() on training-split statistics
    and are not updated afterward (they are not nn.Module parameters)."""

    def __init__(self, num_bins=32):
        self.num_bins = num_bins
        self.edges = None  # (N, num_bins + 1): per-channel [min, q_1, ..., q_{B-1}, max]

    def fit(self, values):
        """values: (num_samples, N) tensor of the raw statistic across the train split."""
        qs = torch.linspace(0, 1, self.num_bins + 1, device=values.device)
        edges = torch.quantile(values.float(), qs, dim=0)  # (num_bins + 1, N)
        self.edges = edges.transpose(0, 1).contiguous()  # (N, num_bins + 1)

    @property
    def centers(self):
        return (self.edges[:, :-1] + self.edges[:, 1:]) / 2  # (N, num_bins)

    def quantize(self, x):
        """x: (..., N) raw statistic -> (..., N) LongTensor bin index in [0, num_bins - 1]."""
        assert self.edges is not None, "call fit() before quantize()"
        edges = self.edges.to(x.device)
        idx = torch.empty_like(x, dtype=torch.long)
        for c in range(x.shape[-1]):
            boundaries = edges[c, 1:-1]  # (num_bins - 1,) interior edges
            idx[..., c] = torch.bucketize(x[..., c].contiguous(), boundaries)
        return idx.clamp_(0, self.num_bins - 1)

    def dequantize(self, idx):
        """idx: (..., N) bin index -> (..., N) bin-center float estimate."""
        centers = self.centers.to(idx.device)
        out = torch.empty(idx.shape, dtype=centers.dtype, device=idx.device)
        for c in range(idx.shape[-1]):
            out[..., c] = centers[c][idx[..., c]]
        return out

    def state_dict(self):
        return {"num_bins": self.num_bins, "edges": self.edges}

    def load_state_dict(self, state):
        self.num_bins = state["num_bins"]
        self.edges = state["edges"]


class RewardQuantizer:
    """Per-channel reward token quantizer (Eq 24). Only the MAP/HR/Pulsatility
    channels carry a reward token; every other channel's token is 0.

    Penalty formulas mirror abiomed_env/reward_func.py's min_map_penalty,
    hypertention_penalty, hr_penalty, and pulsat_penalty exactly (same constants
    and thresholds), reimplemented batched over (B, N, k) patches -- the originals
    reduce with a single scalar torch.min over the whole input tensor and are only
    correct for one (k, N) window at a time, not a training batch.
    """

    MAP_IDX = MAP_IDX
    HR_IDX = HR_IDX
    PULSATILITY_IDX = PULSATILITY_IDX

    @staticmethod
    def _min_map_penalty(map_min):
        return F.relu(7 * (60 - map_min) / 20)

    @staticmethod
    def _hypertension_penalty(map_mean):
        return F.relu((map_mean - 106) / 18)

    @staticmethod
    def _hr_penalty(hr_min):
        return F.relu((hr_min - 75) ** 2 / 250 - 1)

    @staticmethod
    def _pulsat_penalty(pulsat_min):
        return F.relu(7 * (20 - pulsat_min) / 20) + F.relu((pulsat_min - 50) / 20)

    def __call__(self, raw_patch):
        """raw_patch: (B, N, k) unnormalized physiological units -> (B, N) LongTensor in {0,...,8}."""
        B, N, _ = raw_patch.shape
        q_r = torch.zeros(B, N, dtype=torch.long, device=raw_patch.device)

        map_patch = raw_patch[:, self.MAP_IDX, :]
        hr_patch = raw_patch[:, self.HR_IDX, :]
        pulsat_patch = raw_patch[:, self.PULSATILITY_IDX, :]

        r_map = -(self._min_map_penalty(map_patch.min(dim=1).values)
                  + self._hypertension_penalty(map_patch.mean(dim=1)))
        r_hr = -self._hr_penalty(hr_patch.min(dim=1).values)
        r_pulsat = -self._pulsat_penalty(pulsat_patch.min(dim=1).values)

        q_r[:, self.MAP_IDX] = (r_map.clamp(-7, 0) + 8).round().long()
        q_r[:, self.HR_IDX] = (r_hr.clamp(-7, 0) + 8).round().long()
        q_r[:, self.PULSATILITY_IDX] = (r_pulsat.clamp(-7, 0) + 8).round().long()
        return q_r


class GormpoTokenizer(BaseModel):
    def __init__(
        self,
        pretrained_vqvae,
        patch_len,
        num_channels,
        d_model=32,
        num_bins=32,
        softdtw_gamma=0.1,
        lambda_sdtw=1.0,
        use_reward=True,
        freeze_encoder=True,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.num_channels = num_channels
        self.compression_factor = pretrained_vqvae.compression_factor
        self.lambda_sdtw = lambda_sdtw
        self.softdtw_gamma = softdtw_gamma
        self.use_reward = use_reward
        self.freeze_encoder = freeze_encoder

        # Encoder E (Eq 18): shared across channels. When freeze_encoder=True this is a
        # frozen, borrowed-weights encoder pretrained on plain (non-cross-attended)
        # patches -- appropriate when its input distribution still matches what it saw
        # in pretraining. When False, it's trained jointly with everything else instead:
        # a pretrained encoder's filters were tuned for plain-normalized patches, not the
        # cross-attended patches (Eq 10-16) it actually receives here, so letting it adapt
        # (or training it from scratch via from_scratch()) avoids that distribution
        # mismatch at the cost of losing the "frozen bottleneck" guarantee of Axiom 5.
        self.encoder = pretrained_vqvae.encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        # Trainable, warm-started from the pretrained checkpoint when one is given
        # (Eq 29: jointly optimized); random-initialized when built via from_scratch().
        self.vq = pretrained_vqvae.vq
        self.decoder = pretrained_vqvae.decoder

        # New trainable cross-channel attention block (Eq 10-16).
        self.cross_channel_attn = CrossChannelAttention(patch_len, d_model)

        # Fixed (non-gradient) scalar + reward quantizers (Eq 20-24); fit externally.
        self.q_mu = ScalarBinQuantizer(num_bins)
        self.q_sigma = ScalarBinQuantizer(num_bins)
        self.q_min = ScalarBinQuantizer(num_bins)
        self.q_max = ScalarBinQuantizer(num_bins)
        # Reward tokens (Eq 24) only make sense for datasets with MAP/HR/Pulsatility
        # channels at the abiomed_env.cost_func indices (MCS). Disable for others.
        self.reward_quantizer = RewardQuantizer() if use_reward else None

    @torch.no_grad()
    def calibrate_codebook(self, x_sample):
        """Re-initialize the VQ codebook from real encoder outputs on a sample batch,
        instead of VectorQuantizer's default tiny fixed-scale uniform init. Necessary
        when the encoder is also freshly initialized (from_scratch()): a random
        encoder's output scale has no relation to the codebook's default init scale,
        so nearest-neighbor lookup (Eq 19) starts from a degenerate mismatch (observed
        empirically: z std ~0.11 vs. default codebook std ~0.002 for MCS at cf=2),
        which collapses the codebook within a few epochs regardless of learning rate."""
        x_bar, _, _ = self.patch_and_scale(x_sample)
        x_tilde = self.cross_channel_attn(x_bar)
        B, N, k = x_sample.shape
        flat = x_tilde.reshape(B * N, k)
        z = self.encoder(flat, self.compression_factor)  # (B*N, embedding_dim, L)
        z_flat = z.permute(0, 2, 1).reshape(-1, z.shape[1])  # (B*N*L, embedding_dim)

        num_embeddings = self.vq._embedding.weight.shape[0]
        idx = torch.randint(0, z_flat.shape[0], (num_embeddings,), device=z_flat.device)
        self.vq._embedding.weight.data.copy_(z_flat[idx])

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()  # frozen throughout, regardless of the rest of the module's mode
        return self

    def patch_and_scale(self, x, eps=1e-5):
        """Eq 7-9. x: (B, N, k) raw -> (x_bar, mu, sigma), mu/sigma: (B, N)."""
        mu = x.mean(dim=-1)
        sigma = torch.sqrt(x.var(dim=-1, unbiased=False) + eps)
        x_bar = (x - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
        return x_bar, mu, sigma

    def encode(self, x):
        """x: (B, N, k) raw physiological patch -> dict with the full token schema
        (Eq 25) plus the tensors needed to decode/train."""
        B, N, k = x.shape
        x_bar, mu, sigma = self.patch_and_scale(x)  # Eq 7-9

        x_tilde = self.cross_channel_attn(x_bar)  # Eq 10-16

        # Shape proxy F (Eq 17): computed on the attended signal, not fed to the encoder.
        patch_min = x_tilde.min(dim=-1).values
        patch_max = x_tilde.max(dim=-1).values

        flat = x_tilde.reshape(B * N, k)
        # No torch.no_grad() here: encoder params are frozen (requires_grad=False),
        # but gradients must still flow through this input back into cross_channel_attn.
        z = self.encoder(flat, self.compression_factor)  # (B*N, embedding_dim, L)
        vq_loss, quantized, perplexity, _, encoding_indices, _ = self.vq(z)
        L = quantized.shape[-1]
        q_shape = encoding_indices.view(B, N, L)  # one code per channel per patch is L codes (see note below)

        q_mu = self.q_mu.quantize(mu)
        q_sigma = self.q_sigma.quantize(sigma)
        q_min = self.q_min.quantize(patch_min)
        q_max = self.q_max.quantize(patch_max)
        if self.reward_quantizer is not None:
            with torch.no_grad():
                q_r = self.reward_quantizer(x)
        else:
            q_r = torch.zeros(B, N, dtype=torch.long, device=x.device)

        return {
            "q_shape": q_shape, "q_mu": q_mu, "q_sigma": q_sigma,
            "q_min": q_min, "q_max": q_max, "q_r": q_r,
            "quantized": quantized, "vq_loss": vq_loss, "perplexity": perplexity,
            "mu": mu, "sigma": sigma, "x_bar": x_bar, "x_tilde": x_tilde,
        }

    def decode(self, quantized, mu, sigma):
        """quantized: (B*N, embedding_dim, L) as returned by self.vq(). mu/sigma: (B, N).
        Returns (x_hat, x_hat_norm), both (B, N, k) -- raw-scale and normalized-scale
        reconstructions (Eq 28's affine restoration, using ground-truth mu/sigma
        since there is no forecasting step in this tokenizer-only scope)."""
        B, N = mu.shape
        x_hat_norm = self.decoder(quantized, self.compression_factor).view(B, N, -1)
        x_hat = sigma.unsqueeze(-1) * x_hat_norm + mu.unsqueeze(-1)
        return x_hat, x_hat_norm

    def forward(self, x):
        out = self.encode(x)
        x_hat, x_hat_norm = self.decode(out["quantized"], out["mu"], out["sigma"])
        out["x_hat"] = x_hat
        out["x_hat_norm"] = x_hat_norm
        return out

    def configure_optimizers(self, lr=1e-3, encoder_lr=None, vq_lr=None):
        """encoder_lr/vq_lr default to `lr` if not given. Separate rates matter most
        when training the encoder from scratch (freeze_encoder=False): the codebook
        needs to move fast enough to track a still-shifting encoder, while the encoder
        itself needs a slow enough rate that it doesn't destructively chase the
        codebook's own noisy early updates (observed empirically: same-rate training
        left VQ loss 400-1000x larger than recon loss throughout, never converging)."""
        groups = [
            {"params": list(self.cross_channel_attn.parameters()), "lr": lr},
            {"params": list(self.decoder.parameters()), "lr": lr},
            {"params": list(self.vq.parameters()), "lr": vq_lr if vq_lr is not None else lr},
        ]
        if not self.freeze_encoder:
            groups.append({
                "params": list(self.encoder.parameters()),
                "lr": encoder_lr if encoder_lr is not None else lr,
            })
        return torch.optim.Adam(groups, lr=lr)

    def shared_eval(self, batch, optimizer, mode, comet_logger=None):
        """batch: (B, N, k) raw physiological patch. L_tokenizer = L_VQ + L_recon (Eq 29-31)."""

        def _forward_and_loss():
            out = self.forward(batch)
            recon_error = F.mse_loss(out["x_hat"], batch)
            if self.lambda_sdtw > 0:
                # soft_dtw's O(k^2) sequential recurrence gets expensive at large patch
                # lengths (e.g. k=96 for ETTh1 vs k=6 for MCS); skip it entirely when
                # its weight is 0 rather than computing and discarding it.
                sdtw = soft_dtw(
                    out["x_bar"].transpose(1, 2), out["x_hat_norm"].transpose(1, 2),
                    gamma=self.softdtw_gamma,
                )
            else:
                sdtw = torch.zeros((), device=batch.device)
            loss_recon = recon_error + self.lambda_sdtw * sdtw
            loss = out["vq_loss"] + loss_recon
            return out, loss, recon_error, sdtw

        if mode == "train":
            optimizer.zero_grad()
            out, loss, recon_error, sdtw = _forward_and_loss()
            loss.backward()
            optimizer.step()
        elif mode in ("val", "test"):
            with torch.no_grad():
                out, loss, recon_error, sdtw = _forward_and_loss()
        else:
            raise ValueError(f"Unknown mode {mode}")

        if comet_logger is not None:
            comet_logger.log_metric(f"{mode}_tokenizer_loss_each_batch", loss.item())
            comet_logger.log_metric(f"{mode}_tokenizer_vq_loss_each_batch", out["vq_loss"].item())
            comet_logger.log_metric(f"{mode}_tokenizer_recon_loss_each_batch", recon_error.item())
            comet_logger.log_metric(f"{mode}_tokenizer_sdtw_loss_each_batch", sdtw.item())
            comet_logger.log_metric(f"{mode}_tokenizer_perplexity_each_batch", out["perplexity"].item())

        return loss, out

    @classmethod
    def from_pretrained_vqvae(cls, path, patch_len, num_channels, **kwargs):
        pretrained_vqvae = torch.load(path, map_location="cpu", weights_only=False)
        return cls(pretrained_vqvae, patch_len, num_channels, freeze_encoder=True, **kwargs)

    @classmethod
    def from_scratch(cls, vqvae_config, patch_len, num_channels, **kwargs):
        """Random-initialize the encoder/codebook/decoder (lib.models.vqvae.vqvae's own
        init already does this) instead of loading a checkpoint pretrained on a
        different input distribution (plain patches, not cross-attended ones)."""
        fresh_vqvae = vqvae(vqvae_config)
        return cls(fresh_vqvae, patch_len, num_channels, freeze_encoder=False, **kwargs)
