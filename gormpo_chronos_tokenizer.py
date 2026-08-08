"""
GormpoChronosTokenizer -- a chronos.ChronosTokenizer that wraps the trained GORMPO
tokenizer (forecasting/lib/models/tokenizer.py::GormpoTokenizer) in place of
MeanScaleUniformBins, so a pretrained Chronos T5/GPT2 backbone can be retrained to
predict GORMPO's discrete patch tokens instead of Chronos's own per-timestep value bins.

Ordinality audit (this is the reason the design looks the way it does): GormpoTokenizer's
token facets split into two kinds, and are treated differently here as a result --
  - shape codes (q_shape, L per channel per patch): arbitrary VQ codebook indices,
    genuinely categorical -- code 200 is not "bigger" than code 50 (see
    gormpo_world_model.py's own system prompt, which has to say this explicitly to the
    medllama few-shot prompt for the same reason). No ordinal structure to preserve or
    lose; a wrong-but-adjacent id is exactly as wrong as a random one.
  - mu/sigma/min/max (q_mu/q_sigma/q_min/q_max) and reward: ScalarBinQuantizer /
    RewardQuantizer bins, ALREADY ordinal by construction (ScalarBinQuantizer.fit's
    edges come from torch.quantile, monotonically increasing across bin index; reward
    is 0=stable..8=critical). These behave like Chronos's own MeanScaleUniformBins
    tokens (nearby id = nearby value) with no extra work -- no need to round-trip them
    through Chronos's own mean-scale+bucketize, which would be the wrong tool anyway
    (that logic derives its scale from a multi-step series; mu/sigma/min/max are
    already-computed per-patch scale statistics, not raw fluctuating observations).
Every facet still has to be offset into one flat integer vocabulary -- that part is
mechanical, forced by ChronosTokenizer's single-int-per-position contract -- but only
the scalar facets get to keep the "nearby token id = similar value" property a
pretrained Chronos backbone would try to exploit. The shape facet is along for the ride
as an opaque categorical id, same as everywhere else it's used in this codebase
(LLMCodeForecaster.code_embedding, GormpoTokenLLMForecaster.code_embedding: both plain
nn.Embedding, no ordinality assumed there either).

Sequence layout, one channel's patch:
  context (reward included, never predicted): [shape x L] [mu] [sigma] [min] [max] [reward]
  label   (what the model must predict):      [shape x L] [mu] [sigma] [min] [max]
One channel = one Chronos "series" (batch dim) -- matches chronos_world_model.py's
per-channel-independent calls. Cross-channel information isn't lost by treating
channels independently *here*: GormpoTokenizer.encode()'s cross-channel attention
already ran before flattening, so it's baked into the codes, not discarded by this step.

Deviation from MeanScaleUniformBins's contract worth flagging explicitly: context/label
here are raw (B, N, k) multivariate physical-unit patches (batch, channel, raw
timestep within one patch), not (B, time_length) scalar series -- GormpoTokenizer.encode()
is irreducibly multivariate, so the channel dimension is folded into the returned batch
dimension (B*N rows out), not before these methods are called. context_length/
prediction_length in the ChronosConfig this tokenizer is used with must be set in
*token-position* units (L+5 / L+4), not raw timesteps.
"""
import torch
import torch.nn.functional as F

from chronos import ChronosTokenizer


class GormpoChronosTokenizer(ChronosTokenizer):
    def __init__(self, gormpo_tokenizer, config, num_reward_bins=9):
        self.config = config
        self.gormpo_tokenizer = gormpo_tokenizer
        self.num_embeddings = gormpo_tokenizer.vq._embedding.weight.shape[0]
        self.num_bins = gormpo_tokenizer.q_mu.num_bins
        self.num_reward_bins = num_reward_bins
        self.L = gormpo_tokenizer.patch_len // gormpo_tokenizer.compression_factor

        offset = config.n_special_tokens
        self.shape_offset, offset = offset, offset + self.num_embeddings
        self.mu_offset, offset = offset, offset + self.num_bins
        self.sigma_offset, offset = offset, offset + self.num_bins
        self.min_offset, offset = offset, offset + self.num_bins
        self.max_offset, offset = offset, offset + self.num_bins
        self.reward_offset, offset = offset, offset + self.num_reward_bins
        self.vocab_size = offset  # required n_tokens for the wrapped HF model's embedding table

        self.label_len = self.L + 4    # shape codes + mu, sigma, min, max
        self.context_len = self.L + 5  # + reward

    # ── encode: raw multivariate patch -> flat offset ids ──────────────────────

    def _encode_and_flatten(self, x_phys: torch.Tensor, include_reward: bool):
        """x_phys: (B, N, k) raw physical values, k = patch_len.
        Returns (B*N, seq_len) flat token ids, row-major B*N ordering (matches
        llm_forecaster.flatten_channels's bn = b*N + n convention)."""
        tokens = self.gormpo_tokenizer.encode(x_phys)
        B, N, L = tokens["q_shape"].shape
        assert L == self.L, f"tokenizer patch_len/compression_factor gives L={L}, tokenizer configured for L={self.L}"

        parts = [
            tokens["q_shape"] + self.shape_offset,                    # (B, N, L)
            (tokens["q_mu"] + self.mu_offset).unsqueeze(-1),          # (B, N, 1)
            (tokens["q_sigma"] + self.sigma_offset).unsqueeze(-1),
            (tokens["q_min"] + self.min_offset).unsqueeze(-1),
            (tokens["q_max"] + self.max_offset).unsqueeze(-1),
        ]
        if include_reward:
            parts.append((tokens["q_r"] + self.reward_offset).unsqueeze(-1))

        flat = torch.cat(parts, dim=-1)          # (B, N, seq_len)
        return flat.reshape(B * N, -1).long()    # (B*N, seq_len)

    def context_input_transform(self, context: torch.Tensor):
        """context: (B, N, k) raw physical-unit patch.
        tokenizer_state is None: unlike MeanScaleUniformBins's per-series scale,
        GormpoTokenizer's own mu/sigma quantizers already capture per-patch scale
        inside the token ids themselves, so label_input_transform needs nothing
        carried over from here."""
        token_ids = self._encode_and_flatten(context, include_reward=True)
        attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
        return token_ids, attention_mask, None

    def label_input_transform(self, label: torch.Tensor, tokenizer_state=None):
        token_ids = self._encode_and_flatten(label, include_reward=False)
        attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
        return token_ids, attention_mask

    # ── decode: flat offset ids -> raw physical values ──────────────────────────

    def output_transform(self, samples: torch.Tensor, tokenizer_state=None) -> torch.Tensor:
        """samples: (B*N, num_samples, label_len) generated label-block token ids ->
        (B*N, num_samples, k) physical-unit values.

        Ids are clamped into each facet's valid range before touching the
        codebook/dequantize tables -- a freely-generating LM can emit any id in the
        combined vocab at any position (nothing at the model level enforces "position 0
        is a shape id"), and an out-of-range index crashes CUDA with an unrecoverable
        device-side assert deep inside dequantize, not a catchable Python exception
        (same hazard flagged in gormpo_world_model.py's _parse_response)."""
        BN, S, seq_len = samples.shape
        assert seq_len == self.label_len, f"expected {self.label_len} token positions per label block, got {seq_len}"
        N = self.gormpo_tokenizer.num_channels
        assert BN % N == 0, f"batch dim {BN} must be a multiple of num_channels={N} (row order: b*N + n)"
        B = BN // N
        flat = samples.reshape(BN * S, seq_len)

        shape_ids = (flat[:, : self.L] - self.shape_offset).clamp(0, self.num_embeddings - 1)          # (BN*S, L)
        mu_ids = (flat[:, self.L] - self.mu_offset).clamp(0, self.num_bins - 1)                          # (BN*S,)
        sigma_ids = (flat[:, self.L + 1] - self.sigma_offset).clamp(0, self.num_bins - 1)
        # min/max ids are decoded implicitly through the shape codes (they steer the
        # cross-channel-attended patch shape at encode time, Eq 17); GormpoTokenizer.decode
        # only needs mu/sigma for the final affine restoration (Eq 28), same as
        # gormpo_world_model.py's _detokenize_channels -- min/max ids are still range-clamped
        # above (never left unvalidated) even though unused past that point.

        def _dequantize_per_channel(ids_bn_s, quantizer):
            """ids_bn_s: (B*N, S) with row order b*N+n -> per-row channel-correct dequantize.
            ScalarBinQuantizer fit a separate set of bin edges per channel (edges: (N,
            num_bins+1)); dequantize() indexes edges by an id tensor's LAST dimension, so
            channel identity must be that last dimension, not folded into a flat B*N batch
            (which would silently always use channel 0's edges)."""
            ids = ids_bn_s.view(B, N, S).permute(0, 2, 1).reshape(B * S, N)  # (B*S, N), channel = last dim
            vals = quantizer.dequantize(ids)                                 # (B*S, N)
            return vals.view(B, S, N).permute(0, 2, 1).reshape(BN, S)        # back to (B*N, S) row order

        mu_hat = _dequantize_per_channel(mu_ids.view(BN, S), self.gormpo_tokenizer.q_mu).reshape(BN * S, 1)
        sigma_hat = _dequantize_per_channel(sigma_ids.view(BN, S), self.gormpo_tokenizer.q_sigma).reshape(BN * S, 1)

        codebook = self.gormpo_tokenizer.vq._embedding.weight  # (num_embeddings, embedding_dim)
        one_hot = F.one_hot(shape_ids.reshape(-1), self.num_embeddings).to(codebook.dtype)
        quantized = torch.matmul(one_hot, codebook).view(BN * S, self.L, -1).transpose(1, 2)  # (BN*S, emb_dim, L)

        # decode() only uses mu/sigma.shape=(B',N') to reshape its flat output back to a grid
        # (view(B',N',-1) inside GormpoTokenizer.decode) -- it doesn't assign per-channel
        # meaning to that grouping, so B'=BN*S, N'=1 is a safe, unrelated-to-N choice here;
        # the channel-correctness was already resolved above, before this call.
        x_hat, _ = self.gormpo_tokenizer.decode(quantized, mu_hat, sigma_hat)  # (BN*S, 1, k)
        return x_hat.squeeze(1).view(BN, S, -1)  # (B*N, num_samples, k)
