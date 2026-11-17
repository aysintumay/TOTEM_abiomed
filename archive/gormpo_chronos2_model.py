"""
GormpoChronos2Model -- plugs the GORMPO tokenizer's discrete token scheme into
Chronos-2's pretrained encoder stack, replacing Chronos-2's native continuous
patch-embedding front/back end (input_patch_embedding / output_patch_embedding) with
discrete per-facet embeddings / classification heads -- the "swap the tokenizer"
experiment for Chronos-2, analogous to gormpo_chronos_tokenizer.py's swap for original
Chronos, but necessarily different in kind (see mcs.md): Chronos-2 has no
ChronosTokenizer-shaped seam to swap a *preprocessing* step into, since nothing in its
native pipeline is ever discretized. Plugging in GORMPO tokens means replacing the
first and last layers with discrete analogues and the loss with classification instead
of quantile regression -- real surgery on the model, not a preprocessing-level swap.

GroupSelfAttention is deliberately NOT reused here (unlike the native
eval_chronos2_world_model.py / train_chronos2.py runs, which rely on it to share
information between the 11 target channels and the Pump Level covariate): GORMPO's own
tokenizer already performs cross-channel attention BEFORE quantization (Eq 10-16,
forecasting/lib/models/tokenizer.py::CrossChannelAttention) -- every token this model
reads has already had cross-channel information baked into it. Also running
Chronos-2's GroupSelfAttention on top would mix channels a second time, at a different
representational level, for no clear benefit and at real risk of the two mixing
mechanisms fighting each other. So only TimeSelfAttention + FeedForward are kept from
each of the 12 pretrained encoder blocks -- GroupSelfAttention's weights are loaded
nowhere, called nowhere; each channel's 3-position sequence (context, [REG], future)
is processed entirely independently of every other channel/row at this stage.

What's kept from the pretrained amazon/chronos-2 checkpoint, unmodified: each block's
TimeSelfAttention + FeedForward sublayers (12 layers, d_model=768), the encoder's
final_layer_norm/dropout, and the 2-row [REG]/[PAD] embedding. Discarded: Patch,
InstanceNorm, input_patch_embedding, output_patch_embedding, and GroupSelfAttention
(all 12 copies) -- none of it is meaningful here.

Sequence layout, 3 positions per row (one row = one channel of one episode, row order
b*N+n matching lib.models.llm_forecaster.flatten_channels' convention used everywhere
else in this project):
  position 0 (context): sum of this channel's context-patch facet embeddings
    (shape0 + shape1 + shape2 + mu + sigma + min + max + reward_map + reward_hr +
    reward_pulsat -- reward is context-only, per tokenizer.py's RewardQuantizer and
    GormpoTokenLLMForecaster.embed()'s convention).
  position 1 ([REG]): Chronos-2's own pretrained separator embedding, reused as-is.
  position 2 (future): for the 11 target channels, a single learned "unknown future,
    please predict" query vector. For the Pump Level covariate row, the REAL future
    patch's facet embeddings -- reusing the SAME embedding tables as the context
    position (no reward, since reward is never part of a *label* block anywhere in
    this project -- see gormpo_chronos_tokenizer.py's label_input_transform). Reusing
    rather than duplicating the embedding tables mirrors Chronos-2's own native
    design: input_patch_embedding is literally the same module applied to both
    patched_context and patched_future; "this is context" vs. "this is the known
    future" is conveyed by POSITION (0 vs 2, via TimeSelfAttention's RoPE), not by
    separate weights.

Output: for each of the 11 target rows, the position-2 hidden state (after the 12
Time+FFN layers) feeds 7 parallel classification heads (shape0/1/2 over the VQ
codebook size, mu/sigma/min/max over the scalar quantizer's bin count) -- predicting
that channel's future patch token in one shot, non-autoregressively (contrast
GormpoTokenLLMForecaster.generate_codes, which needs a token-by-token loop because its
causal-LM backbone does). Decoding heads' argmax ids back to physical units reuses
GormpoTokenizer.decode() exactly as gormpo_chronos_tokenizer.py's output_transform and
GormpoTokenLLMForecaster's inference() already do.
"""
import torch
import torch.nn as nn

from chronos import Chronos2Pipeline
from chronos.chronos2.model import Chronos2Encoder


class GormpoChronos2Model(nn.Module):
    def __init__(
        self,
        chronos2_model_id="amazon/chronos-2",
        num_embeddings=256,
        num_bins=32,
        num_reward_bins=9,
        freeze_backbone=False,
        device=None,
    ):
        super().__init__()
        print(f"Loading pretrained Chronos-2 backbone: {chronos2_model_id}")
        pipeline = Chronos2Pipeline.from_pretrained(chronos2_model_id, device_map=device or "cpu")
        backbone = pipeline.model

        self.hidden_size = backbone.model_dim
        self.reg_embedding = backbone.shared        # [PAD]/[REG] 2-row embedding, pretrained
        assert backbone.chronos_config.use_reg_token, "GormpoChronos2Model assumes use_reg_token=True"
        self.reg_token_id = backbone.config.reg_token_id

        # TimeSelfAttention + FeedForward only, per block -- GroupSelfAttention
        # (self.layer[1] in each Chronos2EncoderBlock) is never stored/called; see
        # module docstring for why. nn.ModuleList so these stay registered (state_dict,
        # .to(device), optimizer param groups) like any other submodule.
        self.time_attn_layers = nn.ModuleList(block.layer[0] for block in backbone.encoder.block)
        self.feed_forward_layers = nn.ModuleList(block.layer[2] for block in backbone.encoder.block)
        self.final_layer_norm = backbone.encoder.final_layer_norm
        self.encoder_dropout = backbone.encoder.dropout
        backbone_dtype = next(self.time_attn_layers.parameters()).dtype

        if freeze_backbone:
            for layer in (self.time_attn_layers, self.feed_forward_layers, self.reg_embedding):
                for p in layer.parameters():
                    p.requires_grad_(False)

        self.num_embeddings = num_embeddings
        self.num_bins = num_bins
        self.num_reward_bins = num_reward_bins

        def _emb(n):
            e = nn.Embedding(n, self.hidden_size, dtype=backbone_dtype)
            e.weight.data.normal_(mean=0.0, std=0.02)
            return e

        def _head(n):
            h = nn.Linear(self.hidden_size, n, dtype=backbone_dtype)
            h.weight.data.normal_(mean=0.0, std=0.02)
            h.bias.data.zero_()
            return h

        # Shared facet embedding tables, used at BOTH the context position and the
        # Pump Level covariate's future position (see module docstring for why this is
        # a deliberate reuse, not a simplification that loses information).
        self.shape0_embedding = _emb(num_embeddings)
        self.shape1_embedding = _emb(num_embeddings)
        self.shape2_embedding = _emb(num_embeddings)
        self.mu_embedding = _emb(num_bins)
        self.sigma_embedding = _emb(num_bins)
        self.min_embedding = _emb(num_bins)
        self.max_embedding = _emb(num_bins)
        # Context-only (reward is never part of a future/label block).
        self.reward_map_embedding = _emb(num_reward_bins)
        self.reward_hr_embedding = _emb(num_reward_bins)
        self.reward_pulsat_embedding = _emb(num_reward_bins)

        # Learned "unknown future, please predict" query -- used at the 11 target
        # rows' future position; a single shared vector, broadcast to every target
        # row/episode.
        self.future_query = nn.Parameter(torch.randn(self.hidden_size, dtype=backbone_dtype) * 0.02)

        # Output heads, read off the future-position hidden state after the encoder.
        self.shape0_head = _head(num_embeddings)
        self.shape1_head = _head(num_embeddings)
        self.shape2_head = _head(num_embeddings)
        self.mu_head = _head(num_bins)
        self.sigma_head = _head(num_bins)
        self.min_head = _head(num_bins)
        self.max_head = _head(num_bins)

    # ── embedding helpers ────────────────────────────────────────────────────────

    def _embed_patch(self, shape_ids, mu, sigma, min_, max_):
        """shape_ids: (rows, 3) long. mu/sigma/min_/max_: (rows,) long.
        Returns (rows, hidden_size). Shared by context and future-known positions."""
        return (
            self.shape0_embedding(shape_ids[:, 0]) + self.shape1_embedding(shape_ids[:, 1])
            + self.shape2_embedding(shape_ids[:, 2])
            + self.mu_embedding(mu) + self.sigma_embedding(sigma)
            + self.min_embedding(min_) + self.max_embedding(max_)
        )

    # ── per-row-independent encoder (Time + FeedForward only, no cross-row mixing) ─

    def _encode(self, inputs_embeds):
        """inputs_embeds: (rows, 3, H). Returns (rows, 3, H) -- every row processed
        fully independently (no GroupSelfAttention, no cross-row attention of any
        kind); only each row's own 3-position sequence attends to itself."""
        rows, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_len, device=inputs_embeds.device, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones(rows, seq_len, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        extended_attention_mask = Chronos2Encoder._expand_and_invert_time_attention_mask(
            attention_mask, inputs_embeds.dtype
        )

        hidden_states = inputs_embeds
        for time_attn, feed_forward in zip(self.time_attn_layers, self.feed_forward_layers):
            attn_out = time_attn(hidden_states, position_ids=position_ids, attention_mask=extended_attention_mask)
            hidden_states = attn_out[0]
            hidden_states = feed_forward(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.encoder_dropout(hidden_states)
        return hidden_states

    # ── forward ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        ctx_shape_ids, ctx_mu, ctx_sigma, ctx_min, ctx_max, ctx_r_map, ctx_r_hr, ctx_r_pulsat,
        pump_level_row_mask,
        future_shape_ids=None, future_mu=None, future_sigma=None, future_min=None, future_max=None,
    ):
        """
        All context args: (rows,) or (rows, 3) long, rows = B*N (row = b*N+n, matching
        llm_forecaster.flatten_channels's ordering -- caller flattens the same way
        train_gormpo_llm_forecaster.py already does). No num_channels/group_ids
        argument needed: rows are processed fully independently of each other.
        pump_level_row_mask: (rows,) bool -- True at the Pump Level covariate row of
        each episode, False at the 11 target rows.
        future_shape_ids/future_mu/.../future_max: (rows,) or (rows, 3) -- the KNOWN
        future Pump Level patch token; only read at rows where pump_level_row_mask is
        True. Required if pump_level_row_mask has any True entries.

        Returns:
            shape_logits: tuple of 3 (rows, num_embeddings) tensors (c0, c1, c2)
            scalar_logits: dict of 4 (rows, num_bins) tensors (mu, sigma, min, max)
            Both are meaningful (trained/supervised) ONLY at rows where
            pump_level_row_mask is False -- caller masks accordingly before computing
            loss/decoding.
        """
        rows = ctx_shape_ids.shape[0]

        context_embeds = self._embed_patch(ctx_shape_ids, ctx_mu, ctx_sigma, ctx_min, ctx_max) + (
            self.reward_map_embedding(ctx_r_map) + self.reward_hr_embedding(ctx_r_hr)
            + self.reward_pulsat_embedding(ctx_r_pulsat)
        )  # (rows, H)

        reg_ids = torch.full((rows, 1), self.reg_token_id, device=ctx_shape_ids.device, dtype=torch.long)
        reg_embeds = self.reg_embedding(reg_ids).squeeze(1)  # (rows, H)

        future_embeds = self.future_query.unsqueeze(0).expand(rows, -1).clone()  # (rows, H)
        if pump_level_row_mask.any():
            pl_future = self._embed_patch(future_shape_ids, future_mu, future_sigma, future_min, future_max)
            future_embeds = torch.where(pump_level_row_mask.unsqueeze(-1), pl_future, future_embeds)

        inputs_embeds = torch.stack([context_embeds, reg_embeds, future_embeds], dim=1)  # (rows, 3, H)
        hidden = self._encode(inputs_embeds)  # (rows, 3, H)
        future_hidden = hidden[:, 2, :]  # (rows, H) -- position 2's post-encoder representation

        shape_logits = (
            self.shape0_head(future_hidden), self.shape1_head(future_hidden), self.shape2_head(future_hidden),
        )
        scalar_logits = {
            "mu": self.mu_head(future_hidden), "sigma": self.sigma_head(future_hidden),
            "min": self.min_head(future_hidden), "max": self.max_head(future_hidden),
        }
        return shape_logits, scalar_logits
