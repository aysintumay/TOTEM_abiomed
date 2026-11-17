"""
TimeOnlyChronos2Encoder -- a drop-in replacement for Chronos2Encoder (the Time+Group+
FeedForward stack inside a loaded Chronos2Model) that skips GroupSelfAttention
entirely, keeping only each block's TimeSelfAttention + FeedForward sublayers
(pretrained weights, reused as-is). Same technique gormpo_chronos2_model.py's
_encode() already uses for the discrete-token architecture, applied here instead to
Chronos-2's NATIVE continuous pipeline: swap `model.encoder` for this wrapper right
after loading a pretrained Chronos2Model, and everything else (InstanceNorm, Patch,
input_patch_embedding, output_patch_embedding, _compute_loss, the whole quantile-
regression machinery) stays completely untouched/native -- Chronos2Pipeline.fit()/
predict() keep working unmodified, since this wrapper preserves Chronos2Encoder's own
forward() signature exactly.

Why: train_chronos2_xtilde.py / train_chronos2_xtilde_jointtrain.py feed GORMPO's
cross-channel-attended x_tilde into Chronos-2's UNMODIFIED native pipeline, which still
has its own GroupSelfAttention active -- two independent cross-channel-mixing
mechanisms stacked on top of each other (cross_channel_attn, then GroupSelfAttention),
never deliberately reasoned about the way gormpo_chronos2_model.py's removal of
GroupSelfAttention was. Diagnostic evidence this double-mixing may matter: LVEDP and
SYSTOLIC are barely related in the real data (|corr|=0.067) but strongly entangled in
that model's predictions (|corr|=0.510). This module exists to test a variant with
BOTH cross-channel-mixing mechanisms removed (see train_chronos2_xbar_notimegroup.py),
not just one.
"""
import torch

from chronos.chronos2.model import Chronos2Encoder, Chronos2EncoderOutput


class TimeOnlyChronos2Encoder(torch.nn.Module):
    def __init__(self, chronos2_encoder: Chronos2Encoder):
        super().__init__()
        # nn.ModuleList of the ORIGINAL Chronos2EncoderBlock instances, reused by
        # reference (not copied) -- their .layer[0] (TimeSelfAttention) and .layer[2]
        # (FeedForward) keep their pretrained weights; .layer[1] (GroupSelfAttention)
        # is simply never called, so its weights are loaded but inert (not removed
        # from the state_dict -- harmless, just unused compute/params).
        # Kept named "block" (not "blocks") deliberately: state_dict() key compatibility
        # with the native Chronos2Encoder matters for save_pretrained()/checkpoint
        # loading -- a renamed attribute would produce "encoder.blocks.*" keys that
        # mismatch "encoder.block.*" on load. NOTE this does NOT make fit() safe to use
        # (see train_chronos2_xbar_notimegroup.py's docstring for why fit() must still
        # be bypassed), and does NOT make a plain Chronos2Pipeline.from_pretrained() of
        # a checkpoint saved this way skip GroupSelfAttention at inference automatically
        # -- strip_group_self_attention() must be called again after loading, every time.
        self.block = chronos2_encoder.block
        self.final_layer_norm = chronos2_encoder.final_layer_norm
        self.dropout = chronos2_encoder.dropout

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        group_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> Chronos2EncoderOutput:
        batch_size, seq_length = inputs_embeds.size()[:-1]

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        extended_attention_mask = Chronos2Encoder._expand_and_invert_time_attention_mask(
            attention_mask, inputs_embeds.dtype
        )
        # group_ids is accepted (so callers / Chronos2Model.encode() don't need to
        # change at all) but deliberately unused -- that's the whole point of this class.

        hidden_states = self.dropout(inputs_embeds)
        for block in self.block:
            time_attn, feed_forward = block.layer[0], block.layer[2]
            attn_out = time_attn(hidden_states, position_ids=position_ids, attention_mask=extended_attention_mask)
            hidden_states = attn_out[0]
            hidden_states = feed_forward(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return Chronos2EncoderOutput(
            last_hidden_state=hidden_states,
            all_time_self_attn_weights=None,
            all_group_self_attn_weights=None,
        )


def strip_group_self_attention(model):
    """Replaces model.encoder (a Chronos2Encoder) with a TimeOnlyChronos2Encoder
    in-place, on an already-loaded Chronos2Model. Call once, right after
    Chronos2Pipeline.from_pretrained(...) -- everything downstream (fit(), predict(),
    the Trainer) keeps working unmodified."""
    model.encoder = TimeOnlyChronos2Encoder(model.encoder)
    return model
