"""
GORMPO tokenizer + a small, randomly-initialized Transformer forecaster -- the
non-LLM counterpart to GormpoTokenLLMForecaster (lib/models/gormpo_llm_forecaster.py).

Architecturally this mirrors TOTEM's own existing forecaster
(lib/models/decode.py::XcodeYtimeDecoder) rather than the LLM version's causal,
autoregressive generation: bidirectional self-attention over just the context
(x-half) tokens, then a direct flatten+linear projection to *all* target-patch
shape codes and scale-bin predictions at once, in a single forward pass -- no
teacher-forcing/masking, no token-by-token generation loop, no compounding
autoregressive error. Reuses decode.py's own PositionalEncoding/TransformerEncoder
stack instead of hand-rolling a new one, and the same multi-facet token embedding
scheme (shape code + broadcast scalar/reward embeddings) as the LLM version.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.core import BaseModel
from lib.models.decode import PositionalEncoding


class GormpoTransformerForecaster(BaseModel):
    def __init__(
        self,
        num_embeddings=256,
        num_bins=32,
        num_reward_bins=9,
        TCin=3,
        TCout=3,
        d_model=128,
        nhead=4,
        d_hid=256,
        nlayers=4,
        dropout=0.0,
    ):
        super().__init__()
        self.TCin = TCin
        self.TCout = TCout
        self.num_embeddings = num_embeddings

        self.code_embedding = nn.Embedding(num_embeddings, d_model)
        self.mu_embedding = nn.Embedding(num_bins, d_model)
        self.sigma_embedding = nn.Embedding(num_bins, d_model)
        self.min_embedding = nn.Embedding(num_bins, d_model)
        self.max_embedding = nn.Embedding(num_bins, d_model)
        self.reward_embedding = nn.Embedding(num_reward_bins, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, TCin)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)

        self.shape_head = nn.Linear(d_model * TCin, TCout * num_embeddings)
        self.mu_head = nn.Linear(d_model * TCin, num_bins)
        self.sigma_head = nn.Linear(d_model * TCin, num_bins)
        self.min_head = nn.Linear(d_model * TCin, num_bins)
        self.max_head = nn.Linear(d_model * TCin, num_bins)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_ids, x_mu, x_sigma, x_min, x_max, x_r):
        """
        Args:
            x_ids: (N, TCin) context shape codes
            x_mu, x_sigma, x_min, x_max, x_r: (N,) context-patch scalar bin indices
        Returns:
            shape_logits: (N, TCout, num_embeddings)
            scalars: dict of (N, num_bins) logits for the target patch's mu/sigma/min/max
        """
        code_embeds = self.code_embedding(x_ids)  # (N, TCin, d_model)
        scalar_sum = (
            self.mu_embedding(x_mu) + self.sigma_embedding(x_sigma)
            + self.min_embedding(x_min) + self.max_embedding(x_max)
            + self.reward_embedding(x_r)
        )  # (N, d_model)
        embeds = code_embeds + scalar_sum.unsqueeze(1)  # broadcast over TCin positions

        embeds = embeds.permute(1, 0, 2)  # (TCin, N, d_model), decode.py's convention
        embeds = self.pos_encoder(embeds)
        hidden = self.encoder(embeds)  # (TCin, N, d_model)

        flat = hidden.permute(1, 0, 2).reshape(hidden.shape[1], -1)  # (N, TCin * d_model)
        shape_logits = self.shape_head(flat).view(-1, self.TCout, self.num_embeddings)
        scalars = {
            "mu": self.mu_head(flat), "sigma": self.sigma_head(flat),
            "min": self.min_head(flat), "max": self.max_head(flat),
        }
        return shape_logits, scalars

    def configure_optimizers(self, lr=1e-3, weight_decay=0.0):
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

    def shared_eval(self, batch, optimizer, mode, comet_logger=None):
        raise NotImplementedError(
            "train_gormpo_transformer_forecaster.py implements its own train/eval loop; "
            "shared_eval is not used for this model."
        )
