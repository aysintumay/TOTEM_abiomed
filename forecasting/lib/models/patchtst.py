"""
PatchTST (Nie et al. 2023, "A Time Series is Worth 64 Words") baseline -- see
Section 3.1/4 of the tokenizer proposal for its role as the "channel-independent,
no cross-variate structure" reference point.

Thin BaseModel wrapper around HuggingFace transformers' PatchTSTForPrediction, which
is an officially integrated port of the original architecture (IBM Research
collaborated with the original authors), rather than a from-scratch reimplementation:
gets the real patch-padding, batchnorm-based encoder norm, shared channel-independent
embedding/projection, and built-in instance scaling ('std', equivalent to RevIN) that
the paper's actual configuration uses, instead of approximating them by hand.
"""
import torch
from transformers import PatchTSTConfig, PatchTSTForPrediction

from lib.models.core import BaseModel


class PatchTST(BaseModel):
    def __init__(
        self,
        seq_len,
        pred_len,
        n_vars,
        patch_len=2,
        stride=1,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
    ):
        super().__init__()
        config = PatchTSTConfig(
            num_input_channels=n_vars, context_length=seq_len, prediction_length=pred_len,
            patch_length=patch_len, patch_stride=stride, d_model=d_model,
            num_attention_heads=n_heads, num_hidden_layers=n_layers, ffn_dim=d_ff,
            head_dropout=dropout, positional_dropout=dropout, ff_dropout=dropout,
            loss="mse", scaling="std",  # plain point forecast + built-in RevIN-equivalent
        )
        self.model = PatchTSTForPrediction(config)

    def forward(self, x):
        """x: (B, seq_len, n_vars) raw -> (B, pred_len, n_vars) raw."""
        return self.model(past_values=x).prediction_outputs

    def configure_optimizers(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def shared_eval(self, batch, optimizer, mode, comet_logger=None):
        """batch: (x, y), each (B, T, N) raw physiological units."""
        x, y = batch

        if mode == "train":
            optimizer.zero_grad()
            out = self.model(past_values=x, future_values=y)
            out.loss.backward()
            optimizer.step()
        elif mode in ("val", "test"):
            with torch.no_grad():
                out = self.model(past_values=x, future_values=y)
        else:
            raise ValueError(f"Unknown mode {mode}")

        if comet_logger is not None:
            comet_logger.log_metric(f"{mode}_patchtst_loss_each_batch", out.loss.item())

        return out.loss, out.prediction_outputs
