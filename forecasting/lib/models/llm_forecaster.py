import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

from lib.models.core import BaseModel

"""
Autoregressive forecaster head that reuses a pretrained causal LLM's backbone as a
next-code-id predictor over TOTEM's VQVAE codebook, instead of the from-scratch
bidirectional Transformer in decode.py (XcodeYtimeDecoder).

The LLM's own text vocabulary/embedding/lm_head are meaningless for VQ code ids, so
they are never touched: only the inner decoder stack (backbone.model) is reused and
wrapped in LoRA, while a fresh embedding + head map to/from TOTEM's codebook space.
"""


def flatten_channels(x):
    """(B, T, S) -> (B * S, T), row-major so unflatten_channels is a plain reshape."""
    B, T, S = x.shape
    return x.permute(0, 2, 1).reshape(B * S, T)


def unflatten_channels(x, B, S):
    """(B * S, T) -> (B, S, T)"""
    T = x.shape[-1]
    return x.reshape(B, S, T)


def build_training_sequence(x_ids, y_ids):
    """
    Args:
        x_ids: (N, TCin)
        y_ids: (N, TCout)
    Returns:
        inputs: (N, TCin + TCout - 1)
        labels: (N, TCin + TCout - 1) with x-portion masked to -100
    """
    TCin = x_ids.shape[1]
    full = torch.cat([x_ids, y_ids], dim=1)  # (N, TCin + TCout)
    inputs = full[:, :-1]
    labels = full[:, 1:].clone()
    labels[:, : TCin - 1] = -100
    return inputs, labels


class LLMCodeForecaster(BaseModel):
    def __init__(
        self,
        llm_name=None,
        num_embeddings=256,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=None,
        hf_model=None,
    ):
        """
        Args:
            llm_name: HF hub id of a pretrained causal LM (e.g. "Qwen/Qwen2.5-0.5B").
                Ignored if hf_model is given.
            hf_model: an already-constructed *ForCausalLM instance to wrap instead of
                downloading llm_name. Used by the offline smoke test to exercise this
                class against a tiny randomly-initialized model of the same
                architecture family, without a network call.
        """
        super().__init__()
        if lora_target_modules is None:
            lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        if hf_model is None:
            hf_model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype="auto")
        backbone = hf_model.model  # inner decoder stack; hf_model.lm_head is discarded
        self.hidden_size = backbone.config.hidden_size
        # backbone loads in its checkpoint's native dtype (e.g. bfloat16 for Qwen2.5);
        # the fresh embedding/head below must match it or matmuls dtype-mismatch.
        backbone_dtype = next(backbone.parameters()).dtype

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        # activation memory for a forward/backward pass scales with N * L * hidden
        # (and per-layer MLP intermediate size); N here is batch_size * num_channels
        # after flatten_channels, so it gets large fast. Trade recompute for memory.
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.code_embedding = nn.Embedding(num_embeddings, self.hidden_size, dtype=backbone_dtype)
        self.lm_head = nn.Linear(self.hidden_size, num_embeddings, dtype=backbone_dtype)
        self.code_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.lm_head.weight.data.normal_(mean=0.0, std=0.02)
        self.lm_head.bias.data.zero_()

    def forward(self, input_ids):
        """
        Args:
            input_ids: (N, L) code ids
        Returns:
            logits: (N, L, num_embeddings)
        """
        inputs_embeds = self.code_embedding(input_ids)
        out = self.backbone(inputs_embeds=inputs_embeds, use_cache=False)
        return self.lm_head(out.last_hidden_state)

    @torch.no_grad()
    def generate_codes(self, x_ids, TCout):
        """
        Args:
            x_ids: (N, TCin) code ids
            TCout: int, number of future code ids to generate
        Returns:
            y_ids: (N, TCout)
        """
        self.eval()
        generated = x_ids
        for _ in range(TCout):
            logits = self.forward(generated)  # (N, L, num_embeddings)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (N, 1)
            generated = torch.cat([generated, next_id], dim=1)
        return generated[:, -TCout:]

    def shared_eval(self, batch, optimizer, mode, comet_logger=None):
        raise NotImplementedError(
            "train_llm_forecaster.py implements its own train/eval loop; "
            "shared_eval is not used for this model."
        )
