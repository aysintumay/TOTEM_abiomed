"""
Token-conditioned LLM forecaster over the full GORMPO tokenizer output (see
lib/models/tokenizer.py), not just the shape code the way LLMCodeForecaster
(lib/models/llm_forecaster.py) consumes plain TOTEM VQ codes.

Closes the loop the tokenizer proposal's Eq 26-28 actually asks for: the world
model must predict not just the next shape code but also the scale statistics
(mu/sigma/min/max) needed to affine-restore it, since a *future* patch's true
mu/sigma isn't available at inference time -- only the past (context) patch's is.
So the context (x-half) positions are embedded with the full token tuple (shape +
mu/sigma/min/max/reward), while the target (y-half) positions only ever see the
shape code (teacher-forced/generated), and four small classifier heads predict the
target patch's mu/sigma/min/max once, from the hidden state at the context/target
boundary.

Reuses LLMCodeForecaster's frozen-backbone + LoRA pattern and its
flatten_channels/unflatten_channels/build_training_sequence helpers unchanged --
those operate purely on shape-code id tensors and are orthogonal to the new
per-facet embedding/head logic added here.
"""
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from lib.models.core import BaseModel


def flatten_channels_scalar(x):
    """(B, S) -> (B * S,), consistent with flatten_channels' bn = b*S + s ordering."""
    return x.reshape(-1)


def unflatten_channels_scalar(x, B, S):
    """(B * S,) -> (B, S)"""
    return x.reshape(B, S)


class GormpoTokenLLMForecaster(BaseModel):
    def __init__(
        self,
        llm_name=None,
        num_embeddings=256,
        num_bins=32,
        num_reward_bins=9,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=None,
        hf_model=None,
        load_in_4bit=False,
        device=None,
    ):
        super().__init__()
        if lora_target_modules is None:
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        if hf_model is None:
            model_kwargs = {"torch_dtype": "auto"}
            if load_in_4bit:
                # QLoRA: quantize the frozen base weights to 4-bit (e.g. an 8B model's
                # ~16GB bf16 footprint drops to ~4-5GB) so it fits alongside whatever
                # else is already resident on a shared GPU; LoRA adapters/new heads
                # below stay full-precision and trainable as before. device_map pins
                # the quantized weights directly to `device` at load time -- bitsandbytes
                # 4-bit params reject a later whole-model .to(device) call with a
                # different device, so this must happen here, not by the caller.
                model_kwargs["torch_dtype"] = torch.bfloat16
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["device_map"] = {"": device if device is not None else 0}
            hf_model = AutoModelForCausalLM.from_pretrained(llm_name, **model_kwargs)
            if load_in_4bit:
                # This class only ever uses hf_model.model (the inner transformer) below --
                # the LM head is never called (its own shape/scalar heads replace it, and
                # forward() is always given inputs_embeds directly, which also bypasses the
                # input embedding lookup). Drop it now rather than leaving it resident: a
                # peft.prepare_model_for_kbit_training-style fp32 upcast of a Llama-scale
                # (~128k-vocab) head/embedding table costs ~2GB by itself, which is what was
                # blowing the budget here -- every GB matters on a shared GPU running other
                # jobs. LoRA (get_peft_model, below) freezes every non-adapter backbone
                # param on its own, so no separate requires_grad_(False) pass is needed.
                del hf_model.lm_head
        backbone = hf_model.model
        self.hidden_size = backbone.config.hidden_size
        backbone_dtype = next(backbone.parameters()).dtype
        if load_in_4bit:
            # backbone's quantized Params4bit report their storage dtype (uint8), not the
            # bf16 compute dtype the new embeddings/heads below need to stay consistent with.
            backbone_dtype = torch.bfloat16

        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        def _emb(n):
            e = nn.Embedding(n, self.hidden_size, dtype=backbone_dtype)
            e.weight.data.normal_(mean=0.0, std=0.02)
            return e

        def _head(n):
            h = nn.Linear(self.hidden_size, n, dtype=backbone_dtype)
            h.weight.data.normal_(mean=0.0, std=0.02)
            h.bias.data.zero_()
            return h

        # Shape-code embedding/head: same role as LLMCodeForecaster's code_embedding/lm_head.
        self.code_embedding = _emb(num_embeddings)
        self.shape_head = _head(num_embeddings)

        # Context-only scalar-token embeddings (Eq 20-24): added at x-half positions only.
        self.mu_embedding = _emb(num_bins)
        self.sigma_embedding = _emb(num_bins)
        self.min_embedding = _emb(num_bins)
        self.max_embedding = _emb(num_bins)
        # Three separate per-patch reward embeddings (Eq 24-25: q_r_map/q_r_hr/q_r_pulsat),
        # replacing the old single reward_embedding -- see tokenizer.py's RewardQuantizer.
        self.reward_map_embedding = _emb(num_reward_bins)
        self.reward_hr_embedding = _emb(num_reward_bins)
        self.reward_pulsat_embedding = _emb(num_reward_bins)

        # Target-patch scalar heads (Eq 28's "predicted scale statistics"), read off the
        # hidden state at the context/target boundary.
        self.mu_head = _head(num_bins)
        self.sigma_head = _head(num_bins)
        self.min_head = _head(num_bins)
        self.max_head = _head(num_bins)

    def embed(self, input_ids, x_mu=None, x_sigma=None, x_min=None, x_max=None,
              x_r_map=None, x_r_hr=None, x_r_pulsat=None, TCin=None):
        """input_ids: (N, L) shape codes. x_mu/x_sigma/x_min/x_max/x_r_map/x_r_hr/
        x_r_pulsat: (N,) context-patch scalar bin indices, broadcast-added to the first
        TCin positions only (the context/x-half); absent entirely from target/y-half
        positions. The three reward fields are per-patch, not per-channel (see
        tokenizer.py's RewardQuantizer) -- callers broadcast them to (N,) across
        channels before flattening, same as train_gormpo_llm_forecaster.py does."""
        code_embeds = self.code_embedding(input_ids)  # (N, L, H)
        if TCin is not None and x_mu is not None:
            scalar_sum = (
                self.mu_embedding(x_mu) + self.sigma_embedding(x_sigma)
                + self.min_embedding(x_min) + self.max_embedding(x_max)
                + self.reward_map_embedding(x_r_map) + self.reward_hr_embedding(x_r_hr)
                + self.reward_pulsat_embedding(x_r_pulsat)
            )  # (N, H)
            pad = torch.zeros_like(code_embeds)
            pad[:, :TCin, :] = scalar_sum.unsqueeze(1)
            code_embeds = code_embeds + pad
        return code_embeds

    def forward(self, input_ids, x_mu=None, x_sigma=None, x_min=None, x_max=None,
                x_r_map=None, x_r_hr=None, x_r_pulsat=None, TCin=None):
        """
        Returns:
            shape_logits: (N, L, num_embeddings)
            hidden_states: (N, L, hidden_size) -- index [:, TCin - 1, :] is the
                context/target boundary representation used by predict_scalars().
        """
        inputs_embeds = self.embed(input_ids, x_mu, x_sigma, x_min, x_max, x_r_map, x_r_hr, x_r_pulsat, TCin)
        out = self.backbone(inputs_embeds=inputs_embeds, use_cache=False)
        shape_logits = self.shape_head(out.last_hidden_state)
        return shape_logits, out.last_hidden_state

    def predict_scalars(self, boundary_hidden):
        """boundary_hidden: (N, hidden_size) -> dict of (N, num_bins) logits for the
        target patch's mu/sigma/min/max."""
        return {
            "mu": self.mu_head(boundary_hidden),
            "sigma": self.sigma_head(boundary_hidden),
            "min": self.min_head(boundary_hidden),
            "max": self.max_head(boundary_hidden),
        }

    @torch.no_grad()
    def generate_codes(self, x_ids, TCout, x_mu, x_sigma, x_min, x_max, x_r_map, x_r_hr, x_r_pulsat):
        """
        Args:
            x_ids: (N, TCin) context shape codes
            TCout: number of future shape codes to generate
            x_mu, x_sigma, x_min, x_max, x_r_map, x_r_hr, x_r_pulsat: (N,) context-patch
                scalar bin indices
        Returns:
            y_ids: (N, TCout) generated shape codes
            scalars: dict of (N, num_bins) logits, predicted once from the context
        """
        self.eval()
        TCin = x_ids.shape[1]
        generated = x_ids
        boundary_hidden = None
        for step in range(TCout):
            shape_logits, hidden = self.forward(
                generated, x_mu, x_sigma, x_min, x_max, x_r_map, x_r_hr, x_r_pulsat, TCin=TCin
            )
            if step == 0:
                boundary_hidden = hidden[:, TCin - 1, :]
            next_id = torch.argmax(shape_logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)
        scalars = self.predict_scalars(boundary_hidden)
        return generated[:, -TCout:], scalars

    def shared_eval(self, batch, optimizer, mode, comet_logger=None):
        raise NotImplementedError(
            "train_gormpo_llm_forecaster.py implements its own train/eval loop; "
            "shared_eval is not used for this model."
        )
