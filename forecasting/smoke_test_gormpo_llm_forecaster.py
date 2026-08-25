"""
Offline smoke test for lib/models/gormpo_llm_forecaster.py, mirroring
smoke_test_llm_forecaster.py's approach: a tiny randomly-initialized Qwen2 backbone
and random data at the MCS shapes (Tin=Tout=6, compression=2, N=12), no network
access or real trained tokenizer required.

Run with: --real to additionally smoke-test against the real pretrained
Qwen/Qwen2.5-0.5B (requires network access; slower).
"""
import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, Qwen2Config

from lib.models.gormpo_llm_forecaster import (
    GormpoTokenLLMForecaster,
    flatten_channels_scalar,
    unflatten_channels_scalar,
)
from lib.models.llm_forecaster import build_training_sequence, flatten_channels, unflatten_channels


def make_tiny_hf_model():
    cfg = Qwen2Config(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=64, max_position_embeddings=128, vocab_size=8,
    )
    return AutoModelForCausalLM.from_config(cfg)


def run(hf_model, num_embeddings, num_bins):
    B, N = 4, 12
    compression_factor = 2
    patch_len = 6
    TCin = TCout = patch_len // compression_factor

    model = GormpoTokenLLMForecaster(num_embeddings=num_embeddings, num_bins=num_bins, hf_model=hf_model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable} / total params: {total}")
    assert trainable < total, "LoRA should freeze most of the backbone"

    # ----- fake token data at the real extraction shapes ----- #
    codeids_x = torch.randint(0, num_embeddings, (B, TCin, N))
    codeids_y = torch.randint(0, num_embeddings, (B, TCout, N))
    x_mu = torch.randint(0, num_bins, (B, N))
    x_sigma = torch.randint(0, num_bins, (B, N))
    x_min = torch.randint(0, num_bins, (B, N))
    x_max = torch.randint(0, num_bins, (B, N))
    # Three separate per-patch reward fields (Eq 24-25), broadcast to (B, N) same as a
    # real dataloader would (see train_gormpo_llm_forecaster.py::create_gormpo_dataloader).
    x_r_map = torch.randint(1, 9, (B, N))
    x_r_hr = torch.randint(1, 9, (B, N))
    x_r_pulsat = torch.randint(1, 9, (B, N))
    y_mu = torch.randint(0, num_bins, (B, N))
    y_sigma = torch.randint(0, num_bins, (B, N))
    y_min = torch.randint(0, num_bins, (B, N))
    y_max = torch.randint(0, num_bins, (B, N))

    x_ids = flatten_channels(codeids_x)  # (B*N, TCin)
    y_ids = flatten_channels(codeids_y)  # (B*N, TCout)
    x_mu_f, x_sigma_f, x_min_f, x_max_f, x_r_map_f, x_r_hr_f, x_r_pulsat_f = (
        flatten_channels_scalar(t) for t in (x_mu, x_sigma, x_min, x_max, x_r_map, x_r_hr, x_r_pulsat)
    )
    y_mu_f, y_sigma_f, y_min_f, y_max_f = (
        flatten_channels_scalar(t) for t in (y_mu, y_sigma, y_min, y_max)
    )
    assert unflatten_channels_scalar(x_mu_f, B, N).equal(x_mu)

    inputs, labels = build_training_sequence(x_ids, y_ids)
    assert inputs.shape == (B * N, TCin + TCout - 1)

    shape_logits, hidden = model(
        inputs, x_mu_f, x_sigma_f, x_min_f, x_max_f, x_r_map_f, x_r_hr_f, x_r_pulsat_f, TCin=TCin
    )
    assert shape_logits.shape == (B * N, TCin + TCout - 1, num_embeddings)
    assert hidden.shape == (B * N, TCin + TCout - 1, model.hidden_size)

    loss_shape = F.cross_entropy(
        shape_logits.reshape(-1, num_embeddings), labels.reshape(-1), ignore_index=-100
    )
    boundary_hidden = hidden[:, TCin - 1, :]
    scalars_pred = model.predict_scalars(boundary_hidden)
    for name, target in [("mu", y_mu_f), ("sigma", y_sigma_f), ("min", y_min_f), ("max", y_max_f)]:
        assert scalars_pred[name].shape == (B * N, num_bins)
    loss_scalar = sum(
        F.cross_entropy(scalars_pred[name], target)
        for name, target in [("mu", y_mu_f), ("sigma", y_sigma_f), ("min", y_min_f), ("max", y_max_f)]
    )
    loss = loss_shape + loss_scalar
    assert torch.isfinite(loss)
    loss.backward()

    lora_grad_norm = sum(
        p.grad.abs().sum().item()
        for n, p in model.backbone.named_parameters()
        if p.grad is not None and "lora_" in n
    )
    base_grad_present = any(
        p.grad is not None for n, p in model.backbone.named_parameters() if "lora_" not in n
    )
    scalar_head_grad = sum(p.grad.abs().sum().item() for p in model.mu_head.parameters() if p.grad is not None)
    scalar_embed_grad = sum(p.grad.abs().sum().item() for p in model.mu_embedding.parameters() if p.grad is not None)
    assert lora_grad_norm > 0, "LoRA params should receive gradient"
    assert not base_grad_present, "frozen base backbone params should receive no gradient"
    assert scalar_head_grad > 0, "scalar prediction heads should receive gradient"
    assert scalar_embed_grad > 0, "scalar embeddings should receive gradient from the shape-code loss path"
    print("gradient check OK: LoRA + all new embeddings/heads trainable, base backbone frozen")

    # ----- autoregressive generation + scalar prediction ----- #
    y_ids_pred, gen_scalars = model.generate_codes(
        x_ids, TCout, x_mu_f, x_sigma_f, x_min_f, x_max_f, x_r_map_f, x_r_hr_f, x_r_pulsat_f
    )
    assert y_ids_pred.shape == (B * N, TCout)
    assert (y_ids_pred >= 0).all() and (y_ids_pred < num_embeddings).all()
    for name in ("mu", "sigma", "min", "max"):
        assert gen_scalars[name].shape == (B * N, num_bins)
    print("generation check OK: codes", tuple(y_ids_pred.shape), "scalar logits", {k: tuple(v.shape) for k, v in gen_scalars.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="also test Qwen/Qwen2.5-0.5B from the Hub")
    args = parser.parse_args()

    print("=== tiny random Qwen2 backbone ===")
    run(make_tiny_hf_model(), num_embeddings=256, num_bins=32)
    print("ALL CHECKS PASSED (tiny model)")

    if args.real:
        print("=== real Qwen/Qwen2.5-0.5B backbone ===")
        real_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        run(real_model, num_embeddings=256, num_bins=32)
        print("ALL CHECKS PASSED (real model)")

    sys.exit(0)
