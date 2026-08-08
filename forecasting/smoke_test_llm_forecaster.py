"""
Offline smoke test for lib/models/llm_forecaster.py.

Validates the full mechanics (embedding/head swap, LoRA wrapping, masked causal-LM
loss, autoregressive generation, channel flatten/unflatten, decode-back-to-time
shape contract) using a tiny randomly-initialized model of a real architecture
(Qwen2) and random data at the exact shapes the real pipeline produces for ETTh1
(Tin=Tout=96, compression=4, Sin=Sout=7). No network access or real trained
VQVAE/data required.

Run with: --real to additionally smoke-test against the real pretrained
Qwen/Qwen2.5-0.5B (requires network access; slower).
"""
import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, Qwen2Config

from lib.models.llm_forecaster import (
    LLMCodeForecaster,
    build_training_sequence,
    flatten_channels,
    unflatten_channels,
)
from lib.models.vqvae import Decoder
from lib.models.revin import RevIN
from extract_forecasting_data import codes2time


def make_tiny_hf_model():
    cfg = Qwen2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        vocab_size=8,  # irrelevant: this embedding/head is discarded
    )
    return AutoModelForCausalLM.from_config(cfg)


def run(hf_model, num_embeddings, embedding_dim, compression_factor):
    B, Sin, Sout = 4, 7, 7
    Tin, Tout = 96, 96
    TCin, TCout = Tin // compression_factor, Tout // compression_factor

    model = LLMCodeForecaster(num_embeddings=num_embeddings, hf_model=hf_model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable} / total params: {total}")
    assert trainable < total, "LoRA should freeze most of the backbone"

    # ----- forward + masked CE loss + backward ----- #
    codeids_x = torch.randint(0, num_embeddings, (B, TCin, Sin))
    codeids_y = torch.randint(0, num_embeddings, (B, TCout, Sout))

    x_ids = flatten_channels(codeids_x)  # (B*Sin, TCin)
    y_ids = flatten_channels(codeids_y)  # (B*Sout, TCout)
    assert x_ids.shape == (B * Sin, TCin)
    assert unflatten_channels(x_ids, B, Sin).equal(codeids_x.permute(0, 2, 1).reshape(B, Sin, TCin))

    inputs, labels = build_training_sequence(x_ids, y_ids)
    logits = model(inputs)
    assert logits.shape == (B * Sin, TCin + TCout - 1, num_embeddings)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
    )
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
    assert lora_grad_norm > 0, "LoRA params should receive gradient"
    assert not base_grad_present, "frozen base backbone params should receive no gradient"
    print("gradient check OK: LoRA + embedding/head trainable, base backbone frozen")

    # ----- autoregressive generation ----- #
    y_ids_pred = model.generate_codes(x_ids, TCout)
    assert y_ids_pred.shape == (B * Sin, TCout)
    assert torch.isfinite(y_ids_pred).all()
    assert (y_ids_pred >= 0).all() and (y_ids_pred < num_embeddings).all()
    print("generation check OK: shape", tuple(y_ids_pred.shape))

    y_ids_pred = unflatten_channels(y_ids_pred, B, Sin)  # (B, Sin, TCout)

    # ----- decode predicted codes back to continuous time series ----- #
    decoder = Decoder(
        in_channels=embedding_dim,
        num_hiddens=32,
        num_residual_layers=1,
        num_residual_hiddens=16,
        compression_factor=compression_factor,
    )
    codebook = torch.randn(num_embeddings, embedding_dim)
    revin_layer = RevIN(num_features=Sin, affine=False)
    x = torch.randn(B, Tin, Sin)
    _ = revin_layer(x, "norm")  # populate mean/stdev stats

    _, predictions_original_space = codes2time(
        y_ids_pred, codebook, compression_factor, decoder, revin_layer
    )
    assert predictions_original_space.shape == (B, Tout, Sout)
    assert torch.isfinite(predictions_original_space).all()
    print("decode-back check OK: shape", tuple(predictions_original_space.shape))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="also test Qwen/Qwen2.5-0.5B from the Hub")
    args = parser.parse_args()

    print("=== tiny random Qwen2 backbone ===")
    run(make_tiny_hf_model(), num_embeddings=256, embedding_dim=32, compression_factor=4)
    print("ALL CHECKS PASSED (tiny model)")

    if args.real:
        print("=== real Qwen/Qwen2.5-0.5B backbone ===")
        real_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        run(real_model, num_embeddings=256, embedding_dim=real_model.config.hidden_size, compression_factor=4)
        print("ALL CHECKS PASSED (real model)")

    sys.exit(0)
