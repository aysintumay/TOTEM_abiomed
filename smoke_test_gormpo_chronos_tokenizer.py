"""
Offline smoke test for gormpo_chronos_tokenizer.py, mirroring
forecasting/smoke_test_gormpo_llm_forecaster.py's approach: a tiny randomly-initialized
GPT2 backbone and a tiny from-scratch GormpoTokenizer fit on synthetic data at the real
MCS shapes (patch_len=6, compression_factor=2 -> L=3, num_channels=11), no network
access or real trained checkpoints required.

Checks:
  1. Round trip: context_input_transform -> output_transform on the SAME ids reproduces
     exactly what a direct GormpoTokenizer.encode()+decode() call gives (verifies the
     offset/flatten/clamp/reshape bookkeeping introduces no bugs of its own).
  2. resize_token_embeddings to the tokenizer's combined vocab + a full ChronosModel
     forward/generate call runs end to end without shape errors.

Run with: python smoke_test_gormpo_chronos_tokenizer.py
"""
import sys

import torch
from transformers import AutoModelForCausalLM, GPT2Config

sys.path.insert(0, "forecasting")
from lib.models.tokenizer import GormpoTokenizer  # noqa: E402

from chronos import ChronosConfig, ChronosModel  # noqa: E402
from gormpo_chronos_tokenizer import GormpoChronosTokenizer  # noqa: E402


def make_tiny_gormpo_tokenizer(patch_len=6, num_channels=11, compression_factor=2, num_bins=32):
    vqvae_config = dict(
        block_hidden_size=8, num_residual_layers=1, res_hidden_size=8,
        embedding_dim=8, num_embeddings=16, commitment_cost=0.25,
        compression_factor=compression_factor,
    )
    model = GormpoTokenizer.from_scratch(
        vqvae_config, patch_len=patch_len, num_channels=num_channels, num_bins=num_bins, use_reward=True,
    )
    model.eval()

    # fit the fixed scalar quantizers on synthetic data, same as train_tokenizer.py does
    n_fit = 256
    x_fit = torch.randn(n_fit, num_channels, patch_len) * 5 + 10
    with torch.no_grad():
        _, mu, sigma = model.patch_and_scale(x_fit)
        x_tilde = model.cross_channel_attn((x_fit - mu.unsqueeze(-1)) / sigma.unsqueeze(-1))
        patch_min, patch_max = x_tilde.min(dim=-1).values, x_tilde.max(dim=-1).values
    model.q_mu.fit(mu)
    model.q_sigma.fit(sigma)
    model.q_min.fit(patch_min)
    model.q_max.fit(patch_max)
    model.calibrate_codebook(x_fit[:32])

    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_tiny_gpt2(vocab_size):
    cfg = GPT2Config(
        vocab_size=vocab_size, n_embd=32, n_layer=2, n_head=2, n_positions=128,
    )
    return AutoModelForCausalLM.from_config(cfg)


def main():
    B, num_channels, patch_len = 4, 11, 6
    gormpo_tokenizer = make_tiny_gormpo_tokenizer(patch_len=patch_len, num_channels=num_channels)

    chronos_config = ChronosConfig(
        tokenizer_class="GormpoChronosTokenizer",
        tokenizer_kwargs={},
        context_length=999,   # overwritten below once the real token-position length is known
        prediction_length=999,
        n_tokens=0,            # overwritten below
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=False,
        model_type="causal",
        num_samples=5,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    tokenizer = GormpoChronosTokenizer(gormpo_tokenizer, chronos_config)
    chronos_config.n_tokens = tokenizer.vocab_size
    chronos_config.context_length = tokenizer.context_len
    chronos_config.prediction_length = tokenizer.label_len
    print(
        f"L={tokenizer.L}  label_len={tokenizer.label_len}  context_len={tokenizer.context_len}  "
        f"combined vocab={tokenizer.vocab_size}"
    )

    # ----- 1. round-trip check ----- #
    x_context = torch.randn(B, num_channels, patch_len) * 5 + 10
    x_label = torch.randn(B, num_channels, patch_len) * 5 + 10

    ctx_ids, ctx_mask, state = tokenizer.context_input_transform(x_context)
    assert ctx_ids.shape == (B * num_channels, tokenizer.context_len)
    assert ctx_mask.all()

    label_ids, label_mask = tokenizer.label_input_transform(x_label, state)
    assert label_ids.shape == (B * num_channels, tokenizer.label_len)

    # decode the SAME label ids we just produced, treating each row as its own 1-sample batch
    recon = tokenizer.output_transform(label_ids.unsqueeze(1), state)  # (B*N, 1, patch_len)
    assert recon.shape == (B * num_channels, 1, patch_len)

    # cross-check against a direct encode()+decode() call using the tokenizer's own
    # dequantized mu/sigma -- this is exactly what output_transform computes internally,
    # so it must match exactly (float equality, no LM involved in either path).
    direct_tokens = gormpo_tokenizer.encode(x_label)
    direct_mu = gormpo_tokenizer.q_mu.dequantize(direct_tokens["q_mu"])
    direct_sigma = gormpo_tokenizer.q_sigma.dequantize(direct_tokens["q_sigma"])
    direct_x_hat, _ = gormpo_tokenizer.decode(direct_tokens["quantized"], direct_mu, direct_sigma)
    direct_x_hat_flat = direct_x_hat.reshape(B * num_channels, 1, patch_len)
    assert torch.allclose(recon, direct_x_hat_flat, atol=1e-5), "output_transform must exactly match direct encode+decode"
    print("round-trip check OK: context/label flatten + output_transform matches direct GormpoTokenizer encode/decode")

    # ----- 2. out-of-range id clamping doesn't crash ----- #
    garbage_ids = torch.randint(0, tokenizer.vocab_size, (B * num_channels, 3, tokenizer.label_len))
    garbage_recon = tokenizer.output_transform(garbage_ids, state)
    assert garbage_recon.shape == (B * num_channels, 3, patch_len)
    assert torch.isfinite(garbage_recon).all()
    print("clamping check OK: arbitrary in-vocab ids decode to finite values without touching invalid codebook/bin indices")

    # ----- 3. full ChronosModel forward + generate on a tiny GPT2 backbone ----- #
    hf_model = make_tiny_gpt2(tokenizer.vocab_size)
    hf_model.config.pad_token_id = chronos_config.pad_token_id
    hf_model.generation_config.pad_token_id = chronos_config.pad_token_id
    chronos_model = ChronosModel(config=chronos_config, model=hf_model)

    preds = chronos_model(
        input_ids=ctx_ids,
        attention_mask=ctx_mask,
        prediction_length=tokenizer.label_len,
        num_samples=3,
    )  # (B*N, num_samples, label_len)
    assert preds.shape == (B * num_channels, 3, tokenizer.label_len)
    assert (preds >= 0).all() and (preds < tokenizer.vocab_size).all()
    print("generation check OK:", tuple(preds.shape))

    decoded = tokenizer.output_transform(preds, state)
    assert decoded.shape == (B * num_channels, 3, patch_len)
    assert torch.isfinite(decoded).all()
    print("end-to-end check OK: context -> ChronosModel.generate -> output_transform ->", tuple(decoded.shape), "physical values")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
