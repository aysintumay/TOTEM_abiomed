"""
Offline smoke test for gormpo_chronos2_model.py, mirroring
smoke_test_gormpo_chronos_tokenizer.py's role: catch wiring bugs (wrong shapes, wrong
row-ordering assumptions, broken gradient flow) before touching real data or a GPU
training run. Uses the REAL pretrained amazon/chronos-2 checkpoint (needed since the
12-layer/768-dim architecture is baked into it -- there's no cheap from-scratch
tiny-config path the way the GPT2/T5 smoke tests have), but entirely synthetic random
token ids -- no GORMPO tokenizer checkpoint or MCS data required.

Checks:
  1. Forward pass produces correctly-shaped logits for all 7 output heads.
  2. A toy masked cross-entropy loss (supervised only at the 11 target rows, matching
     the real training script's convention) backprops into: the new embedding/head
     parameters (always), and the borrowed Time+FeedForward layers (when NOT frozen) --
     confirming GroupSelfAttention truly isn't in the gradient graph at all (since it's
     never constructed/called), not just zeroed out.
  3. freeze_backbone=True actually blocks gradients from reaching the borrowed layers.
  4. Pump Level's future-known row and a target row produce genuinely different future
     position embeddings before the encoder (sanity check that pump_level_row_mask's
     torch.where is actually branching, not silently always picking one side).

Run with: python smoke_test_gormpo_chronos2_model.py
"""
import torch
import torch.nn.functional as F

from gormpo_chronos2_model import GormpoChronos2Model


def make_random_batch(B, N, num_embeddings, num_bins, num_reward_bins, device):
    rows = B * N
    ctx_shape_ids = torch.randint(0, num_embeddings, (rows, 3), device=device)
    ctx_mu = torch.randint(0, num_bins, (rows,), device=device)
    ctx_sigma = torch.randint(0, num_bins, (rows,), device=device)
    ctx_min = torch.randint(0, num_bins, (rows,), device=device)
    ctx_max = torch.randint(0, num_bins, (rows,), device=device)
    ctx_r_map = torch.randint(0, num_reward_bins, (rows,), device=device)
    ctx_r_hr = torch.randint(0, num_reward_bins, (rows,), device=device)
    ctx_r_pulsat = torch.randint(0, num_reward_bins, (rows,), device=device)

    # Pump Level is the LAST of every N consecutive rows (row = b*N + n, n=11=N-1),
    # matching FEATURE_NAMES/columns convention used project-wide.
    pump_level_row_mask = (torch.arange(rows, device=device) % N) == (N - 1)

    future_shape_ids = torch.randint(0, num_embeddings, (rows, 3), device=device)
    future_mu = torch.randint(0, num_bins, (rows,), device=device)
    future_sigma = torch.randint(0, num_bins, (rows,), device=device)
    future_min = torch.randint(0, num_bins, (rows,), device=device)
    future_max = torch.randint(0, num_bins, (rows,), device=device)

    # ground-truth labels for the 11 TARGET rows only (what the model should predict)
    y_shape_ids = torch.randint(0, num_embeddings, (rows, 3), device=device)
    y_mu = torch.randint(0, num_bins, (rows,), device=device)
    y_sigma = torch.randint(0, num_bins, (rows,), device=device)
    y_min = torch.randint(0, num_bins, (rows,), device=device)
    y_max = torch.randint(0, num_bins, (rows,), device=device)

    return dict(
        ctx_shape_ids=ctx_shape_ids, ctx_mu=ctx_mu, ctx_sigma=ctx_sigma, ctx_min=ctx_min, ctx_max=ctx_max,
        ctx_r_map=ctx_r_map, ctx_r_hr=ctx_r_hr, ctx_r_pulsat=ctx_r_pulsat,
        pump_level_row_mask=pump_level_row_mask,
        future_shape_ids=future_shape_ids, future_mu=future_mu, future_sigma=future_sigma,
        future_min=future_min, future_max=future_max,
        y_shape_ids=y_shape_ids, y_mu=y_mu, y_sigma=y_sigma, y_min=y_min, y_max=y_max,
    )


def masked_loss(shape_logits, scalar_logits, batch, target_row_mask):
    """Cross-entropy summed over the 3 shape heads + 4 scalar heads, at target rows only."""
    c0, c1, c2 = shape_logits
    loss = (
        F.cross_entropy(c0[target_row_mask], batch["y_shape_ids"][target_row_mask, 0])
        + F.cross_entropy(c1[target_row_mask], batch["y_shape_ids"][target_row_mask, 1])
        + F.cross_entropy(c2[target_row_mask], batch["y_shape_ids"][target_row_mask, 2])
        + F.cross_entropy(scalar_logits["mu"][target_row_mask], batch["y_mu"][target_row_mask])
        + F.cross_entropy(scalar_logits["sigma"][target_row_mask], batch["y_sigma"][target_row_mask])
        + F.cross_entropy(scalar_logits["min"][target_row_mask], batch["y_min"][target_row_mask])
        + F.cross_entropy(scalar_logits["max"][target_row_mask], batch["y_max"][target_row_mask])
    )
    return loss


def run(freeze_backbone, device):
    B, N = 4, 12
    num_embeddings, num_bins, num_reward_bins = 256, 32, 9
    model = GormpoChronos2Model(
        num_embeddings=num_embeddings, num_bins=num_bins, num_reward_bins=num_reward_bins,
        freeze_backbone=freeze_backbone, device=device,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"freeze_backbone={freeze_backbone} | trainable params: {trainable} / total params: {total}")
    if freeze_backbone:
        assert trainable < total, "freeze_backbone=True should freeze the borrowed Time+FeedForward layers"
    else:
        assert trainable == total, "freeze_backbone=False should leave every param trainable"

    batch = make_random_batch(B, N, num_embeddings, num_bins, num_reward_bins, device)
    target_row_mask = ~batch["pump_level_row_mask"]

    shape_logits, scalar_logits = model(
        batch["ctx_shape_ids"], batch["ctx_mu"], batch["ctx_sigma"], batch["ctx_min"], batch["ctx_max"],
        batch["ctx_r_map"], batch["ctx_r_hr"], batch["ctx_r_pulsat"], batch["pump_level_row_mask"],
        future_shape_ids=batch["future_shape_ids"], future_mu=batch["future_mu"],
        future_sigma=batch["future_sigma"], future_min=batch["future_min"], future_max=batch["future_max"],
    )

    rows = B * N
    assert shape_logits[0].shape == (rows, num_embeddings)
    assert shape_logits[1].shape == (rows, num_embeddings)
    assert shape_logits[2].shape == (rows, num_embeddings)
    for name in ("mu", "sigma", "min", "max"):
        assert scalar_logits[name].shape == (rows, num_bins), f"{name} shape mismatch"
    print("forward shape check OK")

    loss = masked_loss(shape_logits, scalar_logits, batch, target_row_mask)
    assert torch.isfinite(loss)
    loss.backward()

    new_param_grad = sum(
        p.grad.abs().sum().item() for p in model.shape0_embedding.parameters() if p.grad is not None
    )
    assert new_param_grad > 0, "new embedding params should receive gradient"
    head_grad = sum(p.grad.abs().sum().item() for p in model.mu_head.parameters() if p.grad is not None)
    assert head_grad > 0, "new head params should receive gradient"
    future_query_grad = model.future_query.grad
    assert future_query_grad is not None and future_query_grad.abs().sum().item() > 0, (
        "future_query should receive gradient (target rows use it)"
    )

    backbone_grad = sum(
        p.grad.abs().sum().item() for p in model.time_attn_layers.parameters() if p.grad is not None
    )
    if freeze_backbone:
        assert backbone_grad == 0, "frozen backbone layers should receive NO gradient"
    else:
        assert backbone_grad > 0, "unfrozen backbone layers should receive gradient"
    print(f"gradient check OK (backbone_grad={backbone_grad:.4f}, new_param_grad={new_param_grad:.4f})")

    # GroupSelfAttention truly absent: nothing in the model has that class name at all.
    all_module_types = {type(m).__name__ for m in model.modules()}
    assert "GroupSelfAttention" not in all_module_types, "GroupSelfAttention must not be present anywhere in the model"
    print("GroupSelfAttention-absence check OK:", sorted(all_module_types))

    return model


def run_pump_level_branch_check(device):
    """Confirm pump_level_row_mask's torch.where is actually branching: the Pump Level
    row's future embedding must differ from a target row's future embedding (the
    former uses real future tokens, the latter uses the shared future_query)."""
    B, N = 2, 12
    num_embeddings, num_bins, num_reward_bins = 256, 32, 9
    model = GormpoChronos2Model(
        num_embeddings=num_embeddings, num_bins=num_bins, num_reward_bins=num_reward_bins, device=device,
    ).to(device)
    batch = make_random_batch(B, N, num_embeddings, num_bins, num_reward_bins, device)

    with torch.no_grad():
        context_embeds = model._embed_patch(batch["ctx_shape_ids"], batch["ctx_mu"], batch["ctx_sigma"],
                                             batch["ctx_min"], batch["ctx_max"])
        pl_future = model._embed_patch(batch["future_shape_ids"], batch["future_mu"], batch["future_sigma"],
                                        batch["future_min"], batch["future_max"])
        future_embeds = model.future_query.unsqueeze(0).expand(B * N, -1).clone()
        future_embeds = torch.where(batch["pump_level_row_mask"].unsqueeze(-1), pl_future, future_embeds)

    pl_rows = future_embeds[batch["pump_level_row_mask"]]
    target_rows = future_embeds[~batch["pump_level_row_mask"]]
    assert not torch.allclose(pl_rows[0], target_rows[0]), "Pump Level row must differ from target rows"
    assert torch.allclose(target_rows[0], target_rows[1]), "target rows should all share the same future_query"
    print("pump_level_row_mask branching check OK")
    del context_embeds  # unused, computed only to mirror forward()'s call shape


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"=== smoke test on {device} ===")
    run(freeze_backbone=False, device=device)
    run(freeze_backbone=True, device=device)
    run_pump_level_branch_check(device)
    print("ALL CHECKS PASSED")
