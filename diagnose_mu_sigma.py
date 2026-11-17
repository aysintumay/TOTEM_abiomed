"""
python diagnose_mu_sigma.py

For the Med42-8B LoRA checkpoint, finds test episodes where SYSTOLIC/PULSAT blew up
and prints the true vs. dequantized mu/sigma (bounded, per-channel bin lookups) next
to x_hat_norm (the VQ-decoder output that combines with them) -- confirms whether the
explosion comes from a bad mu/sigma bin or from x_hat_norm's magnitude.
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "forecasting")
from train_gormpo_llm_forecaster import create_gormpo_dataloader, _unpack  # noqa: E402
from lib.models.llm_forecaster import flatten_channels  # noqa: E402
from lib.models.gormpo_llm_forecaster import flatten_channels_scalar, unflatten_channels_scalar  # noqa: E402

TOKEN_DATA_ROOT = "forecasting/data/gormpo_tokens_mcs_scratch_long2000"
TOKENIZER_PATH = "forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth"
CKPT = "forecasting/saved_models/gormpo_llm_med42_mcs/checkpoints/gormpo_llm_forecaster_checkpoint.pth"
DEVICE = "cuda:0"
SYSTOLIC, DIASTOLIC, PULSAT = 5, 6, 7

device = torch.device(DEVICE)
tokenizer = torch.load(TOKENIZER_PATH, map_location=device, weights_only=False)
tokenizer.to(device); tokenizer.eval()
model = torch.load(CKPT, map_location=device, weights_only=False)
model.to(device); model.eval()

dataloaders = create_gormpo_dataloader(TOKEN_DATA_ROOT, batchsize=128, num_workers=0)
test_dataloader = dataloaders["test"]

n_shown = 0
with torch.no_grad():
    for data in test_dataloader:
        (x_orig, y_orig, x_codes, y_codes, x_mu, x_sigma, x_min, x_max,
         x_r_map, x_r_hr, x_r_pulsat, _, _, _, _) = _unpack(data, device)
        B, TCin, N = x_codes.shape
        TCout = y_codes.shape[1]

        x_ids = flatten_channels(x_codes)
        x_mu_f, x_sigma_f, x_min_f, x_max_f, x_r_map_f, x_r_hr_f, x_r_pulsat_f = (
            flatten_channels_scalar(t) for t in (x_mu, x_sigma, x_min, x_max, x_r_map, x_r_hr, x_r_pulsat)
        )
        y_ids_pred, scalars_pred = model.generate_codes(
            x_ids, TCout, x_mu_f, x_sigma_f, x_min_f, x_max_f, x_r_map_f, x_r_hr_f, x_r_pulsat_f
        )

        mu_idx = unflatten_channels_scalar(scalars_pred["mu"].argmax(dim=-1), B, N)
        sigma_idx = unflatten_channels_scalar(scalars_pred["sigma"].argmax(dim=-1), B, N)
        mu_hat = tokenizer.q_mu.dequantize(mu_idx)      # (B, N) bounded, per-channel
        sigma_hat = tokenizer.q_sigma.dequantize(sigma_idx)

        num_code_words, code_dim = tokenizer.vq._embedding.weight.shape
        one_hot = F.one_hot(y_ids_pred, num_code_words).to(tokenizer.vq._embedding.weight.dtype)
        quantized = torch.matmul(one_hot, tokenizer.vq._embedding.weight).transpose(1, 2)  # (B*N, code_dim, TCout)

        x_hat, x_hat_norm = tokenizer.decode(quantized, mu_hat, sigma_hat)  # (B, N, patch_len) each
        pred_time = x_hat.transpose(1, 2)   # (B, patch_len, N)
        true_time = y_orig                  # (B, patch_len, N)

        # true mu/sigma for the SAME target patch, for comparison (not what the model saw --
        # the model only sees CONTEXT mu/sigma; this is just "what the ground-truth target
        # patch's own mu/sigma would dequantize/compare to").
        true_patch = true_time.transpose(1, 2)  # (B, N, patch_len)
        _, true_mu, true_sigma = tokenizer.patch_and_scale(true_patch)

        for ch, cname in [(SYSTOLIC, "SYSTOLIC"), (PULSAT, "PULSAT")]:
            err = (pred_time[:, :, ch] - true_time[:, :, ch]).abs().max(dim=1).values  # (B,) worst timestep
            bad = (err > 100).nonzero(as_tuple=True)[0]
            for b in bad[:3]:
                b = b.item()
                print(f"[{cname}] episode in-batch-idx={b}  max_err={err[b]:.1f}")
                print(f"  true_mu={true_mu[b, ch]:.2f}  true_sigma={true_sigma[b, ch]:.2f}")
                print(f"  mu_hat ={mu_hat[b, ch]:.2f}  sigma_hat={sigma_hat[b, ch]:.2f}"
                      f"   (both bounded by training quantile range)")
                print(f"  x_hat_norm[{cname}] over patch: {x_hat_norm[b, ch].tolist()}")
                print(f"  pred_time[{cname}]: {pred_time[b, :, ch].tolist()}")
                print(f"  true_time[{cname}]: {true_time[b, :, ch].tolist()}")
                print()
                n_shown += 1
        if n_shown >= 6:
            break
print("done")
