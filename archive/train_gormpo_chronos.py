"""
Fine-tune a pretrained Chronos T5 checkpoint against GormpoChronosTokenizer
(gormpo_chronos_tokenizer.py) on real MCS data: the pretrained backbone's own
value-vocabulary embedding/head are discarded (resize_token_embeddings to the
combined GORMPO offset-vocab), what's kept and fine-tuned is the pretrained T5
sequence-modeling backbone -- same value proposition as LoRA-tuning medllama/Qwen on
plain VQ code ids elsewhere in this project, just without LoRA (the vocab/embedding
churn here is large enough that full fine-tuning of a small T5 is cheaper and more
appropriate than trying to preserve most of the backbone frozen).

Data: forecasting/train_tokenizer.py's load_mcs_episode_halves gives episode-paired
(x_half, y_half) patches, (num_episodes, N=12, k=patch_len) raw physical units, N=12
channels in FEATURE_NAMES order (PumpPressure...ESE_lv, Pump Level last) -- this is
the exact format/channel-count the real GORMPO tokenizer checkpoints were trained and
calibrated on (see gormpo_chronos_tokenizer.py's channel-order note), so it's reused
here unmodified rather than re-deriving an equivalent slice from the different
13-raw-column pickle format the world-model scripts use, which risks a channel-order
mismatch against the tokenizer's per-channel fitted bin edges.

Unlike gormpo_world_model.py's medllama few-shot prompting (which excludes Pump Level
from the tokenized rows and injects it as free-form action text instead), this script
follows forecasting/train_gormpo_llm_forecaster.py's convention: all 12 channels,
including Pump Level, are tokenized and predicted uniformly -- Chronos has no
free-text side-channel to inject an action into separately, and this project's own
from-scratch GORMPO-LLM forecaster already treats "predict everything, including
Pump Level" as world modeling's job rather than special-casing the action channel.

Run with: python train_gormpo_chronos.py --tokenizer_path <gormpo checkpoint> --save_path <dir>
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSeq2SeqLM

sys.path.insert(0, "forecasting")
from train_tokenizer import load_mcs_episode_halves  # noqa: E402
from lib.utils.env import seed_all_rng  # noqa: E402

from chronos import ChronosConfig  # noqa: E402
from gormpo_chronos_tokenizer import GormpoChronosTokenizer  # noqa: E402


def build_tokenizer(gormpo_tokenizer_path, device):
    gormpo_tokenizer = torch.load(gormpo_tokenizer_path, map_location=device, weights_only=False)
    gormpo_tokenizer.eval()
    for p in gormpo_tokenizer.parameters():
        p.requires_grad_(False)

    chronos_config = ChronosConfig(
        tokenizer_class="GormpoChronosTokenizer",
        tokenizer_kwargs={},
        context_length=0,      # filled in below
        prediction_length=0,
        n_tokens=0,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=False,
        model_type="seq2seq",
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    tokenizer = GormpoChronosTokenizer(gormpo_tokenizer, chronos_config)
    chronos_config.n_tokens = tokenizer.vocab_size
    chronos_config.context_length = tokenizer.context_len
    chronos_config.prediction_length = tokenizer.label_len
    return tokenizer, chronos_config


def run_epoch(model, tokenizer, loader, device, optimizer=None, grad_clip=1.0):
    """optimizer given -> train; None -> eval (no_grad). Returns (mean_loss, n_batches)."""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, n_batches = 0.0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for x_half, y_half in loader:
            x_half, y_half = x_half.to(device), y_half.to(device)

            context_ids, context_mask, state = tokenizer.context_input_transform(x_half)
            label_ids, label_mask = tokenizer.label_input_transform(y_half, state)
            labels = label_ids.clone()
            labels[~label_mask] = -100

            out = model(input_ids=context_ids, attention_mask=context_mask.long(), labels=labels)
            loss = out.loss

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches, n_batches


def main(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    seed_all_rng(args.seed)

    os.makedirs(os.path.join(args.save_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "configs"), exist_ok=True)

    tokenizer, chronos_config = build_tokenizer(args.tokenizer_path, device)
    print(
        f"L={tokenizer.L}  label_len={tokenizer.label_len}  context_len={tokenizer.context_len}  "
        f"num_channels={tokenizer.gormpo_tokenizer.num_channels}  combined vocab={tokenizer.vocab_size}"
    )

    x_train, y_train = load_mcs_episode_halves(args.data_root, "train", args.patch_len)
    x_val, y_val = load_mcs_episode_halves(args.data_root, "val", args.patch_len)
    print(f"train episodes: {x_train.shape[0]}  val episodes: {x_val.shape[0]}")

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
    )

    if args.resume_from_checkpoint:
        # Weights only -- optimizer state (Adam momentum/variance) isn't persisted by
        # save_pretrained, so this is a warm-started fresh optimizer, not a byte-exact
        # continuation. Vocab is already the combined GORMPO vocab in this checkpoint,
        # so no resize_token_embeddings/pad/eos setup needed (already baked in).
        print(f"Resuming weights from: {args.resume_from_checkpoint}")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.resume_from_checkpoint)
    else:
        print(f"Loading pretrained Chronos backbone: {args.model_id}")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
        model.resize_token_embeddings(tokenizer.vocab_size)
        model.config.pad_token_id = model.generation_config.pad_token_id = chronos_config.pad_token_id
        model.config.eos_token_id = model.generation_config.eos_token_id = chronos_config.eos_token_id
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Backbone params (full fine-tune, no LoRA): {total_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_time = time.time()
    for i in range(args.num_epochs):
        epoch = args.epoch_offset + i
        last = i == args.num_epochs - 1
        train_loss, _ = run_epoch(model, tokenizer, train_loader, device, optimizer, args.grad_clip)
        print(f"| epoch {epoch:3d} | train loss {train_loss:.4f}")

        if epoch % args.val_every == 0 or last:
            val_loss, _ = run_epoch(model, tokenizer, val_loader, device, optimizer=None)
            print(f"| [Val] epoch {epoch:3d} | loss {val_loss:.4f}")

        if epoch % args.save_every == 0 or last:
            model.save_pretrained(os.path.join(args.save_path, "checkpoints", f"epoch_{epoch}"))

    model.save_pretrained(os.path.join(args.save_path, "checkpoints", "final"))

    with open(os.path.join(args.save_path, "configs", "gormpo_chronos_config.json"), "w") as f:
        json.dump(
            {
                "tokenizer_path": args.tokenizer_path,
                "model_id": args.model_id,
                "n_special_tokens": chronos_config.n_special_tokens,
                "pad_token_id": chronos_config.pad_token_id,
                "eos_token_id": chronos_config.eos_token_id,
                "vocab_size": tokenizer.vocab_size,
                "context_len": tokenizer.context_len,
                "label_len": tokenizer.label_len,
                "L": tokenizer.L,
                "num_channels": tokenizer.gormpo_tokenizer.num_channels,
            },
            f,
            indent=2,
        )
    with open(os.path.join(args.save_path, "configs", "config_file.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Total training time: {time.time() - start_time:.1f}s")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_path", type=str,
        default="forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000/checkpoints/final_tokenizer.pth",
    )
    parser.add_argument("--data_root", type=str, default="forecasting/data_raw/abiomed",
                        help="folder with {train,val,test}_data.npy, (episodes, 2*patch_len, num_channels)")
    parser.add_argument("--model_id", type=str, default="amazon/chronos-t5-small")
    parser.add_argument("--save_path", type=str, default="saved_models/gormpo_chronos")
    parser.add_argument("--patch_len", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=16,
                         help="episodes per batch; actual model batch is batch_size * num_channels rows")
    parser.add_argument("--num_epochs", type=int, default=30, help="epochs THIS run does, not a cumulative total")
    parser.add_argument("--resume_from_checkpoint", type=str, default="",
                         help="dir with a save_pretrained() checkpoint to warm-start weights from "
                              "(optimizer state is not preserved -- fresh AdamW moments)")
    parser.add_argument("--epoch_offset", type=int, default=0,
                         help="starting epoch number for logging/checkpoint naming, e.g. 30 when "
                              "continuing after a prior 30-epoch run")
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    main(args)
