import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F

from extract_forecasting_data import codes2time
from lib.models.llm_forecaster import (
    LLMCodeForecaster,
    build_training_sequence,
    flatten_channels,
    unflatten_channels,
)
from lib.models.metrics import dtw, pearsoncor
from lib.models.revin import RevIN
from lib.utils.checkpoint import EarlyStopping
from lib.utils.env import seed_all_rng
from train_forecaster import create_time_series_dataloader, get_params


def train_one_epoch(
    dataloader,
    model,
    codebook,
    compression,
    optimizer,
    scheduler,
    epoch,
    device,
):
    running_loss, last_loss = 0.0, 0.0
    running_acc, last_acc = 0.0, 0.0
    log_every = max(len(dataloader) // 3, 3)

    for i, data in enumerate(dataloader):
        # ----- LOAD DATA ------ #
        _, _, codeids_x, codeids_y_labels = data
        # codeids_x: (B, TCin, Sin)
        # codeids_y_labels: (B, TCout, Sout)
        codeids_x = codeids_x.to(device)
        codeids_y_labels = codeids_y_labels.to(device)

        x_ids = flatten_channels(codeids_x)  # (B * Sin, TCin)
        y_ids = flatten_channels(codeids_y_labels)  # (B * Sout, TCout)

        inputs, labels = build_training_sequence(x_ids, y_ids)
        logits = model(inputs)  # (N, L, num_embeddings)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            mask = labels != -100
            preds = logits.argmax(dim=-1)
            acc = (preds[mask] == labels[mask]).float().mean().item()

        running_loss += loss.item()
        running_acc += acc
        if i % log_every == log_every - 1:
            last_loss = running_loss / log_every
            last_acc = running_acc / log_every
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"| epoch {epoch:3d} | {i+1:5d}/{len(dataloader):5d} batches | "
                f"lr {lr:02.5f} | loss {last_loss:5.4f} | code_acc {last_acc:5.4f}"
            )
            running_loss = 0.0
            running_acc = 0.0


def inference(
    data,
    model,
    codebook,
    compression_factor,
    vqvae_decoder,
    revin_layer,
    device,
):
    """
    Returns:
        out: (B, Tout, Sout)
    """
    x, y, codeids_x, codeids_y_labels = data
    x = x.to(device)
    codeids_x = codeids_x.to(device)

    B, TCin, Sin = codeids_x.shape
    B, TCout, Sout = codeids_y_labels.shape

    # populate revin stats from the input window (same "scheme 2" approximation
    # already used by train_forecaster.py: true output-window stats are unobserved
    # at inference time, so denorm the prediction using the input window's stats)
    _ = revin_layer(x, "norm")

    x_ids = flatten_channels(codeids_x)  # (B * Sin, TCin)
    y_ids_pred = model.generate_codes(x_ids, TCout)  # (B * Sin, TCout)
    y_ids_pred = unflatten_channels(y_ids_pred, B, Sin)  # (B, Sin, TCout)

    _, predictions_original_space = codes2time(
        y_ids_pred, codebook, compression_factor, vqvae_decoder, revin_layer
    )
    return predictions_original_space


def train(args):
    if not os.path.exists(args.file_save_path):
        os.makedirs(args.file_save_path)

    save_name = (
        str(args.data_type)
        + "_Tin"
        + str(args.Tin)
        + "_Tout"
        + str(args.Tout)
        + "_seed"
        + str(args.seed)
        + ".txt"
    )
    save_file = open(args.file_save_path + save_name, "w+")

    device = torch.device("cuda:%d" % (args.cuda_id) if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # -------- SET SEED ------- #
    seed_all_rng(None if args.seed < 0 else args.seed)

    # -------- PARAMS ------- #
    params = get_params(args.data_type, args.data_path)
    dataroot = params["dataroot"]
    Sin, Sout = params["Sin"], params["Sout"]
    datapath = dataroot

    # -------- CHECKPOINT ------- #
    if args.checkpoint:
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
    early_stopping = EarlyStopping(patience=args.patience, path=args.checkpoint_path)

    # -------- FROZEN VQVAE (needed to decode predicted codes back to time series) --- #
    vqvae_model = torch.load(args.trained_vqvae_model_path, map_location=device, weights_only=False)
    vqvae_model.to(device)
    vqvae_model.eval()
    for p in vqvae_model.parameters():
        p.requires_grad_(False)

    # -------- CODEBOOK ------- #
    codebook = np.load(os.path.join(datapath, "codebook.npy"), allow_pickle=True)
    codebook = torch.from_numpy(codebook).to(device=device, dtype=torch.float32)
    vocab_size, _ = codebook.shape
    assert vocab_size == args.codebook_size

    # ------ DATA LOADERS ------- #
    dataloaders = create_time_series_dataloader(datapath=datapath, batchsize=params["batchsize"])
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]
    test_dataloader = dataloaders["test"]

    # ------- MODEL: LLM CODE FORECASTER -------- #
    model = LLMCodeForecaster(
        llm_name=args.llm_name,
        num_embeddings=args.codebook_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules.split(","),
    )
    model.to(device)

    revin_layer = RevIN(num_features=Sin, affine=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        "Total # trainable parameters: ",
        sum(p.numel() for p in trainable_params),
        " / total params: ",
        sum(p.numel() for p in model.parameters()),
    )

    # ------- OPTIMIZER -------- #
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.baselr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.baselr)
    else:
        raise ValueError("Unknown optimizer type %s" % (args.optimizer))

    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args.steps * len(train_dataloader), gamma=0.1
        )
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=args.baselr,
            steps_per_epoch=len(train_dataloader),
            epochs=args.epochs,
            pct_start=0.2,
        )
    else:
        raise ValueError("Unknown scheduler type %s" % (args.scheduler))

    # ------- TRAIN & EVAL -------- #
    for epoch in range(args.epochs):
        model.train()
        train_one_epoch(
            train_dataloader,
            model,
            codebook,
            args.compression,
            optimizer,
            scheduler,
            epoch,
            device,
        )

        if val_dataloader is not None:
            model.eval()
            running_mse, running_mae, running_cor, running_dtw = 0.0, 0.0, 0.0, 0.0
            total_num, total_num_c = 0.0, 0.0
            with torch.no_grad():
                for vdata in val_dataloader:
                    pred_time = inference(
                        vdata, model, codebook, args.compression, vqvae_model.decoder,
                        revin_layer, device,
                    )
                    labels_time = vdata[1].to(device)

                    running_mse += F.mse_loss(pred_time, labels_time, reduction="sum")
                    running_mae += (pred_time - labels_time).abs().sum()
                    running_cor += pearsoncor(pred_time, labels_time, reduction="sum")
                    running_dtw += dtw(pred_time, labels_time, reduction="sum")
                    total_num += labels_time.numel()
                    total_num_c += labels_time.shape[0] * labels_time.shape[2]
            running_mae = running_mae / total_num
            running_mse = running_mse / total_num
            running_cor = running_cor / total_num_c
            running_dtw = running_dtw / total_num_c
            print(
                f"| [Val] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f} dtw {running_dtw:5.4f}"
            )
            save_file.write(
                f"| [Val] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f} dtw {running_dtw:5.4f}\n"
            )

            early_stopping_counter = early_stopping(
                running_mse, running_mae, {"llm_forecaster": model}
            )

        if test_dataloader is not None:
            model.eval()
            running_mse, running_mae, running_cor, running_dtw = 0.0, 0.0, 0.0, 0.0
            total_num, total_num_c = 0.0, 0.0
            with torch.no_grad():
                for tdata in test_dataloader:
                    pred_time = inference(
                        tdata, model, codebook, args.compression, vqvae_model.decoder,
                        revin_layer, device,
                    )
                    labels_time = tdata[1].to(device)

                    running_mse += F.mse_loss(pred_time, labels_time, reduction="sum")
                    running_mae += (pred_time - labels_time).abs().sum()
                    running_cor += pearsoncor(pred_time, labels_time, reduction="sum")
                    running_dtw += dtw(pred_time, labels_time, reduction="sum")
                    total_num += labels_time.numel()
                    total_num_c += labels_time.shape[0] * labels_time.shape[2]
            running_mae = running_mae / total_num
            running_mse = running_mse / total_num
            running_cor = running_cor / total_num_c
            running_dtw = running_dtw / total_num_c
            print(
                f"| [Test] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f} dtw {running_dtw:5.4f}"
            )
            save_file.write(
                f"| [Test] mse {running_mse:5.4f} mae {running_mae:5.4f} corr {running_cor:5.4f} dtw {running_dtw:5.4f}\n"
            )
            save_file.write(f"Early stopping counter is: {early_stopping_counter}\n")

        if early_stopping.early_stop:
            print("Early stopping....")
            save_file.write("Early stopping....")
            save_file.write("Take the test values right before the last early stopping counter = 0")
            save_file.close()
            return

    save_file.close()


def default_argument_parser():
    parser = argparse.ArgumentParser(description="LLM Code Forecaster")
    parser.add_argument("--cuda-id", default=0, type=int)
    parser.add_argument("--seed", default=-1, type=int)

    # ---------- DATA ---------- #
    parser.add_argument("--data-type", default="ETTh1", type=str)
    parser.add_argument("--codebook_size", default=256, type=int)
    parser.add_argument("--compression", default=4, type=int)
    parser.add_argument("--Tin", default=96, type=int)
    parser.add_argument("--Tout", default=96, type=int)
    parser.add_argument("--data_path", default="", type=str, help="path to data")

    # ----------- CHECKPOINT ------------ #
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--checkpoint_path", default="/data/", type=str)
    parser.add_argument("--patience", default=3, type=int)

    # ---------- TRAINING ---------- #
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--scheduler", default="onecycle", type=str)
    parser.add_argument("--baselr", default=0.0001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--steps", default=4, type=int, help="decay LR in epochs")

    # ----------- LLM / LORA ------------ #
    parser.add_argument("--llm_name", default="Qwen/Qwen2.5-0.5B", type=str)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        type=str,
        help="comma-separated list of submodule names to attach LoRA adapters to",
    )
    parser.add_argument(
        "--trained_vqvae_model_path",
        required=True,
        type=str,
        help="path to the frozen trained vqvae model, used to decode predicted codes back to time",
    )

    parser.add_argument("--file_save_path", default="", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = default_argument_parser()
    train(args)
