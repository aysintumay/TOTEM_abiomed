"""
Standalone evaluation of a trained LLMCodeForecaster checkpoint (no training).

Loads the frozen VQVAE + codebook + saved LLM forecaster checkpoint, runs
autoregressive code generation on a data split, decodes back to time series,
and reports mse/mae/corr/dtw. Mirrors the inference() path used inside
train_llm_forecaster.py's val/test loop.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from extract_forecasting_data import codes2time
from lib.models.llm_forecaster import flatten_channels, unflatten_channels
from lib.models.metrics import dtw, pearsoncor
from lib.models.revin import RevIN
from train_forecaster import create_time_series_dataloader, get_params


def inference(data, model, codebook, compression_factor, vqvae_decoder, revin_layer, device):
    x, y, codeids_x, codeids_y_labels = data
    x = x.to(device)
    codeids_x = codeids_x.to(device)

    B, TCin, Sin = codeids_x.shape
    B, TCout, Sout = codeids_y_labels.shape

    _ = revin_layer(x, "norm")

    x_ids = flatten_channels(codeids_x)  # (B * Sin, TCin)
    y_ids_pred = model.generate_codes(x_ids, TCout)  # (B * Sin, TCout)
    y_ids_pred = unflatten_channels(y_ids_pred, B, Sin)  # (B, Sin, TCout)

    _, predictions_original_space = codes2time(
        y_ids_pred, codebook, compression_factor, vqvae_decoder, revin_layer
    )
    return predictions_original_space


def evaluate(args):
    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    params = get_params(args.data_type, args.data_path)
    Sin = params["Sin"]

    # -------- frozen VQVAE (decodes predicted codes back to time series) -------- #
    vqvae_model = torch.load(args.trained_vqvae_model_path, map_location=device, weights_only=False)
    vqvae_model.to(device)
    vqvae_model.eval()
    for p in vqvae_model.parameters():
        p.requires_grad_(False)

    # -------- codebook -------- #
    codebook = np.load(f"{params['dataroot']}/codebook.npy", allow_pickle=True)
    codebook = torch.from_numpy(codebook).to(device=device, dtype=torch.float32)
    vocab_size, _ = codebook.shape
    assert vocab_size == args.codebook_size

    # -------- data -------- #
    dataloaders = create_time_series_dataloader(datapath=params["dataroot"], batchsize=params["batchsize"])
    dataloader = dataloaders[args.split]

    # -------- trained LLM forecaster checkpoint -------- #
    model = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    revin_layer = RevIN(num_features=Sin, affine=False)

    running_mse, running_mae, running_cor, running_dtw = 0.0, 0.0, 0.0, 0.0
    total_num, total_num_c = 0.0, 0.0
    all_predictions = []

    with torch.no_grad():
        for data in dataloader:
            pred_time = inference(
                data, model, codebook, args.compression, vqvae_model.decoder, revin_layer, device
            )
            labels_time = data[1].to(device)

            running_mse += F.mse_loss(pred_time, labels_time, reduction="sum")
            running_mae += (pred_time - labels_time).abs().sum()
            running_cor += pearsoncor(pred_time, labels_time, reduction="sum")
            running_dtw += dtw(pred_time, labels_time, reduction="sum")
            total_num += labels_time.numel()
            total_num_c += labels_time.shape[0] * labels_time.shape[2]
            all_predictions.append(pred_time.cpu().numpy())

    running_mse = running_mse / total_num
    running_mae = running_mae / total_num
    running_cor = running_cor / total_num_c
    running_dtw = running_dtw / total_num_c
    print(
        f"| [{args.split.capitalize()}] mse {running_mse:5.4f} mae {running_mae:5.4f} "
        f"corr {running_cor:5.4f} dtw {running_dtw:5.4f}"
    )

    if args.save_predictions:
        np.save(args.save_predictions, np.concatenate(all_predictions, axis=0))
        print(f"Saved predictions to {args.save_predictions}")

    return running_mse.item(), running_mae.item(), running_cor.item(), running_dtw.item()


def default_argument_parser():
    parser = argparse.ArgumentParser(description="Evaluate a trained LLM Code Forecaster checkpoint")
    parser.add_argument("--cuda-id", default=0, type=int)

    parser.add_argument("--data-type", default="ETTh1", type=str)
    parser.add_argument("--codebook_size", default=256, type=int)
    parser.add_argument("--compression", default=4, type=int)
    parser.add_argument("--data_path", default="", type=str, help="path to the Tin*_Tout* data dir")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], type=str)

    parser.add_argument("--trained_vqvae_model_path", required=True, type=str)
    parser.add_argument("--checkpoint_path", required=True, type=str, help="path to llm_forecaster_checkpoint.pth")

    parser.add_argument("--save_predictions", default="", type=str, help="optional .npy path to save predictions")

    return parser.parse_args()


if __name__ == "__main__":
    args = default_argument_parser()
    evaluate(args)
