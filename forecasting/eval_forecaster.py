"""
Standalone evaluation of a trained TOTEM-original (non-LLM) forecaster checkpoint
(XcodeYtimeDecoder + MuStdModel), no training. Mirrors eval_llm_forecaster.py but
for the decode.py-based forecaster from train_forecaster.py.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from lib.models.metrics import dtw, pearsoncor
from train_forecaster import create_time_series_dataloader, get_params, inference


def evaluate(args):
    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    params = get_params(args.data_type, args.data_path)

    codebook = np.load(f"{params['dataroot']}/codebook.npy", allow_pickle=True)
    codebook = torch.from_numpy(codebook).to(device=device, dtype=torch.float32)
    vocab_size, _ = codebook.shape
    assert vocab_size == args.codebook_size

    dataloaders = create_time_series_dataloader(datapath=params["dataroot"], batchsize=params["batchsize"])
    dataloader = dataloaders[args.split]

    model_decode = torch.load(f"{args.checkpoint_path}/decode_checkpoint.pth", map_location=device, weights_only=False)
    model_mustd = torch.load(f"{args.checkpoint_path}/mustd_checkpoint.pth", map_location=device, weights_only=False)
    model_decode.to(device)
    model_mustd.to(device)
    model_decode.eval()
    model_mustd.eval()

    running_mse, running_mae, running_cor, running_dtw = 0.0, 0.0, 0.0, 0.0
    total_num, total_num_c = 0.0, 0.0
    all_predictions = []

    with torch.no_grad():
        for data in dataloader:
            pred_time = inference(
                data, model_decode, model_mustd, codebook, args.compression, device,
                onehot=args.onehot, scheme=args.scheme,
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
    parser = argparse.ArgumentParser(description="Evaluate a trained TOTEM-original forecaster checkpoint")
    parser.add_argument("--cuda-id", default=0, type=int)
    parser.add_argument("--data-type", default="ETTh1", type=str)
    parser.add_argument("--codebook_size", default=256, type=int)
    parser.add_argument("--compression", default=4, type=int)
    parser.add_argument("--data_path", default="", type=str, help="path to the Tin*_Tout* data dir")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], type=str)
    parser.add_argument("--checkpoint_path", required=True, type=str, help="dir containing decode_checkpoint.pth / mustd_checkpoint.pth")
    parser.add_argument("--onehot", action="store_true")
    parser.add_argument("--scheme", default=1, type=int)
    parser.add_argument("--save_predictions", default="", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = default_argument_parser()
    evaluate(args)
