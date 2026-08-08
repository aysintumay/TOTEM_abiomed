"""
Standalone test-set inference for a trained PatchTST checkpoint (no training).
Mirrors eval_tokenizer.py/eval_gormpo_llm_forecaster.py's role for this baseline.

Run with: python eval_patchtst.py --data_type abiomed --checkpoint_path <checkpoint>.pth
"""
import argparse

import torch

from train_patchtst import build_dataloaders, run_eval


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    loaders, n_vars, _ = build_dataloaders(args.data_type, args.seq_len, args.pred_len, args.num_workers)
    print(f"test batches: {len(loaders['test'])}  n_vars={n_vars}")

    run_eval(loaders["test"], model, device, "Test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="abiomed")
    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--pred_len", type=int, default=6)
    parser.add_argument("--checkpoint_path", type=str, default="saved_models/patchtst_abiomed_tuned/checkpoints/patchtst_checkpoint.pth")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=3)
    args = parser.parse_args()
    main(args)
