"""
Standalone evaluation of a trained GormpoTokenLLMForecaster checkpoint (no
training). Loads the trained tokenizer + LLM forecaster checkpoint, generates
shape codes and predicts scale statistics autoregressively/once from context,
decodes back to raw physiological units through GormpoTokenizer.decode(), and
reports mse/mae/corr/dtw. Mirrors eval_llm_forecaster.py's role for the existing
TOTEM+LLM baseline.

Run with: python eval_gormpo_llm_forecaster.py --tokenizer_path <checkpoint> \
    --llm_path <checkpoint> --data_root <extracted tokens dir>
"""
import argparse

import torch

from train_gormpo_llm_forecaster import create_gormpo_dataloader, run_eval


def main(args):
    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tokenizer = torch.load(args.tokenizer_path, map_location=device, weights_only=False)
    tokenizer.to(device)
    tokenizer.eval()

    model = torch.load(args.llm_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    dataloaders = create_gormpo_dataloader(args.data_root, args.batchsize, args.num_workers)

    class _NullFile:
        def write(self, _):
            pass

    for split, tag in (("val", "Val"), ("test", "Test")):
        run_eval(dataloaders[split], model, tokenizer, device, tag, _NullFile())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_id", default=0, type=int)
    parser.add_argument("--tokenizer_path", type=str, default="saved_models/gormpo_tokenizer_mcs/checkpoints/final_tokenizer.pth")
    parser.add_argument("--llm_path", type=str, default="saved_models/gormpo_llm_mcs/checkpoints/gormpo_llm_forecaster_checkpoint.pth")
    parser.add_argument("--data_root", type=str, default="data/gormpo_tokens_mcs")
    parser.add_argument("--batchsize", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    args = parser.parse_args()
    main(args)
