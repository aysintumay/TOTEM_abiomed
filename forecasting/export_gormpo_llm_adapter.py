"""
Export the LoRA adapter from a trained GormpoTokenLLMForecaster checkpoint as a
standalone PEFT directory, so it can be loaded into GormpoFewShotWorldModel's
text-generation backbone (gormpo_world_model.py) for fine-tuned few-shot inference.

The saved checkpoint (gormpo_llm_forecaster_checkpoint.pth) is the entire
GormpoTokenLLMForecaster object pickled via torch.save, including
model.backbone -- a PeftModel wrapping hf_model.model (note: .model, not the
full AutoModelForCausalLM, since this forecaster discards the LM head in favor
of its own shape/scalar heads). Only that inner backbone's adapter is exported
here, matching the exact scope GormpoFewShotWorldModel.llm.model expects it to
be applied to.

Run with: python export_gormpo_llm_adapter.py \
    --checkpoint_path <gormpo_llm_forecaster_checkpoint.pth> \
    --adapter_out_dir <output dir>
"""
import argparse

import torch


def main(args):
    model = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    model.backbone.save_pretrained(args.adapter_out_dir)
    print(f"Saved LoRA adapter to {args.adapter_out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--adapter_out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
