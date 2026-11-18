# Archived scripts

Earlier experiments, kept for reference. They are not maintained and may rely on data or
checkpoints that have since moved (checkpoints now live under
/public/timeseries_tokenizer/TOTEM_abiomed/, reached through the saved_models symlinks).

- Aug 2026: Chronos v1 + GORMPO tokenizer swap, native Chronos fine-tune, Qwen/MedLlama
  digit-token and few-shot LLM world models, the original Transformer eval, and their
  per-model visualize_predictions_*.py scripts.
- Oct 2026: Chronos-2 with the GORMPO tokenizer (x_tilde / x_bar inputs, time-only encoder,
  joint training) and their train/eval/visualize scripts.
- Nov 2026: visualize_ensemble_predictions.py, replaced by the top-level
  visualize_predictions.py (same figures, models chosen with --models).

The scripts expect the repo root as the working directory and import each other (and
top-level modules such as world_model.py), so run them as:

    PYTHONPATH=.:archive python archive/<script>.py
