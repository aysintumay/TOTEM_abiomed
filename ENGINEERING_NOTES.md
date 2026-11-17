# Engineering notes (not fed to the LLM)

`mcs.md` is loaded in full into every ICL prompt (`gormpo_world_model.py::_load_mcs_md()`), so it
must stay lean -- implementation/investigation detail that isn't needed for the LLM to understand
its task belongs here instead. (A past mistake: adding two verbose sections directly to mcs.md
inflated the zero-shot prompt from a few hundred tokens to 3221, causing OOM crashes that had
nothing to do with the tokenizer itself -- moved here and trimmed back.)

## Data quality: 2 imputed sensor glitches

The raw training split (`/public/gormpo/10min_1hr_all_data.pkl`, key `"train"`) had exactly 2
single-timestep sensor glitches out of 144,612 timestep-readings (episode 2608, t=11; episode
10214, t=0) — SYSTOLIC spiking to 1327/2551 mmHg (true range across the whole dataset is ~46-167
mmHg). These 2 points alone stretched the tokenizer's `q_sigma`/`q_mu` quantile-bin edges for
SYSTOLIC/PULSAT out to physically meaningless values (bin center ~459 instead of ~25), which is why
the LLM occasionally decoded a wrong-but-plausible bin guess into a ±1000 mmHg forecast for those
two channels specifically — nowhere else. PULSAT inherits the same corruption mechanically since
it's computed as `SYSTOLIC - DIASTOLIC` upstream (confirmed correlation 0.99999... across the whole
dataset). val/test splits are unaffected (0 flagged timesteps in either).

Fixed via `impute_outliers_and_save.py`: both cells are replaced with the mean of their patch's
other 5 valid timesteps (SYSTOLIC), then PULSAT is re-derived as `corrected SYSTOLIC - DIASTOLIC`
rather than imputed independently, to preserve that exact identity. Saved to
`forecasting/data_raw/10min_1hr_all_data_imputed.pkl` (repo-local; the external
`/public/gormpo/...` source is left untouched). The tokenizer was fully retrained from scratch on
this data (`saved_models/gormpo_tokenizer_mcs_scratch_long2000_retrained`, matching the original
2000-epoch/from-scratch recipe), and tokens re-extracted
(`data/gormpo_tokens_mcs_scratch_long2000_retrained`). The ICL path's downstream clipping safety
net (`gormpo_world_model.py`'s old `_phys_low`/`_phys_high` clip) has been removed now that the
root cause is fixed at the data/tokenizer level instead of papered over at decode time.

## Two forecaster implementations, both action-conditioned the same way

Both the ICL world model (`gormpo_world_model.py`, prompting a stock/LoRA-merged LLM) and the
fine-tuned forecaster (`forecasting/train_gormpo_llm_forecaster.py` +
`forecasting/lib/models/gormpo_llm_forecaster.py`, LoRA-trained with custom shape/scalar heads)
are given the future Pump Level as a known input and only forecast the 11 physiological channels —
Pump Level is never itself a forecast target in either path. (An earlier version of the fine-tuned
forecaster treated Pump Level as a 12th channel to jointly predict, with no way to condition on a
specific chosen action — not usable as an MBRL world model, since you can't query "what happens if
P-level is set to X" from a model that just guesses its own P-level. Fixed by splitting Pump Level
out of the forecast set entirely.)

The two paths condition on the action differently, appropriate to their own architecture:
- **ICL**: the future P-level sequence is spelled out in the prompt text (`"P8 P8 P8 P8 P8 P7"`) --
  the LLM's own attention over the prompt tokens is what relates each step's action to the rest.
- **Fine-tuned forecaster**: `GormpoTokenLLMForecaster.action_embedding` + `embed()`. Each of the 6
  future P-level values gets its own embedding lookup (a small dedicated `nn.Embedding(9,
  hidden_size)` table, not routed through the tokenizer's 32-bin `q_mu`/`q_sigma` -- P-level is
  already a natural small integer, 2-10) AND its own distinct sequence position, prepended before
  the context shape-code positions rather than summed into a shared bias. This is
  Decision-Transformer-style: the backbone's own rotary positional encoding (already part of the
  pretrained Qwen/Med42 architecture, not anything built here) is what lets attention distinguish
  "action at t=0" from "action at t=5" -- not a hand-built mechanism. Verified end-to-end (not just
  at the embedding-lookup level) that a P-level ramp and its reverse produce different
  `boundary_hidden` representations through the full model, including the frozen backbone +
  LoRA layers.
  (Two earlier attempts here were rejected: summing a per-timestep embedding across positions is
  commutative and can't tell a ramp from its reverse; a single `nn.Linear` reading the ordered
  vector directly -- mirroring `abiomed_env/model.py`'s reference Transformer, which concatenates
  the raw `pl` vector onto its encoder output before the decoder MLP,
  `pp = torch.cat([encoded_src[:, -1, :], pl], 1)` -- IS technically order-aware, but only as a
  flat, position-independent bias; it never gives the backbone's attention anything to attend
  *between*, since the information never occupies its own sequence position.)
  mu/sigma/min/max/reward stay as an additive bias on the context positions, unchanged -- those are
  genuine per-patch aggregate stats with no timestep structure to preserve, unlike the action.
