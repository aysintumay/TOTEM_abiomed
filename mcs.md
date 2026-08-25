# GORMPO — Impella Weaning Project

## Project overview
Model-based offline RL (GORMPO/CORMPO) for Impella pump-level weaning decisions.
Replacing the Transformer-based digital twin in `abiomed_env/model.py` with an LLM world model
that uses in-context learning for multivariate clinical time series forecasting.

## Codebase structure
```
abiomed_env/model.py          # Transformer digital twin — interface to match
LLM_world/                    # New LLM world model — implement here
cormpo/common/buffer.py       # Replay buffer — RL only, not used by world model
```

## Interface contract
`LLM_world/` must expose the same interface as `abiomed_env/model.py`:
```python
predict(context_window, s_t, a_t) -> (s_t1_mean, s_t1_std)
# context_window : list of K prior (s, a, s') transitions — passed in directly by the caller
# s_t            : current state dict (variable names below)
# a_t            : P-level action over the forecast horizon — one integer (2–10) per predicted
#                  timestep (e.g. 6 values for a 6-step horizon), NOT a single scalar averaged
#                  or held constant across the window. A single int is also accepted and is
#                  then held constant across the horizon (rollout convenience), but the
#                  reference Transformer digital twin (abiomed_env/model.py) always consumes
#                  the full per-step trajectory, unpooled — that's the convention to match.
# returns        : mean and std over next-state variables
```
Output must cover all 12 state variables. Uncertainty via token log-probabilities or ensemble sampling.

---

## State variables (10-min intervals, 6 obs/hr)

| Variable | Description | Target / Normal |
|----------|-------------|-----------------|
| PumpPressure | Impella differential pressure (mmHg) | ≥ 60 |
| PumpSpeed | Motor speed (RPM) | — (set by P-level) |
| PumpFlow | Flow output (L/min) | 0.5–3.5 |
| LVP | LV pressure (mmHg) | reduced by support |
| LVEDP | LV end-diastolic pressure (mmHg) | < 18 |
| SYSTOLIC | Systolic arterial pressure (mmHg) | 90–130 |
| DIASTOLIC | Diastolic arterial pressure (mmHg) | 60–90 |
| PULSAT | Cardiac pulsatility (mmHg) — native CO proxy | ≥ 20 |
| PumpCurrent | Motor current (A) — rises with load | — |
| Heart Rate | bpm | 50–100 |
| ESE_lv | LV end-systolic elastance (mmHg/mL) — contractility | higher = better |
| Pump Level | P-level (integer 2–10) | action variable |

## Tokenization (GORMPO)
Each variable's 6-timestep patch history is summarized as a discrete token, not a raw number —
produced by the GORMPO tokenizer (`forecasting/lib/models/tokenizer.py`, a cross-channel-attention
VQ-VAE; checkpoint at `forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000`). Each token
has 10 parts, always in this fixed order, one row per variable:

```
c0 c1 c2 mu sigma min max r_map r_hr r_pulsat
```

- `c0, c1, c2` — three integers in [0, 255], indices into a learned codebook of 256 morphological
  shape patterns (Eq 19). **Categorical, not ordered by magnitude**: code 200 is not "bigger" than
  code 50, it's a different learned shape. The only way to infer what a code implies is by comparing
  it against worked examples.
- `mu, sigma, min, max` — each an integer bin in [0, 31]: the patch's mean/std/min/max value
  quantized into 32 quantile bins (Eq 20-23), ordered low(0)→high(31) within that variable's own
  observed training range. Unlike the shape codes, these ARE ordinal.
- `r_map, r_hr, r_pulsat` — three integers in {1, ..., 8} (Eq 24-25; never 0 -- that value only
  ever meant "not applicable" in the old per-channel encoding, which no longer applies now that
  all three are always computed per patch), one per clinical reward signal
  (based on PumpPressure/MAP, Heart Rate, and PULSAT respectively). These are the SAME three values
  on every row regardless of which variable that row describes — they summarize the whole patch's
  clinical stability, not just the named channel's own row. 8 = no penalty (stable), 1 = maximally
  unstable (clamped). Context only, never predicted.

Feature order (row 0-10 in every context/prediction block; Pump Level excluded — it's the action):
PumpPressure, PumpSpeed, PumpFlow, LVP, LVEDP, SYSTOLIC, DIASTOLIC, PULSAT, PumpCurrent, Heart Rate,
ESE_lv.

Task: given `forecast_horizon` timesteps of context (tokenized as 11 rows in the format above) plus
the chosen Pump Level for each of the next `forecast_horizon` steps (a per-step action sequence,
not one value held constant — see Interface contract above), predict the next patch's token for
each of the 11 variables as `c0 c1 c2 mu sigma min max` (7 integers, no reward).

## Action space
P-level ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10}. Higher P = more support, higher RPM/flow, lower LV preload.
Max change per timestep: ±1 (unless emergency).

---

## Impella physiology

- Pump aspirates blood from LV → ascending aorta; unloads LV throughout cardiac cycle
- P↑ → PumpFlow↑, PumpSpeed↑, LVEDP↓, LVP↓, SYSTOLIC↑
- P↑ → PULSAT↓ (less native ejection; pump does more work)
- Flow is preload-dependent: low LV filling → suction events → PumpFlow drops, arrhythmia risk
- Frank-Starling: LVEDP reduction can improve native contractility if LV was volume-overloaded
- RV interdependence: LV unloading raises RV preload; watch for DIASTOLIC↓ + CVP surrogate signs

## Weaning logic
**Reduce P-level when:** PULSAT ≥ 20, SYSTOLIC ≥ 90, LVEDP < 18, hemodynamics stable
**Hold when:** transitional state or uncertain trajectory
**Escalate when:** SYSTOLIC < 90, LVEDP > 18, PULSAT < 15, PumpFlow dropping unexpectedly

---

## Evaluation targets
- Next-state MAE / CRPS per variable (vs Transformer baseline)
- OOD detection: token log-probabilities as distributional shift signal