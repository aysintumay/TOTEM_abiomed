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
# a_t            : P-level integer (2–10)
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