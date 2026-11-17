# MCS forecasting context (for LLM forecasting prompts)

Forecasting-only companion to mcs.md. Every statement below was checked against the
training split of /public/gormpo/10min_1hr_all_data.pkl (12,051 one-hour windows).
Deliberately omitted from mcs.md: clinical targets / normal ranges (they describe goals,
not typical values -- 51% of timesteps have SYSTOLIC < 90, 31% have LVEDP >= 18 -- and
pull forecasts toward them), weaning logic (it chooses the action; the action is given),
and qualitative rare-event physiology (Frank-Starling, RV interdependence, suction).
Each "## " section is selectable on its own via --context_sections.

## Persistence
Over one hour these values usually stay close to their most recent level. The Pump Level
changes in only about 8% of one-hour windows; when it stays the same, the typical change
over the next hour is small (see Typical change). Large changes mostly follow a change in
Pump Level. Do not extrapolate short-term wiggles into steady trends.

## Typical change
Median absolute change between consecutive 10-minute timesteps when the Pump Level is
unchanged (90th percentile in brackets), in the same units as the history:
PumpPressure 1.2 (4.9) mmHg; PumpSpeed ~0 (it only moves when the Pump Level changes);
PumpFlow 0.8 (3.6); LVP 1.1 (4.3) mmHg; LVEDP 0.9 (3.5) mmHg; SYSTOLIC 1.6 (6.7) mmHg;
DIASTOLIC 1.0 (3.8) mmHg; PULSAT 0.8 (3.4) mmHg; PumpCurrent 0.9 (3.9);
Heart Rate 0.8 (4.6) bpm; ESE_lv 0.02 (0.07) mmHg/mL.

## Pump Level effects
These apply only when the planned Pump Level differs from the current one; if it is
unchanged, ignore this section. Average change over the next hour per +1 Pump Level
(reverse the sign for a decrease):
- Strong, near-deterministic: PumpSpeed about +200 RPM, PumpFlow about +21,
  PumpCurrent about +72.
- Moderate: DIASTOLIC about +1.9 mmHg, PumpPressure about +1.5 mmHg,
  PULSAT about -1.1 mmHg.
- Negligible (no reliable effect within an hour): LVP, LVEDP, SYSTOLIC, Heart Rate, ESE_lv.

## Units
PumpFlow and PumpCurrent are in device units (PumpFlow is roughly L/min x 100, typical
values 200-400; PumpCurrent typically 400-900), not L/min or A. Pump Level is an integer
P2-P10 (higher = more support).
