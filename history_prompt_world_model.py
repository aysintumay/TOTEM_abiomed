"""
Direct LLM forecasting world model for Impella weaning (MCS data), zero-shot, using the
(timestamp, value) prompt format of TimesX (arXiv 2607.06973, CiK-style), adapted to a
multivariate state. One prompt per patient window:

    Metadata: ... 10-minute ICU hemodynamics ... Prediction target period: t7 to t12.
    History: In the format of (timestamp, value), the patient's last 6 timesteps are:
      PumpPressure - <mcs.md description, target>: (t1, 83.1), (t2, 84.9), ..., (t6, 87.2)
      ...                                   <- all 12 state features incl. Pump Level
    Covariates: Pump Level in the prediction window: (t7, P8), ..., (t12, P8)
    Context: mcs.md action space / Impella physiology / weaning logic
    Task: Predict ... one line per variable:
      PumpPressure: [x7, x8, x9, x10, x11, x12]

Values are raw physical units throughout (no normalization in the prompt or response).
LLM response -> numerical forecast (parsed per variable line).

The action is the full per-step P-level trajectory of the next hour (mcs.md interface
contract), not a single averaged value.

Default: a single greedy completion per prompt (temperature 0, num_samples 1) used
directly as the forecast -- no ensemble; std is then 0. num_samples > 1 with
temperature > 0 gives sampled mean/std.

Deliberately independent of DigitTokenWorldModel / FewShotDigitTokenWorldModel;
only reuses LLMWorldModel for model loading, data loading and normalization.
"""

import pickle
import re

import numpy as np
import torch

from world_model import FEATURE_NAMES, LLMWorldModel

# Per-variable description + target/normal range, from mcs.md "State variables".
# PumpFlow / PumpCurrent: this pkl stores them in device units (mean ~296 / ~644), not
# the L/min / A mcs.md lists, so the ranges are restated on the data's scale
# (flow ~= L/min x 100) rather than copied verbatim.
VAR_DESC = {
    "PumpPressure": "Impella differential pressure (mmHg), target >= 60",
    "PumpSpeed":    "Impella motor speed (RPM), set by the Pump Level",
    "PumpFlow":     "Impella flow output (device units, ~L/min x 100; normal ~50-350)",
    "LVP":          "LV pressure (mmHg), reduced by support",
    "LVEDP":        "LV end-diastolic pressure (mmHg), target < 18",
    "SYSTOLIC":     "systolic arterial pressure (mmHg), normal 90-130",
    "DIASTOLIC":    "diastolic arterial pressure (mmHg), normal 60-90",
    "PULSAT":       "cardiac pulsatility (mmHg), native cardiac-output proxy, target >= 20",
    "PumpCurrent":  "Impella motor current (device units), rises with load",
    "Heart Rate":   "heart rate (bpm), normal 50-100",
    "ESE_lv":       "LV end-systolic elastance (mmHg/mL), contractility, higher = better",
    "Pump Level":   "Impella P-level (integer 2-10), the action",
}

# Plain units, used when mcs_context=False (no mcs.md descriptions/targets).
UNITS = {
    "PumpPressure": "mmHg", "PumpSpeed": "RPM", "PumpFlow": "device units", "LVP": "mmHg",
    "LVEDP": "mmHg", "SYSTOLIC": "mmHg", "DIASTOLIC": "mmHg", "PULSAT": "mmHg",
    "PumpCurrent": "device units", "Heart Rate": "bpm", "ESE_lv": "mmHg/mL", "Pump Level": "P2-P10",
}

# mcs.md "Action space", "Impella physiology", "Weaning logic", condensed.
MCS_CONTEXT = (
    "Action space: P-level in {2,...,10}. Higher P = more support, higher RPM/flow, lower LV "
    "preload. Max change per timestep is +/-1 (unless emergency).\n"
    "Impella physiology: the pump aspirates blood from the LV into the ascending aorta and "
    "unloads the LV throughout the cardiac cycle. P up -> PumpFlow up, PumpSpeed up, LVEDP "
    "down, LVP down, SYSTOLIC up. P up -> PULSAT down (less native ejection; the pump does more "
    "work). Flow is preload-dependent: low LV filling -> suction events -> PumpFlow drops, "
    "arrhythmia risk. Frank-Starling: LVEDP reduction can improve native contractility if the "
    "LV was volume-overloaded. RV interdependence: LV unloading raises RV preload; watch for "
    "DIASTOLIC down.\n"
    "Weaning logic: P-level is reduced when PULSAT >= 20, SYSTOLIC >= 90, LVEDP < 18 and "
    "hemodynamics are stable; held when transitional or uncertain; escalated when SYSTOLIC < 90, "
    "LVEDP > 18, PULSAT < 15, or PumpFlow drops unexpectedly."
)
DECIMALS = {"PumpSpeed": 0, "PumpCurrent": 0, "Heart Rate": 0, "ESE_lv": 3, "Pump Level": 0}  # default 1

_NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


class HistoryPromptWorldModel(LLMWorldModel):
    def __init__(self, model_id: str = "m42-health/Llama3-Med42-8B", num_samples: int = 1,
                 temperature: float = 0.0, mcs_context: bool = True,
                 context_file: str | None = None, context_sections: list | None = None,
                 pl_prefix: bool = True, **kwargs):
        # pl_prefix=False writes the planned Pump Level as plain numbers "(t7, 7)" -- the same
        # format as the Pump Level history line -- instead of "(t7, P7)".
        self.pl_prefix = pl_prefix
        self.mcs_context = mcs_context  # False: drop mcs.md descriptions/targets and Context section
        # context_file (e.g. mcs_forecast_context.md) replaces the mcs.md Context section and
        # the mcs.md per-variable targets; history lines then use plain units.
        self.context_text = self._load_context(context_file, context_sections) if context_file else None
        super().__init__(model_id=model_id, num_samples=num_samples, temperature=temperature, **kwargs)
        self.tokenizer.padding_side = "left"  # batched decoder-only generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.n_targets = self.num_features - 1  # Pump Level is the action, not forecast

    @staticmethod
    def _load_context(path: str, sections: list | None) -> str:
        with open(path) as f:
            parts = re.split(r"(?m)^## ", f.read())[1:]  # drop the file preamble
        found = {p.split("\n", 1)[0].strip(): p.split("\n", 1)[1].strip() for p in parts}
        keep = sections or list(found)
        missing = [k for k in keep if k not in found]
        assert not missing, f"sections {missing} not in {path} (have {list(found)})"
        return "\n".join(f"{k}: {' '.join(found[k].split())}" for k in keep)

    def load_data(self, path: str):
        super().load_data(path)
        with open(path, "rb") as f:
            train = np.asarray(pickle.load(f)["train"])[:, :, self.columns].reshape(-1, self.num_features)
        # clip bounds for parsed outputs: guards metrics against a few absurd LLM numbers
        self.clip_low = np.nanpercentile(train, 0.5, axis=0).astype(np.float32)
        self.clip_high = np.nanpercentile(train, 99.5, axis=0).astype(np.float32)

    # ── prompt ────────────────────────────────────────────────────────────────

    def _fmt(self, name: str, v: float) -> str:
        return f"{v:.{DECIMALS.get(name, 1)}f}"

    def _system_prompt(self) -> str:
        return (
            "You are a time-series forecasting assistant. You are given metadata, a patient's "
            "multivariate history, and known future covariates. Answer ONLY in the requested "
            "format, no explanation."
        )

    def _build_prompt(self, ctx_phys: np.ndarray, pl_seq: list) -> str:
        T, H = ctx_phys.shape[0], len(pl_seq)
        hist = "\n".join(
            (f"  {name} - {VAR_DESC[name]}: " if self.mcs_context and self.context_text is None
             else f"  {name} ({UNITS[name]}): ")
            + ", ".join(f"(t{t + 1}, {self._fmt(name, ctx_phys[t, j])})" for t in range(T))
            for j, name in enumerate(FEATURE_NAMES)
        )
        pre = "P" if self.pl_prefix else ""
        plan = ", ".join(f"(t{T + h + 1}, {pre}{p})" for h, p in enumerate(pl_seq))
        template = "\n".join(
            f"{name}: [" + ", ".join(f"x{T + h + 1}" for h in range(H)) + "]"
            for name in FEATURE_NAMES[: self.n_targets]
        )
        return (
            f"Metadata: This multivariate time series records the hemodynamic state of an ICU "
            f"patient on Impella mechanical circulatory support, with a collection frequency of "
            f"10 minutes. Prediction target period: from t{T + 1} to t{T + H}.\n"
            f"History: In the format of (timestamp, value), the patient's last {T} timesteps are:\n"
            f"{hist}\n"
            f"Covariates: Pump Level in the prediction window: {plan}.\n"
            + (f"Context:\n{self.context_text}\n" if self.context_text is not None
               else f"Context:\n{MCS_CONTEXT}\n" if self.mcs_context else "")
            + (
            f"Task: Predict the values of the {self.n_targets} variables below for the next {H} "
            f"timestamps (t{T + 1}-t{T + H}). Respond with exactly one line per variable, in this "
            f"format, replacing each x with a number:\n{template}")
        )

    # ── generation / parsing ──────────────────────────────────────────────────

    def _generate(self, prompts: list, n: int) -> list:
        """prompts: list of user strings -> list (len(prompts)) of lists of n completions."""
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": self._system_prompt()}, {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.llm.device)
        sample = self.temperature > 0
        with torch.no_grad():
            out = self.llm.generate(
                **enc,
                max_new_tokens=12 * self.forecast_horizon * self.n_targets,
                do_sample=sample,
                temperature=self.temperature if sample else None,
                top_p=1.0 if sample else None, top_k=None,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        dec = self.tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [dec[i * n:(i + 1) * n] for i in range(len(prompts))]

    def _parse(self, text: str) -> np.ndarray:
        """-> (H, n_targets) physical, NaN for variables whose line is missing/short."""
        out = np.full((self.forecast_horizon, self.n_targets), np.nan, np.float32)
        for j, name in enumerate(FEATURE_NAMES[: self.n_targets]):
            # brackets optional: some models (e.g. Med42) answer "Name: 1.0, 2.0, ..."
            m = re.search(rf"(?m)^\W*{re.escape(name)}\W*?:[ \t]*\[?([^\n\]]*)", text)
            if m:
                nums = re.findall(_NUM, m.group(1))
                if len(nums) >= self.forecast_horizon:
                    out[:, j] = [float(x) for x in nums[: self.forecast_horizon]]
        return out

    # ── public API ────────────────────────────────────────────────────────────

    def _pl_seq(self, pl) -> list:
        if isinstance(pl, (int, np.integer)):
            return [int(pl)] * self.forecast_horizon
        seq = [int(p) for p in pl]
        assert len(seq) == self.forecast_horizon, f"need {self.forecast_horizon} P-levels, got {len(seq)}"
        return seq

    def predict_batch(self, contexts_norm: list, pls: list, prompt_batch_size: int = 8):
        """
        contexts_norm : list of (T, 12) normalized tensors
        pls           : list of per-step P-level int sequences (len forecast_horizon) or ints
        Returns list of (mean_norm, std_norm, n_failed_vars); tensors are (forecast_horizon, 12).
        A variable with no parseable sample falls back to last-value carry-forward.
        """
        m = self.mean[self.columns].detach().cpu().numpy()
        s = self.std[self.columns].detach().cpu().numpy()
        ctxs = [self._unnorm_context(c) for c in contexts_norm]
        seqs = [self._pl_seq(p) for p in pls]
        prompts = [self._build_prompt(c, q) for c, q in zip(ctxs, seqs)]
        completions = []
        for i in range(0, len(prompts), prompt_batch_size):
            completions += self._generate(prompts[i:i + prompt_batch_size], self.num_samples)

        results = []
        for ctx, seq, comps in zip(ctxs, seqs, completions):
            arr = np.stack([self._parse(c) for c in comps])  # (n, H, 11)
            arr = np.clip(arr, self.clip_low[: self.n_targets], self.clip_high[: self.n_targets])
            ok = ~np.isnan(arr).any(axis=1)  # (n, 11) sample usable for this variable
            failed = int((~ok.any(axis=0)).sum())
            mean = np.zeros((self.forecast_horizon, self.num_features), np.float32)
            std = np.zeros_like(mean)
            for j in range(self.n_targets):
                if ok[:, j].any():
                    mean[:, j] = arr[ok[:, j], :, j].mean(0)
                    std[:, j] = arr[ok[:, j], :, j].std(0)
                else:
                    mean[:, j] = ctx[-1, j]
            mean[:, -1] = seq
            results.append((
                torch.tensor((mean - m) / s, dtype=torch.float32),
                torch.tensor(std / s, dtype=torch.float32),
                failed,
            ))
        return results

    def predict(self, context_norm: torch.Tensor, pl):
        mean_norm, std_norm, _ = self.predict_batch([context_norm], [pl])[0]
        return mean_norm, std_norm

    def _pl_from_norm(self, pl):
        if isinstance(pl, int):
            return pl
        pl = torch.as_tensor(pl, dtype=torch.float32).flatten()
        phys = self.unnorm_pl(pl).round().int().tolist()
        return phys if len(phys) == self.forecast_horizon else int(phys[0])

    def step(self, x: torch.Tensor, pl) -> torch.Tensor:
        mean_norm, _ = self.predict(x.squeeze(0), self._pl_from_norm(pl))
        return mean_norm.unsqueeze(0)
