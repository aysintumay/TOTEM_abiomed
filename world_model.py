"""
Zero-shot LLM world model for Impella pump weaning.
Drop-in replacement for abiomed_env.model.WorldModel.

Interface contract (matches WorldModel exactly):
  .step(x, pl)                -> (1, forecast_horizon, 12) normalized tensor
  .predict(context, pl_int)   -> (mean_norm, std_norm) each (forecast_horizon, 12)
  .unnorm_output(x)           -> unnormalized tensor
  .unnorm_state_vectors(x)    -> unnormalized numpy array
  .normalize_pl(pl)           -> normalized scalar tensor
  .unnorm_pl(pl)              -> unnormalized scalar tensor
  .load_data(path)            -> populates mean, std, data_train/val/test
  .num_features, .forecast_horizon, .columns
  .mean, .std                 -> 13-element tensors (raw data stats)

Patch tokenization
------------------
Each feature value is quantized to an integer bin index 0-999.
Integers 0-999 are all single tokens in the Llama 3 BPE vocabulary, so every
value costs exactly 1 token regardless of its physical magnitude (e.g. PumpSpeed
~4000 RPM maps to a bin in [0,999] just like LVEDP ~15 mmHg).
Bin boundaries are fitted per-feature from the 1st/99th percentiles of training data.
The compact prompt omits per-row feature names; the feature order is defined once
in the system prompt, reducing context from ~580 tokens to ~110 tokens.
"""

import json
import os
import pickle
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "abiomed_env"))
from model import TimeSeriesDataset

# ── Feature metadata ──────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "PumpPressure",  # 0  — differential pressure (mmHg); target ≥ 60
    "PumpSpeed",     # 1  — motor speed (RPM); set by P-level
    "PumpFlow",      # 2  — flow output (L/min); 0.5–3.5
    "LVP",           # 3  — LV pressure (mmHg)
    "LVEDP",         # 4  — LV end-diastolic pressure (mmHg); target < 18
    "SYSTOLIC",      # 5  — systolic arterial pressure (mmHg)
    "DIASTOLIC",     # 6  — diastolic arterial pressure (mmHg)
    "PULSAT",        # 7  — cardiac pulsatility (mmHg); target ≥ 20
    "PumpCurrent",   # 8  — motor current (A)
    "Heart Rate",    # 9  — heart rate (bpm); target 50–100
    "ESE_lv",        # 10 — LV end-systolic elastance (mmHg/mL)
    "Pump Level",    # 11 — Impella P-level (2–10), the action
]

NUM_BINS = 1000   # 0–999: all single tokens in Llama 3 BPE


# ── World model ───────────────────────────────────────────────────────────────

class LLMWorldModel:
    """
    Zero-shot LLM world model with patch tokenization.
    Each feature value → integer bin index 0-999 (1 token).
    Context and prediction are compact rows of 12 integers with no per-row labels.
    """

    columns: list = [i for i in range(0, 13) if i != 11]

    def __init__(
        self,
        model_id: str = "m42-health/Llama3-Med42-8B",
        num_features: int = 12,
        forecast_horizon: int = 6,
        num_samples: int = 10,
        temperature: float = 0.8,
        device: str = "auto",
        load_in_4bit: bool = False,
    ):
        self.num_features     = num_features
        self.forecast_horizon = forecast_horizon
        self.num_samples      = num_samples
        self.temperature      = temperature
        self.num_bins         = NUM_BINS

        # populated by load_data()
        self.mean     = None
        self.std      = None
        self.bin_low  = None   # (12,) float32 — 1st-percentile per feature (training)
        self.bin_high = None   # (12,) float32 — 99th-percentile per feature (training)
        self.data_train = self.data_val = self.data_test = None

        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        model_kwargs = {"dtype": torch.bfloat16, "device_map": device}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )

        print(f"Loading model: {model_id}")
        self.llm = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.llm.eval()
        print("Model ready.")

    # ── data loading ──────────────────────────────────────────────────────────

    def load_model(self, path: str):
        """No-op stub — LLM is loaded in __init__. Exists for interface compatibility."""
        pass

    def load_data(self, path: str):
        """Identical signature to WorldModel.load_data(). Fits per-feature bin boundaries."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        mean = data["mean"]
        std  = data["std"]
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32)
            std  = torch.tensor(std,  dtype=torch.float32)
        self.mean, self.std = mean, std

        norm = lambda split: ((data[split] - data["mean"]) / data["std"])[:, :, self.columns]
        self.data_train = TimeSeriesDataset(norm("train"), self.forecast_horizon, self.forecast_horizon)
        self.data_val   = TimeSeriesDataset(norm("val"),   self.forecast_horizon, self.forecast_horizon)
        self.data_test  = TimeSeriesDataset(norm("test"),  self.forecast_horizon, self.forecast_horizon)

        # fit bin boundaries from training data physical values
        train_raw = data["train"]
        if isinstance(train_raw, torch.Tensor):
            train_raw = train_raw.numpy()
        train_phys = train_raw[:, :, self.columns].reshape(-1, self.num_features).astype(np.float32)
        self.bin_low  = np.percentile(train_phys, 1,  axis=0).astype(np.float32)
        self.bin_high = np.percentile(train_phys, 99, axis=0).astype(np.float32)

        print(
            f"Data loaded | train: {len(self.data_train)}"
            f" val: {len(self.data_val)}"
            f" test: {len(self.data_test)}"
        )

    # ── patch tokenization ────────────────────────────────────────────────────

    def _to_bins(self, x_phys: np.ndarray) -> np.ndarray:
        """(..., 12) physical float → (..., 12) int [0, NUM_BINS-1]. Each value = 1 token."""
        ratio = (x_phys - self.bin_low) / (self.bin_high - self.bin_low + 1e-8)
        return np.clip(np.floor(ratio * self.num_bins).astype(int), 0, self.num_bins - 1)

    def _from_bins(self, bins: np.ndarray) -> np.ndarray:
        """(..., 12) int [0, NUM_BINS-1] → (..., 12) physical float (bin centre)."""
        ratio = (np.asarray(bins, dtype=np.float32) + 0.5) / self.num_bins
        return (ratio * (self.bin_high - self.bin_low) + self.bin_low).astype(np.float32)

    # ── normalisation helpers (identical API to WorldModel) ───────────────────

    def unnorm_output(self, output, ignore_pl=False):
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output)
        cols = self.columns[:-1] if ignore_pl else self.columns
        return output.cpu() * self.std[cols] + self.mean[cols]

    def unnorm_state_vectors(self, state_vectors):
        arr = np.array(state_vectors)
        orig_shape = arr.shape
        flat = arr.reshape(-1, arr.shape[-1])
        m = self.mean[self.columns].detach().cpu().numpy()
        s = self.std[self.columns].detach().cpu().numpy()
        return ((flat * s) + m).reshape(orig_shape)

    def normalize_pl(self, pl):
        if not isinstance(pl, torch.Tensor):
            pl = torch.tensor(pl, dtype=torch.float32)
        return (pl - self.mean[-1]) / self.std[-1]

    def unnorm_pl(self, pl):
        if not isinstance(pl, torch.Tensor):
            pl = torch.tensor(pl, dtype=torch.float32)
        return pl.cpu() * self.std[-1] + self.mean[-1]

    # ── prompt helpers ────────────────────────────────────────────────────────

    def _unnorm_context(self, context_norm: torch.Tensor) -> np.ndarray:
        """(T, 12) normalized → (T, 12) physical units."""
        m = self.mean[self.columns].detach().cpu().numpy()
        s = self.std[self.columns].detach().cpu().numpy()
        return context_norm.cpu().numpy() * s + m

    def _build_system_prompt(self) -> str:
        """
        System prompt built once after load_data().
        Defines feature order and per-feature bin↔physical mapping so the LLM
        can reason clinically despite receiving bin indices instead of raw values.
        """
        col_order = ", ".join(FEATURE_NAMES)
        bin_table = "\n".join(
            f"  {j:2d}  {FEATURE_NAMES[j]:<14}  "
            f"bin 0 ≈ {self.bin_low[j]:8.2f}  →  bin 999 ≈ {self.bin_high[j]:8.2f}"
            for j in range(self.num_features)
        )
        return (
            "You are a clinical AI assistant expert in Impella LVAD management in the ICU.\n\n"
            "The Impella pump supports the failing heart via P-levels P2–P10. "
            "Higher P-level = more support: higher PumpSpeed/PumpFlow, lower LV preload, lower PULSAT.\n\n"
            "=== VALUE ENCODING ===\n"
            "All hemodynamic values are encoded as INTEGER BIN INDICES 0–999.\n"
            "Each feature is mapped independently: bin 0 = lowest observed, bin 999 = highest observed.\n"
            "Physical ranges per feature (for clinical reasoning):\n"
            f"{bin_table}\n\n"
            "=== FEATURE ORDER (columns 0–11) ===\n"
            f"{col_order}\n\n"
            "=== CLINICAL TARGETS (physical units) ===\n"
            "PumpPressure ≥ 60 mmHg  |  LVEDP < 18 mmHg  |  PULSAT ≥ 20 mmHg  "
            "|  Heart Rate 50–100 bpm  |  SYSTOLIC 90–130 mmHg\n\n"
            "Weaning goal: reduce Pump Level gradually while hemodynamics stay stable.\n\n"
            "=== TASK ===\n"
            f"You receive {self.forecast_horizon} rows of context (one row = one 10-min timestep, "
            f"12 bin indices in the column order above), then a new Pump Level action.\n"
            f"Predict the next {self.forecast_horizon} timesteps in the same format: "
            f"a JSON array of {self.forecast_horizon} rows, each row an array of "
            f"12 bin indices (integers 0–999).\n"
            "Return ONLY the JSON array. No explanations, no markdown."
        )

    def _build_user_prompt(self, ctx_physical: np.ndarray, p_level_int: int) -> str:
        """
        ctx_physical : (T, 12) physical, oldest row first.
        Each row is serialised as 12 space-separated bin indices with no feature labels.
        """
        ctx_bins = self._to_bins(ctx_physical)   # (T, 12) int

        rows = "\n".join(" ".join(str(v) for v in row) for row in ctx_bins)

        # encode the new P-level as a bin index for consistency
        pl_phys  = np.array([[0.0] * 11 + [float(p_level_int)]])
        pl_bin   = int(self._to_bins(pl_phys)[0, -1])

        return (
            f"## Context ({len(ctx_bins)} timesteps, oldest first):\n"
            f"{rows}\n\n"
            f"## Action: new Pump Level = P{p_level_int} (bin {pl_bin})\n\n"
            f"## Predict next {self.forecast_horizon} timesteps as a JSON array of arrays:\n"
            "[\n"
            + ",\n".join(
                "  [" + ", ".join("<int>" for _ in range(self.num_features)) + "]"
                for _ in range(self.forecast_horizon)
            )
            + "\n]"
        )

    # ── LLM inference ─────────────────────────────────────────────────────────

    def _run_llm(self, user_prompt: str, n: int = 1) -> list:
        """Generate n completions in a single forward pass."""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user",   "content": user_prompt},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids.to(self.llm.device)
        else:
            input_ids = encoded.to(self.llm.device)

        with torch.no_grad():
            output_ids = self.llm.generate(
                input_ids,
                max_new_tokens=256,   # 6 rows × 12 bin indices + JSON brackets ≈ 150 tokens
                temperature=self.temperature,
                do_sample=True,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = input_ids.shape[1]
        return [
            self.tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
            for ids in output_ids
        ]

    def _parse_response(self, text: str, p_level_int: int) -> np.ndarray | None:
        """
        Parse a JSON array-of-arrays of bin indices.
        Returns (forecast_horizon, 12) float32 in physical units, or None on failure.
        """
        text = re.sub(r"```(?:json)?|```", "", text).strip()

        # extract outermost [ ... ] using bracket counting
        start = text.find("[")
        if start == -1:
            return None
        depth, end = 0, -1
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return None

        try:
            arr = json.loads(text[start:end])
            if not isinstance(arr, list) or len(arr) < self.forecast_horizon:
                return None
            bin_rows = []
            for row in arr[: self.forecast_horizon]:
                if len(row) != self.num_features:
                    return None
                bins = [int(v) for v in row]
                # enforce the requested P-level bin for the last feature
                bins[-1] = int(self._to_bins(np.array([[0.0] * 11 + [float(p_level_int)]]))[0, -1])
                bin_rows.append(bins)
            return self._from_bins(np.array(bin_rows, dtype=np.int32))  # (6, 12) physical
        except (json.JSONDecodeError, ValueError, TypeError, IndexError):
            return None

    def _fallback(self, context_norm: torch.Tensor, p_level_int: int) -> np.ndarray:
        """Last-observation carry-forward in physical units."""
        last = self._unnorm_context(context_norm)[-1].copy()
        last[-1] = float(p_level_int)
        return np.tile(last, (self.forecast_horizon, 1)).astype(np.float32)

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, context_norm: torch.Tensor, p_level_int: int):
        """
        context_norm : (T, 12) normalized tensor.
        p_level_int  : int P-level action (2–10).

        Returns
        -------
        mean_norm : (forecast_horizon, 12) normalized tensor
        std_norm  : (forecast_horizon, 12) normalized tensor
        """
        ctx_physical = self._unnorm_context(context_norm)
        user_prompt  = self._build_user_prompt(ctx_physical, p_level_int)

        raws    = self._run_llm(user_prompt, n=self.num_samples)
        samples = [p for raw in raws if (p := self._parse_response(raw, p_level_int)) is not None]

        if not samples:
            print(f"[LLMWorldModel] All {self.num_samples} responses unparseable; using fallback.")
            samples = [self._fallback(context_norm, p_level_int)]

        arr       = np.stack(samples)
        mean_phys = arr.mean(axis=0)
        std_phys  = arr.std(axis=0)

        m = self.mean[self.columns].detach().cpu().numpy()
        s = self.std[self.columns].detach().cpu().numpy()

        mean_norm = torch.tensor((mean_phys - m) / s, dtype=torch.float32)
        std_norm  = torch.tensor(std_phys / s,         dtype=torch.float32)
        return mean_norm, std_norm

    def step(self, x: torch.Tensor, pl) -> torch.Tensor:
        """Matches WorldModel.step() signature exactly."""
        if isinstance(pl, int):
            p_level_int = pl
        else:
            p_level_int = int(round(self.unnorm_pl(
                pl.float().mean() if isinstance(pl, torch.Tensor) else torch.tensor(pl)
            ).item()))
        context = x.squeeze(0)
        mean_norm, _ = self.predict(context, p_level_int)
        return mean_norm.unsqueeze(0)

    def sample_multiple(self, x: torch.Tensor, pl, num_samples: int = 10) -> torch.Tensor:
        """Returns (num_samples, 1, forecast_horizon, 12) normalized tensor."""
        if isinstance(pl, int):
            p_level_int = pl
        else:
            p_level_int = int(round(self.unnorm_pl(
                pl.float().mean() if isinstance(pl, torch.Tensor) else torch.tensor(pl)
            ).item()))

        ctx_physical = self._unnorm_context(x.squeeze(0))
        user_prompt  = self._build_user_prompt(ctx_physical, p_level_int)

        m = self.mean[self.columns].detach().cpu().numpy()
        s = self.std[self.columns].detach().cpu().numpy()

        raws    = self._run_llm(user_prompt, n=num_samples)
        samples = []
        for raw in raws:
            pred = self._parse_response(raw, p_level_int)
            if pred is not None:
                norm = torch.tensor((pred - m) / s, dtype=torch.float32)
                samples.append(norm.unsqueeze(0))

        if not samples:
            fallback = self._fallback(x.squeeze(0), p_level_int)
            norm     = torch.tensor((fallback - m) / s, dtype=torch.float32)
            samples  = [norm.unsqueeze(0)] * num_samples

        return torch.stack(samples)


# ── Digit-token world model ───────────────────────────────────────────────────

class DigitTokenWorldModel(LLMWorldModel):
    """
    LLM world model with per-digit tokenization on z-score normalized values.

    Pipeline:
      1. Z-score normalize using training mean/std (same as WorldModel/TimeSeriesDataset).
      2. Encode each normalized scalar as individual digit tokens:
           sign(+/-) | integer digits (omitted when integer part is 0) | 3 decimal digits
      3. LLM predicts next timesteps in the same format.
      4. Decode digit tokens back to normalized space; unnorm only for visualization.

    Encoding examples:
      +0.123  ->  + 1 2 3
      +1.230  ->  + 1 2 3 0
      +12.300 ->  + 1 2 3 0 0
      -0.456  ->  - 4 5 6
      -1.230  ->  - 1 2 3 0

    All tokens (+, -, 0-9) are single tokens in the Llama 3 BPE vocabulary.
    Values within a row are separated by ' | '; rows are newline-separated.
    """

    # ── value encoding / decoding ─────────────────────────────────────────────

    def _encode_value(self, v: float) -> str:
        sign  = '+' if v >= 0 else '-'
        v_abs = min(abs(float(v)), 999.999)
        int_part = int(v_abs)
        dec_int  = min(round((v_abs - int_part) * 1000), 999)
        dec_str  = f"{dec_int:03d}"
        if int_part == 0:
            tokens = [sign] + list(dec_str)
        else:
            tokens = [sign] + list(str(int_part)) + list(dec_str)
        return ' '.join(tokens)

    def _decode_value(self, s: str) -> float:
        tokens = s.strip().split()
        if not tokens:
            return 0.0
        sign = 1.0
        i = 0
        if tokens[0] in ('+', '-'):
            sign = -1.0 if tokens[0] == '-' else 1.0
            i = 1
        remaining = tokens[i:]
        if len(remaining) >= 3:
            dec_tokens = remaining[-3:]
            int_tokens = remaining[:-3]
        else:
            dec_tokens = (remaining + ['0', '0', '0'])[:3]
            int_tokens = []
        try:
            int_part = int(''.join(int_tokens)) if int_tokens else 0
            dec_part = int(''.join(dec_tokens)) / 1000.0
            return sign * (int_part + dec_part)
        except ValueError:
            return 0.0

    def _encode_row(self, row_norm: np.ndarray) -> str:
        return ' | '.join(self._encode_value(v) for v in row_norm)

    def _decode_row(self, s: str) -> np.ndarray | None:
        parts = s.split(' | ')
        if len(parts) != self.num_features:
            return None
        try:
            return np.array([self._decode_value(p) for p in parts], dtype=np.float32)
        except Exception:
            return None

    # ── prompts ───────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        norm_table = "\n".join(
            f"  {j:2d}  {FEATURE_NAMES[j]:<14}  "
            f"mean={self.mean[self.columns[j]].item():8.3f}  "
            f"std={self.std[self.columns[j]].item():8.3f}"
            for j in range(self.num_features)
        )
        col_order = ", ".join(FEATURE_NAMES)
        return (
            "You are a clinical AI assistant expert in Impella LVAD management in the ICU.\n\n"
            "The Impella pump supports the failing heart via P-levels P2–P10. "
            "Higher P-level = more support: higher PumpSpeed/PumpFlow, lower LV preload, lower PULSAT.\n\n"
            "=== VALUE ENCODING ===\n"
            "All values are Z-SCORE NORMALIZED using training-set mean and std.\n"
            "Each normalized value is encoded as INDIVIDUAL DIGIT TOKENS:\n"
            "  sign (+/-), then integer digits (omitted when integer part is 0), "
            "then exactly 3 decimal digits (zero-padded on the right).\n\n"
            "Encoding examples:\n"
            "  +0.123  ->  + 1 2 3\n"
            "  +1.230  ->  + 1 2 3 0\n"
            "  -0.456  ->  - 4 5 6\n"
            "  -1.230  ->  - 1 2 3 0\n\n"
            "Normalization statistics per feature:\n"
            f"{norm_table}\n\n"
            "=== FEATURE ORDER (columns 0–11) ===\n"
            f"{col_order}\n\n"
            "=== CLINICAL TARGETS (physical units) ===\n"
            "PumpPressure ≥ 60 mmHg  |  LVEDP < 18 mmHg  |  PULSAT ≥ 20 mmHg  "
            "|  Heart Rate 50–100 bpm  |  SYSTOLIC 90–130 mmHg\n\n"
            "Weaning goal: reduce Pump Level gradually while hemodynamics stay stable.\n\n"
            "=== TASK ===\n"
            f"You receive {self.forecast_horizon} rows of context "
            f"(one row per 10-min timestep, 12 digit-token values separated by ' | ').\n"
            f"Predict the next {self.forecast_horizon} timesteps in the SAME format: "
            f"{self.forecast_horizon} rows, one per line, 12 digit-token values per row separated by ' | '.\n"
            "Return ONLY the rows. No explanations, no markdown."
        )

    def _build_user_prompt(self, ctx_norm: np.ndarray, p_level_int: int) -> str:
        """ctx_norm: (T, 12) already z-score normalized."""
        rows     = "\n".join(self._encode_row(row) for row in ctx_norm)
        pl_norm  = float(self.normalize_pl(torch.tensor(float(p_level_int))).item())
        pl_enc   = self._encode_value(pl_norm)
        return (
            f"## Context ({len(ctx_norm)} timesteps, oldest first):\n"
            f"{rows}\n\n"
            f"## Action: new Pump Level = P{p_level_int} (normalized: {pl_enc})\n\n"
            f"## Predict next {self.forecast_horizon} timesteps "
            f"(12 digit-token values per row, separated by ' | '):"
        )

    # ── response parsing ──────────────────────────────────────────────────────

    def _parse_response(self, text: str, p_level_int: int) -> np.ndarray | None:
        """Returns (forecast_horizon, 12) normalized float32, or None on failure."""
        pl_norm = float(self.normalize_pl(torch.tensor(float(p_level_int))).item())
        rows = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            row = self._decode_row(line)
            if row is not None:
                row[-1] = pl_norm   # enforce requested P-level
                rows.append(row)
            if len(rows) == self.forecast_horizon:
                break
        if len(rows) < self.forecast_horizon:
            return None
        return np.stack(rows).astype(np.float32)

    # ── fallback (normalized space) ───────────────────────────────────────────

    def _fallback(self, context_norm: torch.Tensor, p_level_int: int) -> np.ndarray:
        last = context_norm[-1].cpu().numpy().copy()
        last[-1] = float(self.normalize_pl(torch.tensor(float(p_level_int))).item())
        return np.tile(last, (self.forecast_horizon, 1)).astype(np.float32)

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, context_norm: torch.Tensor, p_level_int: int):
        ctx_np      = context_norm.cpu().numpy()          # (T, 12) normalized
        user_prompt = self._build_user_prompt(ctx_np, p_level_int)

        raws    = self._run_llm(user_prompt, n=self.num_samples)
        samples = [p for raw in raws if (p := self._parse_response(raw, p_level_int)) is not None]

        if not samples:
            print(f"[DigitTokenWorldModel] All {self.num_samples} responses unparseable; using fallback.")
            samples = [self._fallback(context_norm, p_level_int)]

        arr       = np.stack(samples)
        mean_norm = torch.tensor(arr.mean(axis=0), dtype=torch.float32)
        std_norm  = torch.tensor(arr.std(axis=0),  dtype=torch.float32)
        return mean_norm, std_norm

    def step(self, x: torch.Tensor, pl) -> torch.Tensor:
        if isinstance(pl, int):
            p_level_int = pl
        else:
            p_level_int = int(round(self.unnorm_pl(
                pl.float().mean() if isinstance(pl, torch.Tensor) else torch.tensor(pl)
            ).item()))
        mean_norm, _ = self.predict(x.squeeze(0), p_level_int)
        return mean_norm.unsqueeze(0)

    def sample_multiple(self, x: torch.Tensor, pl, num_samples: int = 10) -> torch.Tensor:
        if isinstance(pl, int):
            p_level_int = pl
        else:
            p_level_int = int(round(self.unnorm_pl(
                pl.float().mean() if isinstance(pl, torch.Tensor) else torch.tensor(pl)
            ).item()))

        ctx_np      = x.squeeze(0).cpu().numpy()
        user_prompt = self._build_user_prompt(ctx_np, p_level_int)
        raws        = self._run_llm(user_prompt, n=num_samples)
        samples     = []
        for raw in raws:
            pred = self._parse_response(raw, p_level_int)
            if pred is not None:
                samples.append(torch.tensor(pred, dtype=torch.float32).unsqueeze(0))

        if not samples:
            fallback = self._fallback(x.squeeze(0), p_level_int)
            t        = torch.tensor(fallback, dtype=torch.float32).unsqueeze(0)
            samples  = [t] * num_samples

        return torch.stack(samples)


# ── Few-shot digit-token world model ──────────────────────────────────────────

class FewShotDigitTokenWorldModel(DigitTokenWorldModel):
    """
    Combines digit-token encoding (DigitTokenWorldModel) with k-NN few-shot
    retrieval (FewShotLLMWorldModel). The k nearest training trajectories are
    prepended as alternating user/assistant turns before the real query, with
    all values encoded as digit tokens in normalized space.
    """

    def __init__(self, k_shot: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.k_shot = k_shot
        self._train_last_norm = None   # (N, 12) last context state, normalized
        self._train_ctx_norm  = None   # (N, T, 12) full context, normalized
        self._train_fc_norm   = None   # (N, T, 12) forecast incl. PL, normalized
        self._train_pl_int    = None   # (N,) P-level int

    # ── data loading ──────────────────────────────────────────────────────────

    def load_data(self, path: str):
        super().load_data(path)
        if self.k_shot > 0:
            self._build_retrieval_index()

    def _build_retrieval_index(self):
        n = len(self.data_train)
        T = self.forecast_horizon
        print(f"[FewShotDigitToken] Building retrieval index over {n} training samples ...")

        last_norm = np.empty((n, self.num_features),    dtype=np.float32)
        ctx_norm  = np.empty((n, T, self.num_features), dtype=np.float32)
        fc_norm   = np.empty((n, T, self.num_features), dtype=np.float32)  # incl. PL
        pl_int    = np.empty(n, dtype=np.int32)

        for i in range(n):
            xi, pli, yi = self.data_train[i]
            xi_np = np.asarray(xi, dtype=np.float32)                               # (T, 12)
            yi_np = np.asarray(yi, dtype=np.float32).reshape(T, self.num_features - 1)  # (T, 11)

            pl_raw      = float(np.asarray(pli).mean())
            pl_phys     = pl_raw * float(self.std[-1]) + float(self.mean[-1])
            pl_norm_col = np.full((T, 1), pl_raw, dtype=np.float32)

            last_norm[i] = xi_np[-1]
            ctx_norm[i]  = xi_np
            fc_norm[i]   = np.concatenate([yi_np, pl_norm_col], axis=1)  # (T, 12)
            pl_int[i]    = int(round(pl_phys))

        self._train_last_norm = last_norm
        self._train_ctx_norm  = ctx_norm
        self._train_fc_norm   = fc_norm
        self._train_pl_int    = pl_int
        print(f"[FewShotDigitToken] Retrieval index ready ({n} examples).")

    # ── retrieval ─────────────────────────────────────────────────────────────

    def _retrieve_examples(self, query_last_norm: np.ndarray, k: int) -> list:
        """Returns list of (ctx_norm, pl_int, fc_norm) — all in normalized space."""
        diffs = self._train_last_norm - query_last_norm[None, :]
        dists = (diffs ** 2).sum(axis=1)
        k_eff = min(k, len(dists))
        idxs  = np.argpartition(dists, k_eff)[:k_eff]
        idxs  = idxs[np.argsort(dists[idxs])]
        return [
            (self._train_ctx_norm[i], int(self._train_pl_int[i]), self._train_fc_norm[i])
            for i in idxs
        ]

    # ── few-shot prompt helpers ───────────────────────────────────────────────

    def _format_example_user(self, ctx_norm: np.ndarray, pl_int: int) -> str:
        rows   = "\n".join(self._encode_row(row) for row in ctx_norm)
        pl_enc = self._encode_value(float(self.normalize_pl(torch.tensor(float(pl_int))).item()))
        return (
            f"## Context ({len(ctx_norm)} timesteps, oldest first):\n"
            f"{rows}\n\n"
            f"## Action: new Pump Level = P{pl_int} (normalized: {pl_enc})\n\n"
            f"## Predict next {self.forecast_horizon} timesteps "
            f"(12 digit-token values per row, separated by ' | '):"
        )

    def _format_example_assistant(self, fc_norm: np.ndarray, pl_int: int) -> str:
        pl_norm = float(self.normalize_pl(torch.tensor(float(pl_int))).item())
        rows    = fc_norm.copy()
        rows[:, -1] = pl_norm
        return "\n".join(self._encode_row(row) for row in rows)

    # ── LLM call with few-shot injection ─────────────────────────────────────

    def _run_llm(self, query_msg: str, few_shot_messages: list | None = None, n: int = 1) -> list:
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        if few_shot_messages:
            for user_str, asst_str in few_shot_messages:
                messages.append({"role": "user",      "content": user_str})
                messages.append({"role": "assistant", "content": asst_str})
        messages.append({"role": "user", "content": query_msg})

        encoded = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids.to(self.llm.device)
        else:
            input_ids = encoded.to(self.llm.device)

        results = []
        with torch.no_grad():
            for _ in range(n):
                out = self.llm.generate(
                    input_ids,
                    max_new_tokens=512,   # digit tokens are ~5× more verbose than bins
                    temperature=self.temperature,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                results.append(
                    self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                )
        return results

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, context_norm: torch.Tensor, p_level_int: int):
        ctx_np      = context_norm.cpu().numpy()
        user_prompt = self._build_user_prompt(ctx_np, p_level_int)

        few_shot_msgs = None
        if self.k_shot > 0 and self._train_last_norm is not None:
            examples      = self._retrieve_examples(ctx_np[-1], self.k_shot)
            few_shot_msgs = [
                (self._format_example_user(ctx_ex, pl_ex),
                 self._format_example_assistant(fc_ex, pl_ex))
                for ctx_ex, pl_ex, fc_ex in examples
            ]

        raws    = self._run_llm(user_prompt, few_shot_messages=few_shot_msgs, n=self.num_samples)
        samples = [p for raw in raws if (p := self._parse_response(raw, p_level_int)) is not None]

        if not samples:
            print(f"[FewShotDigitToken] All {self.num_samples} responses unparseable; using fallback.")
            samples = [self._fallback(context_norm, p_level_int)]

        arr       = np.stack(samples)
        mean_norm = torch.tensor(arr.mean(axis=0), dtype=torch.float32)
        std_norm  = torch.tensor(arr.std(axis=0),  dtype=torch.float32)
        return mean_norm, std_norm

    def sample_multiple(self, x: torch.Tensor, pl, num_samples: int = 10) -> torch.Tensor:
        if isinstance(pl, int):
            p_level_int = pl
        else:
            p_level_int = int(round(self.unnorm_pl(
                pl.float().mean() if isinstance(pl, torch.Tensor) else torch.tensor(pl)
            ).item()))

        ctx_np      = x.squeeze(0).cpu().numpy()
        user_prompt = self._build_user_prompt(ctx_np, p_level_int)

        few_shot_msgs = None
        if self.k_shot > 0 and self._train_last_norm is not None:
            examples      = self._retrieve_examples(ctx_np[-1], self.k_shot)
            few_shot_msgs = [
                (self._format_example_user(ctx_ex, pl_ex),
                 self._format_example_assistant(fc_ex, pl_ex))
                for ctx_ex, pl_ex, fc_ex in examples
            ]

        raws    = self._run_llm(user_prompt, few_shot_messages=few_shot_msgs, n=num_samples)
        samples = []
        for raw in raws:
            pred = self._parse_response(raw, p_level_int)
            if pred is not None:
                samples.append(torch.tensor(pred, dtype=torch.float32).unsqueeze(0))

        if not samples:
            fallback = self._fallback(x.squeeze(0), p_level_int)
            t        = torch.tensor(fallback, dtype=torch.float32).unsqueeze(0)
            samples  = [t] * num_samples

        return torch.stack(samples)
