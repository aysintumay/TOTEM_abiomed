"""
Zero-shot Chronos (original, T5/GPT2 + MeanScaleUniformBins tokenizer) world model
for Impella pump weaning. Same interface contract as world_model.py / gormpo_world_model.py
(see mcs.md) -- a drop-in baseline to compare against the medllama and GORMPO world models
before swapping MeanScaleUniformBins for GormpoTokenizer.

Unlike world_model.py/gormpo_world_model.py, no prompt engineering is involved: Chronos
takes each of the 11 forecast channels as an independent univariate series (its pretrained
tokenizer/model have no notion of the other channels or the clinical task) and returns
per-channel sample paths directly from ChronosPipeline.predict -- no parsing, no fallback
needed, since generation can't produce anything unparseable.

Run with: python eval_chronos_world_model.py --num_examples 20
"""
import pickle
import sys
import os

import numpy as np
import torch
from chronos import BaseChronosPipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "abiomed_env"))
from model import TimeSeriesDataset  # noqa: E402

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


class ChronosWorldModel:
    """Interface matches abiomed_env.model.WorldModel / world_model.py's classes."""

    columns: list = [i for i in range(0, 13) if i != 11]

    def __init__(
        self,
        model_id: str = "amazon/chronos-t5-small",
        forecast_horizon: int = 6,
        num_samples: int = 20,
        device: str = "auto",
    ):
        self.forecast_horizon = forecast_horizon
        self.num_samples = num_samples

        # populated by load_data()
        self.mean = self.std = None
        self.data_train = self.data_val = self.data_test = None

        print(f"Loading Chronos pipeline: {model_id}")
        self.pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map=device, dtype="auto")
        print("Model ready.")

    def load_model(self, path: str):
        """No-op stub — pipeline is loaded in __init__. Exists for interface compatibility."""
        pass

    # ── data loading ──────────────────────────────────────────────────────────

    def load_data(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)

        mean, std = data["mean"], data["std"]
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32)
            std = torch.tensor(std, dtype=torch.float32)
        self.mean, self.std = mean, std

        norm = lambda split: ((data[split] - data["mean"]) / data["std"])[:, :, self.columns]
        self.data_train = TimeSeriesDataset(norm("train"), self.forecast_horizon, self.forecast_horizon)
        self.data_val = TimeSeriesDataset(norm("val"), self.forecast_horizon, self.forecast_horizon)
        self.data_test = TimeSeriesDataset(norm("test"), self.forecast_horizon, self.forecast_horizon)

        print(
            f"Data loaded | train: {len(self.data_train)}"
            f" val: {len(self.data_val)}"
            f" test: {len(self.data_test)}"
        )

    # ── normalisation helpers (identical API to WorldModel/world_model.py) ────

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

    def _unnorm_context(self, context_norm: torch.Tensor) -> np.ndarray:
        m = self.mean[self.columns].detach().cpu().numpy()
        s = self.std[self.columns].detach().cpu().numpy()
        return context_norm.cpu().numpy() * s + m

    # ── Chronos inference ─────────────────────────────────────────────────────

    def _generate_samples(self, context_norm: torch.Tensor, p_level_int: int, n: int) -> np.ndarray:
        """Returns (n, forecast_horizon, 12) physical-unit samples.

        Each of the 11 forecast channels (Pump Level excluded -- it's the action) is
        given to Chronos as an independent univariate series in one batched call
        (batch dim = channel); Pump Level itself is just carried forward as the
        requested action, same as the fallback path in the other world models.
        """
        ctx_physical = self._unnorm_context(context_norm)  # (T, 12)
        context_tensor = torch.tensor(ctx_physical[:, :-1].T, dtype=torch.float32)  # (11, T)

        samples = self.pipeline.predict(
            context_tensor, prediction_length=self.forecast_horizon, num_samples=n
        )  # (11, n, forecast_horizon)
        samples = samples.permute(1, 2, 0).numpy()  # (n, forecast_horizon, 11)

        pl_col = np.full((n, self.forecast_horizon, 1), float(p_level_int), dtype=np.float32)
        return np.concatenate([samples, pl_col], axis=-1).astype(np.float32)  # (n, forecast_horizon, 12)

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, context_norm: torch.Tensor, p_level_int: int):
        """
        context_norm : (T, 12) normalized tensor.
        p_level_int  : int P-level action (2–10).
        Returns (mean_norm, std_norm), each (forecast_horizon, 12) normalized tensor.
        """
        arr = self._generate_samples(context_norm, p_level_int, self.num_samples)
        mean_phys, std_phys = arr.mean(axis=0), arr.std(axis=0)

        m = self.mean[self.columns].detach().cpu().numpy()
        s = self.std[self.columns].detach().cpu().numpy()
        mean_norm = torch.tensor((mean_phys - m) / s, dtype=torch.float32)
        std_norm = torch.tensor(std_phys / s, dtype=torch.float32)
        return mean_norm, std_norm

    def _resolve_pl(self, pl) -> int:
        """pl is either a plain P-level int, or a normalized tensor/array to unnormalize first."""
        if isinstance(pl, int):
            return pl
        pl_t = pl.float().mean() if isinstance(pl, torch.Tensor) else torch.tensor(pl, dtype=torch.float32)
        return int(round(self.unnorm_pl(pl_t).item()))

    def step(self, x: torch.Tensor, pl) -> torch.Tensor:
        """Matches WorldModel.step() signature exactly."""
        mean_norm, _ = self.predict(x.squeeze(0), self._resolve_pl(pl))
        return mean_norm.unsqueeze(0)

    def sample_multiple(self, x: torch.Tensor, pl, num_samples: int = 10) -> torch.Tensor:
        """Returns (num_samples, 1, forecast_horizon, 12) normalized tensor -- genuinely
        distinct per-sample generations, not the mean repeated."""
        context = x.squeeze(0)
        arr = self._generate_samples(context, self._resolve_pl(pl), num_samples)  # (num_samples, T, 12) physical

        m = self.mean[self.columns].detach().cpu().numpy()
        s = self.std[self.columns].detach().cpu().numpy()
        norm = torch.tensor((arr - m) / s, dtype=torch.float32).unsqueeze(1)  # (num_samples, 1, T, 12)
        return norm
