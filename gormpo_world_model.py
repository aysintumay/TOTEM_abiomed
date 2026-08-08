"""
Few-shot, in-context-learning LLM world model for Impella pump weaning that uses
the GORMPO tokenizer (forecasting/lib/models/tokenizer.py::GormpoTokenizer) as its
encoding scheme, instead of world_model.py's raw-value bin/digit tokenization.

Same interface contract as world_model.py's classes (see mcs.md) and the same
overall pattern as FewShotDigitTokenWorldModel: a frozen LLM (no fine-tuning, no
LoRA -- pure .generate() calls), k-NN retrieval of similar training trajectories
injected as alternating user/assistant few-shot turns, then the real query.

What's different from world_model.py: each channel's context/target patch is
encoded through the trained GormpoTokenizer into its actual token schema --
q_shape (a codebook index per compressed position, Eq 19), q_mu/q_sigma/q_min/q_max
(quantized scale statistics, Eq 20-23), q_r (clinical reward bin, Eq 24, context
only for MAP/HR/Pulsatility channels) -- and the system prompt explicitly teaches
the LLM what each of these token components means, since (unlike a raw-value bin
index) a VQ shape code has no inherent physical ordering the LLM could infer on its
own; the few-shot examples are what let it ground codes to outcomes in context.

Tokenizer used: the from-scratch, MCS-specialist tokenizer that showed no
overfitting across 2000 epochs of training (saved_models/gormpo_tokenizer_mcs_scratch_long2000
in the forecasting/ tree) -- NOT the frozen-generalist-encoder variant.
"""

import os
import pickle
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "abiomed_env"))
sys.path.insert(0, os.path.join(_ROOT, "forecasting"))
from model import TimeSeriesDataset  # noqa: E402

# Reused verbatim from world_model.py so both world models describe the same 12
# channels the same way; channel 0 is called "MAP" in abiomed_env/reward_func.py's
# formulas internally but "PumpPressure" here -- flagging this naming inconsistency
# rather than silently resolving it, since I can't confirm from data alone which is
# the clinically-intended name for this channel.
FEATURE_NAMES = [
    "PumpPressure",  # 0  — differential pressure (mmHg); target ≥ 60 (used as "MAP" in reward_func.py)
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

MAP_IDX, PULSAT_IDX, HR_IDX = 0, 7, 9  # matches abiomed_env.cost_func indices

_COLUMNS = [i for i in range(0, 13) if i != 11]  # matches GormpoFewShotWorldModel.columns


def compute_variable_ylims(data_path: str, low_pct: float = 1, high_pct: float = 99) -> dict:
    """Per-variable (low, high) physical-unit bounds from the training split's raw
    values (1st/99th percentile by default), for the 11 forecast variables (Pump
    Level excluded -- it's the action, given a fixed small-integer range instead).

    Model-independent: only depends on the training data distribution, not on any
    particular tokenizer/forecaster. This is the single source of truth for both
    GormpoFewShotWorldModel's output-clipping bounds (see _build_retrieval_index)
    and any visualization script's fixed y-axis limits -- computing it once, here,
    keeps "what the model is allowed to output" and "what the plot shows" in sync,
    and lets any other model's comparison script reuse the exact same bounds
    without having to load a whole model just to get them.
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    train_phys = np.asarray(data["train"])[:, :, _COLUMNS[:-1]].reshape(-1, len(_COLUMNS) - 1).astype(np.float32)
    low = np.percentile(train_phys, low_pct, axis=0)
    high = np.percentile(train_phys, high_pct, axis=0)
    ylims = {FEATURE_NAMES[j]: (float(low[j]), float(high[j])) for j in range(len(_COLUMNS) - 1)}
    ylims[FEATURE_NAMES[-1]] = (1.0, 11.0)  # Pump Level: fixed natural range (2-10), not data-percentile-derived
    return ylims


class GormpoFewShotWorldModel:
    """Interface matches abiomed_env.model.WorldModel / world_model.py's classes."""

    columns: list = [i for i in range(0, 13) if i != 11]

    def __init__(
        self,
        tokenizer_path: str,
        model_id: str = "m42-health/Llama3-Med42-8B",
        forecast_horizon: int = 6,
        k_shot: int = 5,
        num_samples: int = 10,
        temperature: float = 0.8,
        device: str = "auto",
        load_in_4bit: bool = False,
    ):
        self.forecast_horizon = forecast_horizon
        self.k_shot = k_shot
        self.num_samples = num_samples
        self.temperature = temperature

        # populated by load_data()
        self.mean = self.std = None
        self.data_train = self.data_val = self.data_test = None
        self._train_last_norm = None   # (N, 12) last context state, normalized -- retrieval key
        self._train_ctx_phys = None    # (N, T, 12) context, physical units
        self._train_fc_phys = None     # (N, T, 12) forecast incl. PL, physical units
        self._train_pl_int = None      # (N,) P-level int
        self._phys_low = None          # (11,) 1st-percentile clipping bound per variable
        self._phys_high = None         # (11,) 99th-percentile clipping bound per variable

        print(f"Loading GORMPO tokenizer: {tokenizer_path}")
        self.gormpo_tokenizer = torch.load(tokenizer_path, map_location="cpu", weights_only=False)
        self.gormpo_tokenizer.eval()
        for p in self.gormpo_tokenizer.parameters():
            p.requires_grad_(False)
        assert self.gormpo_tokenizer.patch_len == forecast_horizon, (
            f"tokenizer patch_len={self.gormpo_tokenizer.patch_len} != forecast_horizon={forecast_horizon}"
        )
        self.num_channels = self.gormpo_tokenizer.num_channels
        self.num_embeddings = self.gormpo_tokenizer.vq._embedding.weight.shape[0]
        self.num_bins = self.gormpo_tokenizer.q_mu.num_bins
        self.L = self.gormpo_tokenizer.patch_len // self.gormpo_tokenizer.compression_factor

        print(f"Loading tokenizer: {model_id}")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_id)

        model_kwargs = {"dtype": torch.bfloat16, "device_map": device}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )

        print(f"Loading model: {model_id}")
        self.llm = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.llm.eval()
        self.gormpo_tokenizer.to(self.llm.device)
        print("Model ready.")

    def load_model(self, path: str):
        """No-op stub — everything is loaded in __init__. Exists for interface compatibility."""
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

        self.ylims = compute_variable_ylims(path)  # {feature_name: (low, high)}, model-independent
        self._phys_low = np.array([self.ylims[n][0] for n in FEATURE_NAMES[:-1]], dtype=np.float32)
        self._phys_high = np.array([self.ylims[n][1] for n in FEATURE_NAMES[:-1]], dtype=np.float32)

        if self.k_shot > 0:
            self._build_retrieval_index()

    def _build_retrieval_index(self):
        n = len(self.data_train)
        T = self.forecast_horizon
        print(f"[GormpoFewShot] Building retrieval index over {n} training samples ...")

        last_norm = np.empty((n, self.num_channels), dtype=np.float32)
        ctx_phys = np.empty((n, T, self.num_channels), dtype=np.float32)
        fc_phys = np.empty((n, T, self.num_channels), dtype=np.float32)
        pl_int = np.empty(n, dtype=np.int32)

        m = self.mean[self.columns].numpy()
        s = self.std[self.columns].numpy()

        for i in range(n):
            xi, pli, yi = self.data_train[i]
            xi_np = np.asarray(xi, dtype=np.float32)  # (T, 12) normalized
            yi_np = np.asarray(yi, dtype=np.float32).reshape(T, self.num_channels - 1)

            pl_raw = float(np.asarray(pli).mean())
            pl_phys = pl_raw * float(self.std[-1]) + float(self.mean[-1])
            pl_norm_col = np.full((T, 1), pl_raw, dtype=np.float32)
            fc_norm = np.concatenate([yi_np, pl_norm_col], axis=1)  # (T, 12)

            last_norm[i] = xi_np[-1]
            ctx_phys[i] = xi_np * s + m
            fc_phys[i] = fc_norm * s + m
            pl_int[i] = int(round(pl_phys))

        self._train_last_norm = last_norm
        self._train_ctx_phys = ctx_phys
        self._train_fc_phys = fc_phys
        self._train_pl_int = pl_int
        print(f"[GormpoFewShot] Retrieval index ready ({n} examples).")

    def _retrieve_examples(self, query_last_norm: np.ndarray, k: int) -> list:
        """Returns list of (ctx_phys, pl_int, fc_phys), physical units."""
        diffs = self._train_last_norm - query_last_norm[None, :]
        dists = (diffs ** 2).sum(axis=1)
        k_eff = min(k, len(dists))
        idxs = np.argpartition(dists, k_eff)[:k_eff]
        idxs = idxs[np.argsort(dists[idxs])]
        return [
            (self._train_ctx_phys[i], int(self._train_pl_int[i]), self._train_fc_phys[i])
            for i in idxs
        ]

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

    # ── GORMPO tokenization ───────────────────────────────────────────────────

    @torch.no_grad()
    def _tokenize_patch(self, x_phys_TxC: np.ndarray) -> dict:
        """x_phys_TxC: (T=patch_len, C=num_channels) physical units, time-major.
        Returns tokenizer.encode()'s dict, batch size 1, channel-major (1, N, ...)."""
        x = torch.from_numpy(x_phys_TxC).float().transpose(0, 1).unsqueeze(0)  # (1, N, T)
        x = x.to(self.gormpo_tokenizer.vq._embedding.weight.device)
        return self.gormpo_tokenizer.encode(x)

    @torch.no_grad()
    def _detokenize_channels(self, q_shape_per_channel, q_mu_per_channel, q_sigma_per_channel) -> np.ndarray:
        """
        q_shape_per_channel: (N, L) predicted shape code indices
        q_mu_per_channel, q_sigma_per_channel: (N,) predicted scale bin indices
        Returns (patch_len, N) physical units, time-major.
        """
        device = self.gormpo_tokenizer.vq._embedding.weight.device
        q_shape = torch.as_tensor(q_shape_per_channel, dtype=torch.long, device=device)
        mu_idx = torch.as_tensor(q_mu_per_channel, dtype=torch.long, device=device).unsqueeze(0)
        sigma_idx = torch.as_tensor(q_sigma_per_channel, dtype=torch.long, device=device).unsqueeze(0)
        mu_hat = self.gormpo_tokenizer.q_mu.dequantize(mu_idx)
        sigma_hat = self.gormpo_tokenizer.q_sigma.dequantize(sigma_idx)

        N, L = q_shape.shape
        codebook = self.gormpo_tokenizer.vq._embedding.weight  # (num_embeddings, code_dim)
        one_hot = torch.nn.functional.one_hot(q_shape.reshape(-1), self.num_embeddings).to(codebook.dtype)
        quantized = torch.matmul(one_hot, codebook).view(N, L, -1).transpose(1, 2)  # (N, code_dim, L)

        x_hat, _ = self.gormpo_tokenizer.decode(quantized, mu_hat, sigma_hat)  # (1, N, patch_len)
        return x_hat.squeeze(0).transpose(0, 1).cpu().numpy()  # (patch_len, N)

    # ── prompt helpers ────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        col_order = ", ".join(FEATURE_NAMES)
        return (
            "You are a clinical AI assistant expert in Impella LVAD management in the ICU.\n\n"
            "The Impella pump supports the failing heart via P-levels P2–P10. "
            "Higher P-level = more support: higher PumpSpeed/PumpFlow, lower LV preload, lower PULSAT.\n\n"
            "=== VALUE ENCODING: GORMPO TOKENIZER ===\n"
            "Each variable's history is summarized as a discrete token, not a raw number, produced by a "
            "learned tokenizer (a cross-channel-attention VQ-VAE). Each token has these parts, always in "
            "this fixed order, space-separated, ONE ROW PER VARIABLE (no labels -- row i is always "
            "variable i from the FEATURE ORDER list below):\n\n"
            "  c0 c1 c2 mu sigma min max reward\n\n"
            "  c0,c1,c2 = three integers in [0, 255], indices into a learned codebook of 256 morphological "
            "patterns. Codes are CATEGORICAL, not ordered by magnitude: code 200 is not \"bigger\" than "
            "code 50, it is a different learned shape. The only way to know what a given code implies is "
            "by comparing it against the worked examples given to you below.\n"
            "  mu,sigma,min,max = each an integer bin in [0, 31], the patch's mean/std/min/max value "
            "quantized into 32 bins, ordered low(0) to high(31) within that variable's own observed range.\n"
            "  reward = an integer in [0, 8]. Only meaningful for PumpPressure (used as MAP), Heart Rate, "
            "and PULSAT (rows 0, 9, 7); 0 for every other row, always. 1-8 encodes increasing clinical "
            "instability risk (based on standard MAP/HR/pulsatility thresholds); it is context only, "
            "never something you predict.\n\n"
            "=== FEATURE ORDER (row 0 - row 10 in every context/prediction block) ===\n"
            f"{col_order}\n\n"
            "=== CLINICAL TARGETS (physical units) ===\n"
            "PumpPressure ≥ 60 mmHg  |  LVEDP < 18 mmHg  |  PULSAT ≥ 20 mmHg  "
            "|  Heart Rate 50–100 bpm  |  SYSTOLIC 90–130 mmHg\n\n"
            "Weaning goal: reduce Pump Level gradually while hemodynamics stay stable.\n\n"
            "=== TASK ===\n"
            f"You receive {self.forecast_horizon} timesteps of context, tokenized as 11 rows (one per "
            "variable, Pump Level excluded since it's the action) in the format above, then a new Pump "
            "Level action. Predict the NEXT patch's token for each of the 11 variables: c0 c1 c2 mu sigma "
            "min max (7 integers, no reward).\n"
            "A few worked examples (similar past situations and what actually happened next) are given "
            "before the real query -- use them to infer what the codes and bins mean in context.\n\n"
            "Return ONLY 11 rows, one per line, 7 space-separated integers each, no labels, no "
            "explanations, no markdown -- nothing else."
        )

    def _format_patch_tokens(self, tokens: dict, include_reward: bool) -> str:
        """tokens: dict from _tokenize_patch (batch dim 1). One compact row per channel, no
        labels -- row order matches FEATURE_NAMES, as declared once in the system prompt."""
        lines = []
        for i in range(self.num_channels - 1):  # exclude Pump Level channel (index 11), it's the action
            c0, c1, c2 = tokens["q_shape"][0, i].tolist()
            mu = int(tokens["q_mu"][0, i])
            sigma = int(tokens["q_sigma"][0, i])
            mn = int(tokens["q_min"][0, i])
            mx = int(tokens["q_max"][0, i])
            vals = [c0, c1, c2, mu, sigma, mn, mx]
            if include_reward:
                vals.append(int(tokens["q_r"][0, i]))
            lines.append(" ".join(str(v) for v in vals))
        return "\n".join(lines)

    def _build_user_prompt(self, ctx_tokens: dict, p_level_int: int) -> str:
        return (
            f"## Context ({self.forecast_horizon} timesteps, tokenized, 11 rows):\n"
            f"{self._format_patch_tokens(ctx_tokens, include_reward=True)}\n\n"
            f"## Action: new Pump Level = P{p_level_int}\n\n"
            "## Predict the next patch's token for each variable (11 rows, 7 ints each, see format above):"
        )

    def _format_example_assistant(self, fc_tokens: dict) -> str:
        return self._format_patch_tokens(fc_tokens, include_reward=False)

    # ── LLM inference ─────────────────────────────────────────────────────────

    def _run_llm(self, query_msg: str, few_shot_messages: list = None, n: int = 1) -> list:
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        if few_shot_messages:
            for user_str, asst_str in few_shot_messages:
                messages.append({"role": "user", "content": user_str})
                messages.append({"role": "assistant", "content": asst_str})
        messages.append({"role": "user", "content": query_msg})

        encoded = self.llm_tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        input_ids = encoded.input_ids.to(self.llm.device) if hasattr(encoded, "input_ids") else encoded.to(self.llm.device)

        with torch.no_grad():
            output_ids = self.llm.generate(
                input_ids,
                max_new_tokens=256,  # 11 rows * 7 ints, compact (no labels/JSON) -- measured ~154 tokens
                temperature=self.temperature,
                do_sample=True,
                num_return_sequences=n,
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )
        prompt_len = input_ids.shape[1]
        return [self.llm_tokenizer.decode(ids[prompt_len:], skip_special_tokens=True) for ids in output_ids]

    def _parse_response(self, text: str) -> dict:
        """Parses the compact row format ("c0 c1 c2 mu sigma min max" per line, no
        labels, one row per variable in FEATURE_NAMES order). Returns a tokens dict
        (batch dim 1, like _tokenize_patch's output) or None on failure."""
        text = re.sub(r"```(?:\w+)?|```", "", text).strip()

        def _in_range(v, lo, hi):
            return lo <= v <= hi

        rows = []
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) != 7:
                continue  # skip blank/stray lines (e.g. accidental labels, markdown)
            try:
                rows.append([int(p) for p in parts])
            except ValueError:
                continue  # a non-integer token on this line -- not a valid row
            if len(rows) == self.num_channels - 1:
                break

        if len(rows) != self.num_channels - 1:
            return None

        q_shape = np.zeros((1, self.num_channels - 1, self.L), dtype=np.int64)
        q_mu = np.zeros((1, self.num_channels - 1), dtype=np.int64)
        q_sigma = np.zeros((1, self.num_channels - 1), dtype=np.int64)
        for i, row in enumerate(rows):
            shape, mu, sigma = row[:3], row[3], row[4]
            # The LLM is free-generating integers; anything out of the tokenizer's actual
            # valid range (codebook size / bin count) must be rejected here, not passed
            # through -- an out-of-range index crashes CUDA with an unrecoverable
            # device-side assert deep inside the tokenizer's dequantize, not a catchable
            # Python exception.
            if not all(_in_range(c, 0, self.num_embeddings - 1) for c in shape):
                return None
            if not (_in_range(mu, 0, self.num_bins - 1) and _in_range(sigma, 0, self.num_bins - 1)):
                return None
            q_shape[0, i] = shape
            q_mu[0, i] = mu
            q_sigma[0, i] = sigma
        return {"q_shape": q_shape, "q_mu": q_mu, "q_sigma": q_sigma}

    def _fallback(self, context_norm: torch.Tensor, p_level_int: int) -> np.ndarray:
        """Last-observation carry-forward in physical units, (forecast_horizon, 12)."""
        last = self._unnorm_context(context_norm)[-1].copy()
        last[-1] = float(p_level_int)
        return np.tile(last, (self.forecast_horizon, 1)).astype(np.float32)

    # ── public API ────────────────────────────────────────────────────────────

    def _generate_samples(self, context_norm: torch.Tensor, p_level_int: int, n: int) -> np.ndarray:
        """Returns (n_valid, forecast_horizon, 12) physical-unit samples (n_valid <= n;
        falls back to a single last-observation-carry-forward sample if none parse)."""
        ctx_physical = self._unnorm_context(context_norm)  # (T, 12)
        ctx_tokens = self._tokenize_patch(ctx_physical[:, :-1])  # exclude Pump Level column

        few_shot_msgs = None
        if self.k_shot > 0 and self._train_last_norm is not None:
            query_last_norm = context_norm[-1].cpu().numpy()
            examples = self._retrieve_examples(query_last_norm, self.k_shot)
            few_shot_msgs = []
            for ctx_ex, pl_ex, fc_ex in examples:
                ex_ctx_tokens = self._tokenize_patch(ctx_ex[:, :-1])
                ex_fc_tokens = self._tokenize_patch(fc_ex[:, :-1])
                few_shot_msgs.append((
                    self._build_user_prompt(ex_ctx_tokens, pl_ex),
                    self._format_example_assistant(ex_fc_tokens),
                ))

        user_prompt = self._build_user_prompt(ctx_tokens, p_level_int)
        raws = self._run_llm(user_prompt, few_shot_messages=few_shot_msgs, n=n)

        samples = []
        for raw in raws:
            pred_tokens = self._parse_response(raw)
            if pred_tokens is None:
                continue
            try:
                pred_phys = self._detokenize_channels(
                    pred_tokens["q_shape"][0], pred_tokens["q_mu"][0], pred_tokens["q_sigma"][0]
                )  # (forecast_horizon, N-1)
            except Exception as e:
                # _parse_response already range-checks every index against the tokenizer's
                # actual codebook/bin sizes (the one class of error that corrupts the CUDA
                # context if it reaches a kernel uncaught); this is a last-resort net for
                # anything else unexpected, so one bad sample can't take down the whole run.
                print(f"[GormpoFewShot] Unexpected decode failure, skipping sample: {e}")
                continue
            if self._phys_low is not None:
                # Safety net: the LLM occasionally predicts a shape-code/scale-bin
                # combination that decodes to something physically impossible (observed:
                # SYSTOLIC crashing to ~-450 mmHg), since nothing in the tokenizer's
                # decoder inherently bounds its output range. This doesn't fix why that
                # happens, it just stops it from propagating into the reported
                # prediction -- clipped to the same training-data percentile bounds
                # (compute_variable_ylims) used for this model's plotting axes too.
                pred_phys = np.clip(pred_phys, self._phys_low, self._phys_high)
            pl_col = np.full((self.forecast_horizon, 1), float(p_level_int), dtype=np.float32)
            samples.append(np.concatenate([pred_phys, pl_col], axis=1))  # (forecast_horizon, 12)

        if not samples:
            print(f"[GormpoFewShot] All {n} responses unparseable; using fallback.")
            samples = [self._fallback(context_norm, p_level_int)]

        return np.stack(samples)

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
        arr = self._generate_samples(context, self._resolve_pl(pl), num_samples)  # (n_valid, T, 12) physical

        m = self.mean[self.columns].detach().cpu().numpy()
        s = self.std[self.columns].detach().cpu().numpy()
        norm = torch.tensor((arr - m) / s, dtype=torch.float32).unsqueeze(1)  # (n_valid, 1, T, 12)

        if norm.shape[0] < num_samples:  # pad by repeating last valid sample (fallback-only case)
            pad = norm[-1:].expand(num_samples - norm.shape[0], -1, -1, -1)
            norm = torch.cat([norm, pad], dim=0)
        return norm
