"""
python diagnose_qwen_fallback.py

Prints the RAW generated text for a handful of Qwen zero-shot episodes, to see
exactly why so many fail to parse (~14-15% fallback rate) -- rather than guessing.
"""
import numpy as np
import torch

from gormpo_world_model import GormpoFewShotWorldModel

DATA_PATH = "/public/gormpo/10min_1hr_all_data.pkl"
TOKENIZER_PATH = "forecasting/saved_models/gormpo_tokenizer_mcs_scratch_long2000_retrained/checkpoints/final_tokenizer.pth"
DEVICE = "cuda:4"
N_EPISODES = 20

model = GormpoFewShotWorldModel(
    tokenizer_path=TOKENIZER_PATH, model_id="Qwen/Qwen2.5-7B-Instruct",
    forecast_horizon=6, k_shot=0, num_samples=3, device=DEVICE,
)
model.load_data(DATA_PATH)

n_fail = 0
for i in range(N_EPISODES):
    x, pl, y = model.data_test[i]
    context = torch.as_tensor(np.asarray(x)).float()
    pl_seq = model.unnorm_pl(torch.as_tensor(np.asarray(pl)).float()).round().long().numpy()

    ctx_physical = model._unnorm_context(context)
    ctx_tokens = model._tokenize_patch(ctx_physical[:, :-1])
    user_prompt = model._build_user_prompt(ctx_tokens, pl_seq)
    raws = model._run_llm(user_prompt, few_shot_messages=None, n=3)

    all_bad = True
    for j, raw in enumerate(raws):
        parsed = model._parse_response(raw)
        ok = parsed is not None
        all_bad = all_bad and not ok
        if not ok:
            print(f"=== episode {i} sample {j}: FAILED TO PARSE ===")
            print(repr(raw))
            print()
    if all_bad:
        n_fail += 1

print(f"\n{n_fail}/{N_EPISODES} episodes had all 3 samples fail")
