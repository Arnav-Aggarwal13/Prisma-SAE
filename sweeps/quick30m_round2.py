"""Round 2 of the quick pass (same protocol as quick30m.py: TOTAL_TOKENS=127000000, STOP_AFTER_TOKENS=30000000, so lr follows the full-run schedule).

Round 1 found: tied init (W_enc = W_dec^T) takes TopK k=64 from 92.8% dead / nmse 0.382 to 2% dead / nmse 0.315, and gated at L1 2e-4 was dense (l0 ~11k).

Entries 0-3: gated SAE, L1 large enough to be sparse. Gated loss = small normalized mse + l1 * sum(gate * |W_dec row|) + a summed (not normalized) squared-error aux loss of order ||x||^2 ~ 2000 per token, so the useful L1 is order 1-10, not 1e-3. Look for l0 near 64 and compare nmse there.
Entries 4-7: TopK + tied init combined with one other change each (all k=64 except entry 6), now that latents are alive.
Run as four jobs: SWEEP_INDEX=0+1, 2+3, 4+5, 6+7.
"""

TIED = {"initialization_method": "encoder_transpose_decoder"}
GATED = {"activation_fn_str": "relu", "activation_fn_kwargs": {}, "architecture": "gated"}


def make_cfgs() -> list[dict]:
    return [
        {**GATED, "l1_coefficient": 0.5},  # 0 gated
        {**GATED, "l1_coefficient": 1.5},  # 1
        {**GATED, "l1_coefficient": 5.0},  # 2
        {**GATED, "l1_coefficient": 15.0},  # 3
        {**TIED, "lr": 3e-3},  # 4 tied + higher lr
        {**TIED, "expansion_factor": 32},  # 5 tied + 32x (53k latents)
        {**TIED, "k": 128},  # 6 tied + k=128 (compare at its own l0)
        {**TIED, "train_batch_size": 16384, "lr_warm_up_steps": 300},  # 7 tied + saev-sized batch
    ]
