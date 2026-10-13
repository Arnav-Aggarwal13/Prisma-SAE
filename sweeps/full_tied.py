"""Full 127M-token runs (no early stop) of the round 1+2 winner: TopK, tied init (W_enc = W_dec^T), batch 16384 with 300 warmup steps.

Base (entry 0): k=64, 16x, lr 1e-3. The other entries each change one thing from it.
dead_feature_window=610 steps (~10M tokens at batch 16384, as saev's) so W&B's sparsity/dead_features is comparable; TopK does not use it for training.
Run as four jobs: SWEEP_INDEX=0, 1, 2, 3 (TOTAL_TOKENS=127000000, no STOP_AFTER_TOKENS).
"""

BEST = {"initialization_method": "encoder_transpose_decoder", "train_batch_size": 16384, "lr_warm_up_steps": 300, "dead_feature_window": 610}


def make_cfgs() -> list[dict]:
    return [
        {**BEST},  # 0 base: k=64, 16x, lr 1e-3
        {**BEST, "expansion_factor": 32},  # 1 53k latents
        {**BEST, "k": 128},  # 2 compare at its own l0
        {**BEST, "lr": 5e-4},  # 3 lower lr (3e-3 hurt in round 2)
    ]
