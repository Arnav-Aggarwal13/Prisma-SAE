"""Quick pass over Prisma options, aimed at the 90% dead-latent problem. Base = TopK k=64, 16x, lr 1e-3, batch 4096.

Each run has the lr schedule of the full 127M-token run (TOTAL_TOKENS=127000000) but stops at 30M (STOP_AFTER_TOKENS=30000000), so lr is still ~0.87x peak at the stop and runs are comparable to the first sweep's curves at the same step. Entry 0 is the base config under this protocol.
Run as three jobs: SWEEP_INDEX=0+1+2, 3+4+5, 6+7+8.
Compare normalized_mse at matched l0 (ReLU+L1 runs land at whatever l0 their L1 gives), and dead_pct.
"""

RELU = {"activation_fn_str": "relu", "activation_fn_kwargs": {}}


def make_cfgs() -> list[dict]:
    return [
        {},  # 0 base: TopK k=64, 16x, lr 1e-3
        {"initialization_method": "encoder_transpose_decoder"},  # 1 TopK, W_enc = W_dec^T at init
        {**RELU, "use_ghost_grads": True, "l1_coefficient": 2e-4},  # 2 ReLU+L1+ghost grads (Prisma's old recipe)
        {**RELU, "use_ghost_grads": True, "l1_coefficient": 8e-5},  # 3 same, weaker L1 (higher l0)
        {**RELU, "use_ghost_grads": True, "l1_coefficient": 8e-4},  # 4 same, stronger L1 (lower l0)
        {"lr": 3e-3},  # 5 TopK, higher lr
        {"train_batch_size": 16384, "lr_warm_up_steps": 300},  # 6 TopK, saev-sized batch (fewer steps, shorter warmup)
        {**RELU, "architecture": "gated", "l1_coefficient": 2e-4},  # 7 gated SAE
        {"normalize_activations": "layer_norm"},  # 8 TopK with Prisma's default per-token input normalization
    ]
