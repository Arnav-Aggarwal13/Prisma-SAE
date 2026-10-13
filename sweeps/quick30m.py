"""Quick 30M-token pass over Prisma options, aimed at the 90% dead-latent problem. Base = TopK k=64, 16x, lr 1e-3, batch 4096.

Run as two jobs: SWEEP_INDEX=0+1+2+3 and SWEEP_INDEX=4+5+6+7, with TOTAL_TOKENS=30000000.
Compare normalized_mse at matched l0 (ReLU+L1 runs land at whatever l0 their L1 gives), and dead_pct.
"""

RELU = {"activation_fn_str": "relu", "activation_fn_kwargs": {}}


def make_cfgs() -> list[dict]:
    return [
        {"initialization_method": "encoder_transpose_decoder"},  # 0 TopK, W_enc = W_dec^T at init
        {**RELU, "use_ghost_grads": True, "l1_coefficient": 2e-4},  # 1 ReLU+L1+ghost grads (Prisma's old recipe)
        {**RELU, "use_ghost_grads": True, "l1_coefficient": 8e-5},  # 2 same, weaker L1 (higher l0)
        {**RELU, "use_ghost_grads": True, "l1_coefficient": 8e-4},  # 3 same, stronger L1 (lower l0)
        {"lr": 3e-3},  # 4 TopK, higher lr
        {"train_batch_size": 16384, "lr_warm_up_steps": 300},  # 5 TopK, saev-sized batch (fewer steps, shorter warmup)
        {**RELU, "architecture": "gated", "l1_coefficient": 2e-4},  # 6 gated SAE
        {"normalize_activations": "layer_norm"},  # 7 TopK with Prisma's default per-token input normalization
    ]
