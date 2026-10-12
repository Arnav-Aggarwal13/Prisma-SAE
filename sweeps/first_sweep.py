"""First Prisma sweep for bigG ln_post tokens: lr, then k, then expansion. Use with train_prisma.py --sweep."""


def make_cfgs() -> list[dict]:
    lrs = [{"lr": lr} for lr in (3e-4, 1e-3)]
    ks = [{"k": k} for k in (32, 128)]  # k=64 is covered by the lr runs; set "lr" here to the winner if it is not 1e-3
    expansions = [{"expansion_factor": 32}]
    return lrs + ks + expansions
