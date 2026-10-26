"""Cosine similarity between each token and its SAE reconstruction on the whole Open Images val set (all val shards, ~5.1M tokens), uploaded to each SAE's own W&B run. Run from the repo root:

  uv run python eval_val_cosine.py --ckpt CKPT1.pt CKPT2.pt ...      # compute (one pass over the shards for all SAEs) and upload
  uv run python eval_val_cosine.py --ckpt ... --upload-only           # upload the saved results again (e.g. from a login node)

Per token x and its reconstruction y = decode(encode(x)):
  uncentered: cos(y, x)
  centered:   cos(y - m, x - m), m = mean of the val-shard tokens (so the dataset's common direction is removed from both)
Each is summarised over all tokens (mean, median, 1st percentile, min). Also logged: val-set normalized_mse (SSE(x - y) / SSE(x - m)) as a cross-check against the eval in results.jsonl.
Pass 1 computes m (cached in --out-dir/val_token_mean.pt); pass 2 runs every SAE on each batch, in order. Results are saved to --out-dir/<run name>.json first, then uploaded to the run's W&B summary as val_full/*.
"""

import argparse
import json
import pathlib
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import prisma_bigg as pb  # noqa: E402
from analyze_sae import load_sae  # noqa: E402
from apply_sae import wandb_run  # noqa: E402


def batches(shards_dpath: pathlib.Path, layer: int, batch_size: int, max_tokens: int | None):
    """In-order val tokens, (batch, d) fp32 on the GPU."""
    import saev.data.ordered as ordered

    loader = ordered.DataLoader(ordered.Config(shards=shards_dpath, tokens="content", layer=layer, batch_size=batch_size))
    n = len(loader) if not max_tokens else -(-max_tokens // batch_size)
    it = iter(loader)
    try:
        for _ in range(n):
            yield next(it)["act"].cuda().float()
    finally:
        pb._close(it)


@torch.no_grad()
def token_mean(shards_dpath, layer, batch_size, max_tokens) -> tuple[torch.Tensor, int]:
    total, n = None, 0
    for x in tqdm(batches(shards_dpath, layer, batch_size, max_tokens), desc="pass 1: val token mean", mininterval=20):
        s = x.sum(0, dtype=torch.float64)
        total = s if total is None else total + s
        n += len(x)
    return (total / n).float(), n


def stats(c: torch.Tensor) -> dict:
    return dict(mean=c.mean().item(), median=c.median().item(), p1=c.quantile(0.01).item(), min=c.min().item())


@torch.no_grad()
def evaluate(models: dict, shards_dpath, layer, batch_size, max_tokens, m: torch.Tensor) -> dict:
    """models: {name: sae}. One pass over the shards; every SAE sees each batch."""
    acc = {n: dict(cu=[], cc=[], sse=torch.zeros((), dtype=torch.float64, device="cuda")) for n in models}
    sst, n_tok = torch.zeros((), dtype=torch.float64, device="cuda"), 0
    for x in tqdm(batches(shards_dpath, layer, batch_size, max_tokens), desc="pass 2: SAE reconstructions", mininterval=20):
        xc = x - m
        sst += xc.pow(2).sum(dtype=torch.float64)
        n_tok += len(x)
        for n, sae in models.items():
            y = sae.decode(sae.encode(x)[1])
            acc[n]["cu"].append(F.cosine_similarity(y, x, dim=1))
            acc[n]["cc"].append(F.cosine_similarity(y - m, xc, dim=1))
            acc[n]["sse"] += (y - x).pow(2).sum(dtype=torch.float64)
    return {
        n: {"n_tokens": n_tok, "cosine_uncentered": stats(torch.cat(a["cu"])), "cosine_centered": stats(torch.cat(a["cc"])), "normalized_mse": (a["sse"] / sst).item()}
        for n, a in acc.items()
    }


def upload(ckpt: pathlib.Path, res: dict):
    """Write the results into the SAE's own (finished) W&B run summary."""
    import wandb

    info = wandb_run(ckpt)
    api = wandb.Api(timeout=60)
    run = wandb.init(project=ckpt.parent.name.split("-", 1)[1], entity=api.default_entity, id=info["wandb_run_id"], resume="must")
    s = {f"val_full/cosine_{kind}_{k}": v for kind in ("uncentered", "centered") for k, v in res[f"cosine_{kind}"].items()}
    s.update({"val_full/normalized_mse": res["normalized_mse"], "val_full/n_tokens": res["n_tokens"], "val_full/centered_about": "mean of the val-shard tokens"})
    run.summary.update(s)
    run.finish()
    print(f"uploaded {len(s)} values to {info['wandb_url']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True, help="Prisma SAE checkpoints (folder named <8 hex>-<wandb project>)")
    p.add_argument("--val-shards", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--saev-layer", type=int, default=pb.N_BLOCKS)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--max-tokens", type=int, help="only the first N tokens (smoke tests); results are then not uploaded")
    p.add_argument("--upload-only", action="store_true", help="upload the saved --out-dir/<run name>.json files, no compute")
    p.add_argument("--no-upload", action="store_true")
    args = p.parse_args()
    ckpts, out_dir, t0 = [pathlib.Path(c).expanduser().resolve() for c in args.ckpt], pathlib.Path(args.out_dir), time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.upload_only:
        assert torch.cuda.is_available(), "needs a usable GPU"
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # exact fp32 matmuls
        shards_dpath = pathlib.Path(args.val_shards)
        mean_path = out_dir / "val_token_mean.pt"
        if args.max_tokens is None and mean_path.exists():
            m = torch.load(mean_path)["mean"].cuda()
            print(f"loaded cached val token mean ({mean_path})", flush=True)
        else:
            m, n = token_mean(shards_dpath, args.saev_layer, args.batch_size, args.max_tokens)
            if args.max_tokens is None:
                torch.save(dict(mean=m.cpu(), n_tokens=n, shards=str(shards_dpath), layer=args.saev_layer), mean_path)
            print(f"val token mean over {n} tokens: norm {m.norm().item():.2f} ({time.time() - t0:.0f}s)", flush=True)
        models = {c.parent.name: load_sae(str(c), "cuda") for c in ckpts}
        assert all(s.cfg.d_in == len(m) for s in models.values()), "SAE d_in != token dim"
        results = evaluate(models, shards_dpath, args.saev_layer, args.batch_size, args.max_tokens, m)
        for n, r in results.items():
            r["ckpt"] = str(next(c for c in ckpts if c.parent.name == n))
            (out_dir / f"{n}.json").write_text(json.dumps(r, indent=1))
            print(n, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "ckpt"}), flush=True)
        print(f"computed ({time.time() - t0:.0f}s)", flush=True)

    if not (args.no_upload or args.max_tokens):
        for c in ckpts:
            upload(c, json.loads((out_dir / f"{c.parent.name}.json").read_text()))
    print(f"done ({time.time() - t0:.0f}s). Results in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
