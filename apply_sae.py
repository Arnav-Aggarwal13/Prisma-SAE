"""Apply a Prisma SAE to MindEye2's cached prior_out tokens, token by token, and save the reconstruction so it can be decoded into images and evaluated. Run from the Prisma-SAE repo root:

  uv run python apply_sae.py --ckpt CKPT.pt                    # cond sae_pre: SAE(base_all_prior_out)
  uv run python apply_sae.py --ckpt CKPT.pt --oracle           # cond sae_pre_oracle: SAE(base_clip_target), the true image embeddings

Input (read only): (1000, 256, 1664) fp16, row i = COCO id base_image_idx.npy[i]. It is applied raw (no normalization or rescaling), as the SAE was trained on raw bigG ln_post patch tokens. The cached tensor is never resampled.
The SAE acts on each token independently (TopK per token): the tensor is flattened to (256000, 1664), run in contiguous in-order slices, and reshaped back. No shuffling or sorting, so row i of the output comes from row i of the input.

Output: <out-root>/<cond><wandb run id>/ with <cond>_all_prior_out.pt (fp16, same shape and order), <cond>_image_idx.npy (copy of base_image_idx.npy) and <cond>_sae_record.json. Nothing is written into base/.
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import prisma_bigg  # noqa: E402,F401  (checkpoints pickle HookedSAEConfig)
from analyze_sae import load_sae  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent
OUT_ROOT = pathlib.Path("~/orcd/scratch/mindeye/outputs").expanduser()
BASE = OUT_ROOT / "base"
N_TOK, D = 256, 1664


def git_info() -> dict:
    run = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()
    return dict(git_commit=run("rev-parse", "HEAD"), git_dirty=bool(run("status", "--porcelain", "--untracked-files=no")))


def wandb_run(ckpt: pathlib.Path) -> dict:
    """W&B run of the SAE. The checkpoint folder is '<8 hex>-<project>', which is also the run's display name."""
    import wandb

    name = ckpt.parent.name
    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{api.default_entity}/{name.split('-', 1)[1]}", filters={"display_name": name}))
    assert len(runs) == 1, f"expected 1 W&B run named {name}, found {len(runs)}; pass --wandb-run-id"
    return dict(wandb_run_id=runs[0].id, wandb_run_name=name, wandb_url=runs[0].url)


def train_eval_row(ckpt: pathlib.Path) -> dict | None:
    """The val eval row train_prisma.py wrote for this checkpoint, if results.jsonl is next to the run folder."""
    res = ckpt.parent.parent / "results.jsonl"
    for line in res.read_text().splitlines() if res.exists() else []:
        row = json.loads(line)
        if row["checkpoint_dir"].rstrip("/").endswith(ckpt.parent.name):
            return {k: row[k] for k in ("normalized_mse", "l0", "dead_pct", "dense_pct", "n_tokens")}
    return None


def load_tokens(path: pathlib.Path, n_images: int | None) -> torch.Tensor:
    x = torch.load(path, map_location="cpu")
    assert isinstance(x, torch.Tensor) and x.dtype == torch.float16, f"{path.name}: expected an fp16 tensor"
    assert x.ndim == 3 and x.shape[1:] == (N_TOK, D), f"{path.name}: shape {tuple(x.shape)}"
    assert torch.isfinite(x).all(), f"{path.name} has NaN/inf"
    return x[:n_images].contiguous() if n_images else x


@torch.no_grad()
def apply_sae(sae, flat16: torch.Tensor, batch_size: int, desc: str | None = None) -> tuple[torch.Tensor, float]:
    """flat16: (N, d) fp16 on the SAE's device. Contiguous in-order slices, fp32 compute, fp16 out. Returns (out16, mean L0)."""
    out, n_active = torch.empty_like(flat16), 0
    for s in tqdm(range(0, len(flat16), batch_size), desc=desc, disable=desc is None, mininterval=10):
        _, acts = sae.encode(flat16[s : s + batch_size].float())
        out[s : s + batch_size] = sae.decode(acts).half()
        n_active += (acts > 0).sum().item()
    return out, n_active / len(flat16)


@torch.no_grad()
def check_same_as_forward(sae, flat16: torch.Tensor, out16: torch.Tensor, n: int = 1024):
    """encode+decode must be the SAE's own forward pass."""
    ref = sae(flat16[:n].float())[0].half()
    assert torch.allclose(ref.float(), out16[:n].float(), rtol=0, atol=1e-2), "encode+decode differs from sae.forward"


@torch.no_grad()
def check_batch_invariance(sae, flat16: torch.Tensor, out16: torch.Tensor, n: int = 16384, bs: int = 1000) -> dict:
    """Re-run the first n tokens with a different batch size. Rows must not depend on their batch-mates (fp rounding aside)."""
    n = min(n, len(flat16))
    alt, _ = apply_sae(sae, flat16[:n], bs)
    d = (alt.float() - out16[:n].float()).norm(dim=1) / out16[:n].float().norm(dim=1)
    res = dict(n_tokens=n, batch_sizes=[bs], rel_l2_overall=((alt.float() - out16[:n].float()).norm() / out16[:n].float().norm()).item(), max_row_rel_diff=d.max().item(), n_rows_over_1e3=int((d > 1e-3).sum()))
    assert res["rel_l2_overall"] < 1e-2, f"output rows depend on the batch size: {res}"
    return res


@torch.no_grad()
def order_check(x16: torch.Tensor, out16: torch.Tensor) -> dict:
    """Flat per-image cosine: SAE(x)[i] must be closest to x[i] among all x[j] (top-1 retrieval near 100%)."""
    n = len(x16)
    X = F.normalize(x16.reshape(n, -1).float(), dim=1)
    O = F.normalize(out16.reshape(n, -1).float(), dim=1)
    cos = O @ X.T
    eye = torch.eye(n, dtype=torch.bool, device=cos.device)
    diag, best_other = cos.diag(), cos.masked_fill(eye, -2).amax(1)
    res = dict(top1_retrieval=(cos.argmax(1) == torch.arange(n, device=cos.device)).float().mean().item(), mean_diag_cos=diag.mean().item(), mean_offdiag_cos=((cos.sum(1) - diag) / (n - 1)).mean().item(), min_margin_diag_minus_best_other=(diag - best_other).min().item())
    assert res["top1_retrieval"] >= 0.99, f"row order check failed: {res}"
    return res


@torch.no_grad()
def recon_stats(x16: torch.Tensor, out16: torch.Tensor, chunk: int = 16384) -> dict:
    """Reconstruction stats of the SAVED fp16 output against the fp16 input, all tokens, fp64 sums. nmse = SSE(x - x_hat) / SSE(x - mean(x)), as in the SAE's val eval."""
    n, dev = len(x16), x16.device
    sum_x, sum_x2 = torch.zeros(x16.shape[1], dtype=torch.float64, device=dev), 0.0
    for s in range(0, n, chunk):
        x = x16[s : s + chunk].float()
        sum_x += x.sum(0, dtype=torch.float64)
        sum_x2 += x.pow(2).sum(dtype=torch.float64).item()
    sst = sum_x2 - sum_x.dot(sum_x).item() / n
    sse, cos, nin, nout = 0.0, [], [], []
    for s in range(0, n, chunk):
        x, y = x16[s : s + chunk].float(), out16[s : s + chunk].float()
        sse += (y - x).pow(2).sum(dtype=torch.float64).item()
        cos.append(F.cosine_similarity(y, x, dim=1))
        nin.append(x.norm(dim=1))
        nout.append(y.norm(dim=1))
    cos, nin, nout = torch.cat(cos), torch.cat(nin), torch.cat(nout)
    return dict(
        nmse=sse / sst,
        token_cosine=dict(mean=cos.mean().item(), min=cos.min().item(), p1=cos.quantile(0.01).item(), median=cos.median().item()),
        token_norm_in=dict(mean=nin.mean().item(), median=nin.median().item()),
        token_norm_out=dict(mean=nout.mean().item(), median=nout.median().item()),
        norm_ratio_out_over_in=dict(of_means=(nout.mean() / nin.mean()).item(), per_token_mean=(nout / nin).mean().item(), per_token_median=(nout / nin).median().item()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Prisma SAE checkpoint .pt (config.json beside it, folder named <8 hex>-<wandb project>)")
    p.add_argument("--cond", help="condition name (letters, digits, _; no 'enhanced'). Default sae_pre, or sae_pre_oracle with --oracle")
    p.add_argument("--oracle", action="store_true", help="apply the SAE to base_clip_target.pt (true image embeddings) instead of base_all_prior_out.pt")
    p.add_argument("--input", help="override the input tensor path")
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--wandb-run-id", help="override the W&B run id lookup")
    p.add_argument("--dir-sep", default="", help="text between <cond> and the run id in the output folder name (spec: none)")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--limit", type=int, help="only the first N images (smoke tests; refuses to write under the real out-root)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    t0, timings = time.time(), {}
    lap = lambda name: timings.__setitem__(name, round(time.time() - t0, 1))
    cond = args.cond or ("sae_pre_oracle" if args.oracle else "sae_pre")
    assert re.fullmatch(r"[A-Za-z0-9_]+", cond) and "enhanced" not in cond.lower(), f"bad condition name {cond!r}"
    ckpt, out_root = pathlib.Path(args.ckpt).expanduser().resolve(), pathlib.Path(args.out_root).expanduser()
    assert args.limit is None or out_root.resolve() != OUT_ROOT.resolve(), "--limit is for smoke tests: pass a different --out-root"
    in_path = pathlib.Path(args.input).expanduser() if args.input else BASE / ("base_clip_target.pt" if args.oracle else "base_all_prior_out.pt")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # exact fp32 matmuls

    wb = dict(wandb_run_id=args.wandb_run_id) if args.wandb_run_id else wandb_run(ckpt)
    out_dir = out_root / f"{cond}{args.dir_sep}{wb['wandb_run_id']}"
    assert BASE.resolve() not in (out_dir.resolve(), *out_dir.resolve().parents), "refusing to write into base/"
    assert not out_dir.exists() or args.overwrite, f"{out_dir} exists (use --overwrite)"
    print(f"cond {cond}, input {in_path.name}, SAE {ckpt.parent.name} (W&B run {wb['wandb_run_id']}) -> {out_dir}", flush=True)

    idx = np.load(BASE / "base_image_idx.npy")
    x3 = load_tokens(in_path, None)
    assert len(idx) == len(x3) and len(np.unique(idx)) == len(idx) and (np.diff(idx) > 0).all(), "base_image_idx.npy must be sorted, unique, one per row"
    if args.limit:
        x3, idx = x3[: args.limit], idx[: args.limit]
    n_img = len(x3)
    in_sha = hashlib.sha256(np.ascontiguousarray(x3.numpy()).view(np.uint8)).hexdigest()
    print(f"input {tuple(x3.shape)} {x3.dtype}, ids {idx[:3].tolist()}..{idx[-1]}, sha256 {in_sha[:12]}", flush=True)
    lap("input loaded")

    sae = load_sae(str(ckpt), args.device)
    cfg = {k: getattr(sae.cfg, k, None) for k in ("architecture", "d_in", "d_sae", "expansion_factor", "activation_fn_str", "activation_fn_kwargs", "initialization_method", "normalize_activations", "hook_point_name", "train_batch_size", "lr", "num_epochs")}
    assert cfg["d_in"] == D and cfg["normalize_activations"] == "none", f"unexpected SAE config {cfg}"
    lap("SAE loaded")

    flat = x3.reshape(-1, D).to(args.device)
    out_flat, l0 = apply_sae(sae, flat, args.batch_size, desc="apply SAE")
    assert torch.isfinite(out_flat).all(), "output has NaN/inf (fp16 overflow?)"
    out3 = out_flat.reshape(n_img, N_TOK, D)
    assert out3.shape == x3.shape and out3.dtype == torch.float16
    lap("SAE applied")

    check_same_as_forward(sae, flat, out_flat)
    inv = check_batch_invariance(sae, flat, out_flat)
    order = order_check(x3.to(args.device), out3)
    stats = recon_stats(flat, out_flat)
    stats["mean_l0"] = l0
    print("order check:", json.dumps(order), "\nbatch invariance:", json.dumps(inv), "\nstats:", json.dumps(stats), flush=True)
    lap("checks done")

    out_dir.mkdir(parents=True, exist_ok=args.overwrite)
    out_pt = out_dir / f"{cond}_all_prior_out.pt"
    torch.save(out3.cpu().contiguous(), str(out_pt) + ".tmp")
    os.replace(str(out_pt) + ".tmp", out_pt)
    assert torch.equal(torch.load(out_pt, map_location="cpu"), out3.cpu()), "saved file does not match"
    idx_path = out_dir / f"{cond}_image_idx.npy"
    if args.limit:
        np.save(idx_path, idx)
    else:
        shutil.copyfile(BASE / "base_image_idx.npy", idx_path)
        assert np.array_equal(np.load(idx_path), idx)
    lap("saved")

    record = dict(
        cond=cond, n_images=n_img, limit=args.limit,
        input=dict(path=str(in_path), sha256=in_sha, shape=list(x3.shape), dtype="float16", row_order="base_image_idx.npy (sorted unique COCO ids)", applied="raw: no normalization or rescaling"),
        sae=dict(checkpoint=str(ckpt), checkpoint_bytes=ckpt.stat().st_size, config=cfg, **wb, train_val_eval=train_eval_row(ckpt)),
        application=dict(batch_size=args.batch_size, order="contiguous in-order slices of the flattened (n*256, 1664) tokens", compute="fp32, TF32 off", output_dtype="float16", device=args.device),
        recon_stats=stats, order_check=order, batch_invariance=inv,
        outputs=[p.name for p in (out_pt, idx_path)], **git_info(), date=datetime.datetime.now().isoformat(timespec="seconds"),
        gpu=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu", slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        versions=dict(torch=torch.__version__, numpy=np.__version__), timings_s=timings,
        peak_gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1) if args.device.startswith("cuda") else None,
    )
    (out_dir / f"{cond}_sae_record.json").write_text(json.dumps(record, indent=1))
    print(f"done in {time.time() - t0:.0f}s. Output in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
