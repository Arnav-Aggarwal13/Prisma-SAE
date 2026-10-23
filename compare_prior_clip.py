"""Why does the SAE reconstruct MindEye2's predicted tokens (prior_out) with lower nmse than the true tokens (clip_target)?

Profiles each input and the SAE's reconstruction of it (all on the GPU, fp32), so the two can be compared on more than nmse:
- scale: variance about the mean (SST per element), absolute error per element, token norms and their tails
- how the error is spread: relative error per token, share of the error in the worst 1% of tokens
- structure of the input: how much of its energy is the mean token, how much of its variance is between images, how smooth it is across neighbouring patches
- how far prior_out is from clip_target (cosine, relative L2), raw and about the clip_target mean
Needs the four tensors from apply_sae.py (cond sae_pre and sae_pre_oracle) and base/. Run from the repo root, on a GPU node (slurm/compare_prior_clip.sbatch).
"""

import argparse
import json
import pathlib
import time

import torch
import torch.nn.functional as F

M = pathlib.Path("~/orcd/scratch/mindeye/outputs").expanduser()


def q(t: torch.Tensor, qs=(0.001, 0.01, 0.5, 0.9, 0.99, 0.999)) -> dict:
    return {f"p{p * 100:g}": round(t.flatten().quantile(p).item(), 4) for p in qs}


@torch.no_grad()
def profile(x3: torch.Tensor, y3: torch.Tensor, mean: torch.Tensor) -> dict:
    """x3: input, y3: SAE reconstruction, both (n_img, 256, d) fp32 on the GPU; mean: the input's token mean (d,)."""
    n_img, _, d = x3.shape
    n_el = x3.numel()
    err = y3 - x3
    sst, sse = (x3 - mean).pow(2).sum(dtype=torch.float64).item(), err.pow(2).sum(dtype=torch.float64).item()
    norm = x3.norm(dim=-1)  # (n_img, 256)
    rel = err.norm(dim=-1) / norm
    tok_sse = err.pow(2).sum(-1).flatten()
    worst = tok_sse.topk(len(tok_sse) // 100).values.sum().item() / tok_sse.sum().item()
    cos, cos_c = F.cosine_similarity(y3, x3, dim=-1), F.cosine_similarity(y3 - mean, x3 - mean, dim=-1)
    grid, grid_c = x3.reshape(n_img, 16, 16, d), (x3 - mean).reshape(n_img, 16, 16, d)
    adj = lambda g: torch.cat([F.cosine_similarity(g[:, :, 1:], g[:, :, :-1], dim=-1).flatten(), F.cosine_similarity(g[:, 1:], g[:, :-1], dim=-1).flatten()]).mean().item()
    return dict(
        nmse=sse / sst,
        var_per_element=sst / n_el,
        sse_per_element=sse / n_el,
        token_norm=dict(mean=norm.mean().item(), std=norm.std().item(), **q(norm)),
        frac_tokens_norm_over_1p3_median=(norm > 1.3 * norm.median()).float().mean().item(),
        rel_err_per_token=dict(mean=rel.mean().item(), **q(rel)),
        share_of_sse_in_worst_1pct_tokens=worst,
        cosine_to_input=dict(mean=cos.mean().item(), **q(cos)),
        cosine_to_input_about_mean=dict(mean=cos_c.mean().item(), **q(cos_c)),
        mean_token_energy_share=(mean.pow(2).sum() / norm.pow(2).mean()).item(),
        between_image_variance_share=((x3.mean(1) - mean).pow(2).sum().item() * 256 / sst),
        neighbour_patch_cosine=adj(grid),
        neighbour_patch_cosine_about_mean=adj(grid_c),
    )


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prior-sae", required=True, help="folder with sae_pre_all_prior_out.pt")
    p.add_argument("--clip-sae", required=True, help="folder with sae_pre_oracle_all_prior_out.pt")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"
    t0, dev = time.time(), "cuda"
    load = lambda path: torch.load(path, map_location="cpu").to(dev).float()
    prior, clip = load(M / "base/base_all_prior_out.pt"), load(M / "base/base_clip_target.pt")
    prior_hat = load(pathlib.Path(args.prior_sae).expanduser() / "sae_pre_all_prior_out.pt")
    clip_hat = load(pathlib.Path(args.clip_sae).expanduser() / "sae_pre_oracle_all_prior_out.pt")
    print(f"loaded ({time.time() - t0:.0f}s)", flush=True)
    mean_p, mean_c = prior.mean((0, 1)), clip.mean((0, 1))
    res = dict(prior_out=profile(prior, prior_hat, mean_p), clip_target=profile(clip, clip_hat, mean_c))
    # the same reconstruction errors, but both normalised by the SAME variance (the true tokens'), so nmse is comparable across inputs
    sst_c = (clip - mean_c).pow(2).sum(dtype=torch.float64).item()
    res["nmse_with_clip_target_variance"] = dict(prior_out=(prior_hat - prior).pow(2).sum(dtype=torch.float64).item() / sst_c, clip_target=(clip_hat - clip).pow(2).sum(dtype=torch.float64).item() / sst_c)
    d = prior - clip
    res["prior_vs_clip"] = dict(
        rel_l2_distance=dict(mean=(d.norm(dim=-1) / clip.norm(dim=-1)).mean().item()),
        cosine=dict(mean=F.cosine_similarity(prior, clip, dim=-1).mean().item()),
        cosine_about_clip_mean=dict(mean=F.cosine_similarity(prior - mean_c, clip - mean_c, dim=-1).mean().item()),
        var_ratio_prior_over_clip=((prior - mean_p).pow(2).sum(dtype=torch.float64) / (clip - mean_c).pow(2).sum(dtype=torch.float64)).item(),
        sae_output_cosine_to_other_input=dict(prior_hat_vs_clip=F.cosine_similarity(prior_hat, clip, dim=-1).mean().item(), clip_hat_vs_prior=F.cosine_similarity(clip_hat, prior, dim=-1).mean().item()),
    )
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    print(f"done ({time.time() - t0:.0f}s). Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
