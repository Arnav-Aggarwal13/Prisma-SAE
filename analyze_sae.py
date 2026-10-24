"""Look inside a trained Prisma SAE using the saev val shards (no ViT compute). Run from the Prisma-SAE repo root:

  uv run python analyze_sae.py --ckpt CKPT.pt --val-shards DIR --images VAL_IMAGE_DIR --out DIR

Stage 1 (encode): pass the val shards through the SAE once, in order. Per latent: firing frequency, mean and max activation, and the top-K (activation, image, patch) over all val tokens.
Stage 2 (render): for groups of latents (random alive, most frequent, rarest alive, strongest, plus --features), show the top distinct images with the latent's activation over the 16x16 patches as a heatmap.
Output in --out: features.npz, summary.json, frequency_hist.png, grids/<group>_<page>.png.
--skip-encode reuses features.npz, so other latents can be rendered with --features without redoing stage 1.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import prisma_bigg as pb  # noqa: E402

GRID, PATCH = 16, 14  # 224 / 14 patches per side


def load_sae(ckpt: str, device: str):
    from vit_prisma.sae.sae import SparseAutoencoder

    # Prisma's loader calls torch.load without map_location; checkpoints saved on a GPU then fail on CPU-only nodes.
    torch_load = torch.load
    torch.load = lambda *a, **k: torch_load(*a, **{"map_location": device, **k})
    try:
        return SparseAutoencoder.load_from_pretrained(ckpt, current_cfg={"_device": device}).to(device).eval()
    finally:
        torch.load = torch_load


@torch.no_grad()
def encode_pass(sae, shards_dpath: pathlib.Path, layer: int, batch_size: int, top_k: int, max_tokens: int | None, device: str) -> dict:
    """One ordered pass over the shards. Keeps, per latent, the top_k (activation, example, patch) over all tokens."""
    import saev.data.ordered as ordered

    loader = ordered.DataLoader(ordered.Config(shards=shards_dpath, tokens="content", layer=layer, batch_size=batch_size))
    n_lat = sae.cfg.d_sae
    fire = torch.zeros(n_lat, dtype=torch.long, device=device)
    total = torch.zeros(n_lat, dtype=torch.float64, device=device)
    top_v = torch.full((top_k, n_lat), -1.0, device=device)
    top_ex = torch.zeros((top_k, n_lat), dtype=torch.long, device=device)
    top_tok = torch.zeros_like(top_ex)
    n_batches = len(loader) if not max_tokens else -(-max_tokens // batch_size)
    n_tok, it = 0, iter(loader)
    try:
        for _ in tqdm(range(n_batches), desc="encode val shards", mininterval=20):
            b = next(it)
            x = b["act"].to(device)
            _, a = sae.encode(x)  # (batch, d_sae), sparse (TopK)
            fire += (a > 0).sum(0)
            total += a.sum(0, dtype=torch.float64)
            v, i = a.topk(min(top_k, a.shape[0]), dim=0)  # best tokens of this batch, per latent
            ex, tok = b["example_idx"].to(device).long()[i], b["token_idx"].to(device).long()[i]
            assert tok.max() < GRID * GRID, f"token_idx {tok.max().item()} is not a content patch index"
            top_v, j = torch.cat([top_v, v]).topk(top_k, dim=0)  # merge with the running best
            top_ex = torch.cat([top_ex, ex]).gather(0, j)
            top_tok = torch.cat([top_tok, tok]).gather(0, j)
            n_tok += x.shape[0]
    finally:
        pb._close(it)
    fire_c, total_c = fire.cpu().numpy(), total.cpu().numpy()
    return dict(
        n_tok=n_tok,
        freq=fire_c / n_tok,
        mean_act=np.where(fire_c > 0, total_c / np.maximum(fire_c, 1), 0.0),
        max_act=top_v[0].cpu().numpy(),
        top_v=top_v.cpu().numpy(),
        top_ex=top_ex.cpu().numpy(),
        top_tok=top_tok.cpu().numpy(),
    )


class ExampleReader:
    """Reads one example's 256 content tokens straight from the shard memmaps (read only)."""

    def __init__(self, shards_dpath: pathlib.Path, layer: int):
        from saev.data import shards

        self.dpath, self.md, self.mm = shards_dpath, shards.Metadata.load(shards_dpath), {}
        self.layer_i = self.md.layers.index(layer)

    def tokens(self, ex: int) -> torch.Tensor:
        shard_i, local = divmod(int(ex), self.md.examples_per_shard)
        if shard_i not in self.mm:
            self.mm[shard_i] = np.memmap(self.dpath / f"acts{shard_i:06}.bin", mode="r", dtype=np.float32, shape=self.md.shard_shape)
        return torch.from_numpy(np.array(self.mm[shard_i][local, self.layer_i, int(self.md.cls_token) :]))


def pick_groups(st: dict, n: int, seed: int, extra: list[int]) -> dict[str, np.ndarray]:
    freq, n_fire = st["freq"], st["freq"] * st["n_tok"]
    alive, enough = np.nonzero(freq > 0)[0], np.nonzero(n_fire >= 20)[0]
    groups = {
        "random": np.random.default_rng(seed).choice(alive, min(n, len(alive)), replace=False),
        "frequent": np.argsort(-freq)[:n],
        "rare": enough[np.argsort(freq[enough])[:n]],  # rarest latents that still fire on >= 20 tokens
        "strongest": np.argsort(-st["max_act"])[:n],
    }
    if extra:
        groups["chosen"] = np.array(extra)
    return groups


@torch.no_grad()
def top_images(f: int, st: dict, reader: ExampleReader, sae, n_img: int, device: str) -> list:
    """Top distinct images for latent f: [(example_idx, peak_act, 16x16 activation map)]."""
    picks, seen = [], set()
    for v, e in zip(st["top_v"][:, f], st["top_ex"][:, f]):
        if v <= 0:
            break
        if int(e) not in seen:
            seen.add(int(e))
            picks.append((int(e), float(v)))
        if len(picks) == n_img:
            break
    if not picks:
        return []
    _, a = sae.encode(torch.cat([reader.tokens(e) for e, _ in picks]).to(device))
    heat = a[:, f].reshape(len(picks), GRID, GRID).cpu().numpy()
    return [(e, v, heat[i]) for i, (e, v) in enumerate(picks)]


def render(groups: dict, st: dict, reader: ExampleReader, sae, ds, out: pathlib.Path, n_img: int, page_size: int, heat_floor: float, device: str):
    """One PNG per page of `page_size` latents (a row each, `n_img` images across). Heat is drawn only on patches with activation >= heat_floor * the latent's max; the rest of the image is left clear."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torchvision.transforms as T

    show = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)])
    (out / "grids").mkdir(parents=True, exist_ok=True)
    for name, feats in groups.items():
        for p0 in range(0, len(feats), page_size):
            page = feats[p0 : p0 + page_size]
            fig, axes = plt.subplots(len(page), n_img, figsize=(1.9 * n_img, 2.15 * len(page)), squeeze=False)
            for r, f in tqdm(list(enumerate(page)), desc=f"render {name} {p0 // page_size}", mininterval=20):
                vmax = float(st["max_act"][f])
                items = top_images(int(f), st, reader, sae, n_img, device)
                for c in range(n_img):
                    axes[r, c].axis("off")
                    if c < len(items):
                        e, v, heat = items[c]
                        axes[r, c].imshow(show(ds[e][0]))
                        hot = np.ma.masked_less(np.kron(heat, np.ones((PATCH, PATCH))), heat_floor * vmax)  # masked = transparent, so weak patches show the plain image
                        axes[r, c].imshow(hot, alpha=0.55, cmap="jet", vmin=0, vmax=vmax)
                axes[r, 0].set_title(f"#{int(f)}  freq {st['freq'][f]:.1e}  max {vmax:.1f}  ({len(items)} imgs)", fontsize=9, loc="left")
            fig.tight_layout()
            fig.savefig(out / "grids" / f"{name}_{p0 // page_size}.png", dpi=90)
            plt.close(fig)


def summarize(st: dict, out: pathlib.Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freq = st["freq"]
    s = dict(
        n_tokens=int(st["n_tok"]), n_latents=int(len(freq)), n_alive=int((freq > 0).sum()), n_dead=int((freq == 0).sum()),
        n_dense=int((freq > 0.01).sum()), mean_l0=float(freq.sum()), median_freq_alive=float(np.median(freq[freq > 0])),
        n_fire_ge20_tokens=int((freq * st["n_tok"] >= 20).sum()),
    )
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    plt.figure(figsize=(6, 3.5))
    plt.hist(np.log10(freq[freq > 0]), bins=60)
    plt.xlabel("log10 firing frequency (alive latents)")
    plt.ylabel("latents")
    plt.title(f"{s['n_alive']} alive / {s['n_latents']}, mean L0 {s['mean_l0']:.1f}")
    plt.tight_layout()
    plt.savefig(out / "frequency_hist.png", dpi=110)
    plt.close()
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Prisma SAE checkpoint .pt (config.json beside it)")
    p.add_argument("--val-shards", required=True, help="saev shards dir to encode")
    p.add_argument("--images", required=True, help="ImageFolder root the shards were written from (the val folder)")
    p.add_argument("--out", required=True)
    p.add_argument("--saev-layer", type=int, default=pb.N_BLOCKS)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--top-k", type=int, default=64, help="top tokens kept per latent (several can be the same image)")
    p.add_argument("--n-images", type=int, default=8, help="distinct images shown per latent")
    p.add_argument("--n-per-group", type=int, default=20)
    p.add_argument("--page-size", type=int, default=5, help="latents per PNG (20 per group = 4 PNGs)")
    p.add_argument("--heat-floor", type=float, default=0.1, help="draw heat only on patches with activation >= this fraction of the latent's max; weaker patches show the plain image")
    p.add_argument("--features", default="", help="comma-separated latent ids to render as an extra 'chosen' group")
    p.add_argument("--max-tokens", type=int, help="stage 1 on only the first N tokens (smoke tests)")
    p.add_argument("--skip-encode", action="store_true", help="reuse OUT/features.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    import torchvision

    out, shards_dpath = pathlib.Path(args.out), pathlib.Path(args.val_shards)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sae = load_sae(args.ckpt, args.device)
    print(f"loaded SAE: d_in {sae.cfg.d_in}, d_sae {sae.cfg.d_sae}, {sae.cfg.activation_fn_str} {sae.cfg.activation_fn_kwargs} ({time.time() - t0:.0f}s)", flush=True)
    reader = ExampleReader(shards_dpath, args.saev_layer)
    ds = torchvision.datasets.ImageFolder(args.images)
    assert sae.cfg.d_in == reader.md.d_model, f"SAE d_in {sae.cfg.d_in} != shards d_model {reader.md.d_model}"
    assert len(ds) == reader.md.n_examples, f"{len(ds)} images but shards have {reader.md.n_examples} examples; wrong --images folder?"

    if args.skip_encode:
        st = dict(np.load(out / "features.npz"))
        st["n_tok"] = int(st["n_tok"])
    else:
        print(f"stage 1: encoding {args.max_tokens or reader.md.n_examples * reader.md.content_tokens_per_example} tokens", flush=True)
        st = encode_pass(sae, shards_dpath, args.saev_layer, args.batch_size, args.top_k, args.max_tokens, args.device)
        np.savez(out / "features.npz", **st)
        print(f"stage 1 done ({time.time() - t0:.0f}s total)", flush=True)

    s = summarize(st, out)
    print("summary:", json.dumps(s), flush=True)
    extra = [int(i) for i in args.features.split(",") if i]
    groups = pick_groups(st, args.n_per_group, args.seed, extra)
    print("stage 2: rendering", {k: len(v) for k, v in groups.items()}, flush=True)
    render(groups, st, reader, sae, ds, out, args.n_images, args.page_size, args.heat_floor, args.device)
    print(f"done ({time.time() - t0:.0f}s total). Output in {out}", flush=True)


if __name__ == "__main__":
    main()
