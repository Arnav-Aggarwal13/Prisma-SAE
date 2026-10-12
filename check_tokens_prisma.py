"""Check that Prisma's hook_ln_final[:, 1:] is the target tensor. Must PASS before training. Run from the Prisma-SAE repo root:

  uv run python check_tokens_prisma.py --images TRAIN_THUMBNAILS_DIR [--shards TRAIN_SHARDS_DIR]

Compares, for a few images spread across the ImageFolder:
  1. preprocessing: Prisma's get_clip_val_transforms vs open_clip's own preprocess (pixels)
  2. Prisma hook_ln_final[:, 1:] vs open_clip model.visual(x) output_tokens (ln_post, unprojected), fp32 without TF32
  3. with --shards: saev shard tokens (layer --saev-layer, content tokens) vs Prisma, matched by ImageFolder index. Shards were written with TF32, so this uses a looser tolerance.
"""

import argparse
import pathlib
import sys

import open_clip
import torch
import torchvision
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import prisma_bigg as pb  # noqa: E402


def compare(name: str, got: torch.Tensor, ref: torch.Tensor, rel_tol: float) -> bool:
    assert got.shape == ref.shape, f"{name}: shape {tuple(got.shape)} != {tuple(ref.shape)}"
    got, ref = got.double(), ref.double()
    rel = ((got - ref).norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(got.reshape(-1, got.shape[-1]), ref.reshape(-1, ref.shape[-1]), dim=-1).min().item()
    ok = rel < rel_tol
    print(f"{'PASS' if ok else 'FAIL'} {name}: rel L2 err {rel:.2e} (tol {rel_tol:.0e}), max abs {(got - ref).abs().max().item():.2e}, min token cosine {cos:.6f}")
    return ok


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="ImageFolder root that the shards were written from")
    p.add_argument("--shards", help="saev shards dir to compare against (optional)")
    p.add_argument("--saev-layer", type=int, default=pb.N_BLOCKS)
    p.add_argument("--n", type=int, default=8, help="number of images")
    p.add_argument("--oc-arch", default="ViT-bigG-14")
    p.add_argument("--oc-pretrained", default="laion2b_s39b_b160k", help="open_clip pretrained tag or a local checkpoint path")
    p.add_argument("--prisma-model", default=pb.MODEL_NAME)
    p.add_argument("--model-local-path", help="local open_clip_pytorch_model.bin for Prisma (offline nodes)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from vit_prisma.models.model_loader import load_hooked_model
    from vit_prisma.transforms import get_clip_val_transforms

    oc_model, oc_preprocess = open_clip.create_model_from_pretrained(args.oc_arch, pretrained=args.oc_pretrained)
    visual = oc_model.visual.to(args.device).eval()
    del oc_model
    visual.output_tokens = True
    prisma = load_hooked_model(args.prisma_model, device=args.device, local_path=args.model_local_path).eval()

    ds = torchvision.datasets.ImageFolder(args.images)
    idx = torch.linspace(0, len(ds) - 1, min(args.n, len(ds))).long().tolist()
    pil = [Image.open(ds.samples[i][0]).convert("RGB") for i in idx]
    x_oc = torch.stack([oc_preprocess(im) for im in pil]).to(args.device)
    x_pr = torch.stack([get_clip_val_transforms()(im) for im in pil]).to(args.device)

    ok = compare("preprocessing (pixels)", x_pr, x_oc, rel_tol=1e-5)
    _, ref_tokens = visual(x_oc)
    acts = pb.get_hook_acts(prisma, x_oc, pb.LN_POST, prisma.cfg.n_layers)
    ok &= compare("Prisma hook_ln_final[:, 1:] vs open_clip output_tokens", acts[:, 1:], ref_tokens, rel_tol=1e-4)
    print(f"tokens per image: {acts.shape[1]} (incl. CLS), d = {acts.shape[2]}, mean token norm {acts[:, 1:].norm(dim=-1).mean().item():.2f}")

    if args.shards:
        import saev.data.indexed as indexed

        sds = indexed.Dataset(indexed.Config(shards=pathlib.Path(args.shards), tokens="content", layer=args.saev_layer))
        md = sds.md
        assert md.n_examples == len(ds), f"shards have {md.n_examples} examples but the ImageFolder has {len(ds)}; was the folder changed after sharding?"
        n_tok = md.content_tokens_per_example
        rows = []
        for i in idx:
            items = [sds[i * n_tok + j] for j in range(n_tok)]
            assert all(it["example_idx"] == i for it in items)
            rows.append(torch.stack([it["act"] for it in items]))
        ok &= compare("saev shard tokens vs Prisma", torch.stack(rows).to(args.device), acts[:, 1:], rel_tol=1e-2)

    print("ALL PASS" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
