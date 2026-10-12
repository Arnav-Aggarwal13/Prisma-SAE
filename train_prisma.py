"""Train and evaluate ViT-Prisma SAEs on bigG final patch tokens. Run from the Prisma-SAE repo root.

  # existing saev shards (no ViT compute)
  uv run python train_prisma.py --route shards --train-shards DIR --val-shards DIR --ckpt-dir DIR --total-tokens 128000000
  # Open Images thumbnails, bigG run on the fly (no shards)
  uv run python train_prisma.py --route images --train-images DIR --val-images DIR --ckpt-dir DIR --total-tokens 128000000

--sweep FILE trains one SAE per entry of FILE's make_cfgs() -> list[dict] (bigG_config keyword overrides, e.g. {"k": 32, "expansion_factor": 32}). --set key=value overrides any VisionModelSAERunnerConfig field for every run. Eval metrics for each run are appended to --results (default CKPT_DIR/results.jsonl).
"""

import argparse
import ast
import importlib.util
import json
import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import prisma_bigg as pb  # noqa: E402


def parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, raw = item.partition("=")
        assert key and raw, f"--set expects key=value, got {item!r}"
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def load_sweep(fpath: str | None) -> list[dict]:
    if fpath is None:
        return [{}]
    spec = importlib.util.spec_from_file_location("sweep", fpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfgs = mod.make_cfgs()
    assert isinstance(cfgs, list) and all(isinstance(c, dict) for c in cfgs), "make_cfgs() must return list[dict]"
    return cfgs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--route", choices=["images", "shards"], required=True)
    p.add_argument("--train-images", help="ImageFolder root (images route)")
    p.add_argument("--val-images", help="ImageFolder root for eval (images route)")
    p.add_argument("--train-shards", help="saev shards dir with metadata.json (shards route)")
    p.add_argument("--val-shards", help="held-out saev shards dir (shards route)")
    p.add_argument("--saev-layer", type=int, default=pb.N_BLOCKS, help="layer index in the shards; 48 = ln_post after the clip.py patch")
    p.add_argument("--target", default="ln_post", help='"ln_post" or a block index (that block\'s hook_resid_post)')
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--total-tokens", type=int, required=True)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--expansion", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--sweep", help="python file defining make_cfgs() -> list[dict]")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override for all runs, repeatable")
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--eval-images", type=int, default=2000, help="images route: eval images (x256 tokens)")
    p.add_argument("--model-local-path", help="local open_clip_pytorch_model.bin (with open_clip_config.json beside it) for offline nodes")
    p.add_argument("--results", help="jsonl file for eval metrics")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True  # same as the saev shard writer
    torch.backends.cudnn.allow_tf32 = True
    target = args.target if args.target == "ln_post" else int(args.target)
    common = dict(k=args.k, expansion_factor=args.expansion, lr=args.lr, train_batch_size=args.batch_size)
    common.update(parse_set(args.set))
    results_fpath = args.results or os.path.join(args.ckpt_dir, "results.jsonl")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    def make_cfg(overrides: dict, n_blocks: int) -> pb.HookedSAEConfig:
        kw = {**common, **overrides}
        kw.update(pb.target_fields(target, n_blocks))
        return pb.bigG_config(target, total_tokens=args.total_tokens, checkpoint_path=args.ckpt_dir, **kw)

    if args.route == "shards":
        from saev.data import shards

        assert args.train_shards and args.val_shards, "--train-shards and --val-shards are required"
        mds = [shards.Metadata.load(pathlib.Path(d)) for d in (args.train_shards, args.val_shards)]
        for md, dpath in zip(mds, (args.train_shards, args.val_shards)):
            assert args.saev_layer in md.layers, f"layer {args.saev_layer} not in {md.layers} ({dpath})"
        # For ln_post, use the saev layer index as hook_point_layer (48 for bigG). A block target must match the shards' layer.
        n_blocks = args.saev_layer
        assert target == "ln_post" or target == args.saev_layer, "block target must equal --saev-layer"

        def make_trainer(cfg):
            for md in mds:
                assert md.d_model == cfg.d_in, f"shards d_model {md.d_model} != d_in {cfg.d_in}"
                assert md.content_tokens_per_example == cfg.context_size, f"shards have {md.content_tokens_per_example} patches, cfg.context_size={cfg.context_size}"
            return pb.SaevSAETrainer(cfg, pb.saev_loader(args.train_shards, args.saev_layer, cfg.train_batch_size, seed=17))

        def make_eval(cfg):
            return pb.saev_loader(args.val_shards, args.saev_layer, cfg.train_batch_size, seed=0)

    else:
        import torchvision
        from vit_prisma.models.model_loader import load_hooked_model
        from vit_prisma.transforms import get_clip_val_transforms

        assert args.train_images and args.val_images, "--train-images and --val-images are required"
        probe = make_cfg({}, pb.N_BLOCKS)
        model = load_hooked_model(probe.model_name, device=probe.device, local_path=args.model_local_path).eval()
        n_blocks = model.cfg.n_layers
        tf = get_clip_val_transforms()  # == open_clip's bigG preprocessing
        train_ds = torchvision.datasets.ImageFolder(args.train_images, transform=tf)
        val_ds = torchvision.datasets.ImageFolder(args.val_images, transform=tf)
        print(f"train images: {len(train_ds)}, val images: {len(val_ds)}", flush=True)

        def make_trainer(cfg):
            return pb.ImagesSAETrainer(cfg, model, train_ds)

        def make_eval(cfg):
            return pb.image_token_batches(model, val_ds, cfg, n_images=args.eval_images, batch_images=cfg.store_batch_size, num_workers=min(8, cfg.num_workers))

    for overrides in load_sweep(args.sweep):
        cfg = make_cfg(overrides, n_blocks)
        print(f"=== run {overrides or '(defaults)'}: hook {cfg.hook_point}, d_sae {cfg.d_sae}, k {cfg.activation_fn_kwargs.get('k')}, steps {cfg.total_training_steps}", flush=True)
        t0 = time.time()
        trainer = make_trainer(cfg)
        sae = trainer.run()
        metrics = pb.evaluate_sae(sae, make_eval(cfg), n_batches=args.eval_batches)
        row = dict(route=args.route, overrides=overrides, common=common, hook_point=cfg.hook_point, checkpoint_dir=trainer.cfg.checkpoint_path, minutes=round((time.time() - t0) / 60, 1), **metrics)
        print(json.dumps(row), flush=True)
        with open(results_fpath, "a") as fd:
            fd.write(json.dumps(row) + "\n")
        del trainer, sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
