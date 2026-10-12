"""ViT-Prisma SAEs on OpenCLIP ViT-bigG/14 final patch tokens.

Target tensor: open_clip `visual.ln_post` output, unprojected, CLS dropped: 256 x 1664 per image. This is the same tensor as the saev setup in the SAETraining repo (contrib/bigg_openimages; saev layer 48 after its clip.py patch). In Prisma it is `hook_ln_final[:, 1:]` (checked by check_tokens_prisma.py).

Two ways to feed Prisma's VisionSAETrainer, both training the same SAE code:
- images: run bigG on the fly over an ImageFolder (the Open Images thumbnails). ImagesSAETrainer.
- shards: read the existing saev shards. SaevSAETrainer.
Evaluate either with evaluate_sae, whose normalized MSE uses saev's definition, so numbers are comparable across routes and with saev runs.

Prisma quirks handled here (see CLAUDE.md for details):
- VisionModelSAERunnerConfig can only name blocks.* hooks, so HookedSAEConfig adds hook_point_name.
- The stock on-the-fly store always stops the forward pass at a block, so hook_ln_final never fires; ImageHookStore runs the full forward for non-block hooks.
- With use_patches_only, the stock buffer needs context_size == number of kept tokens (256), not 257.
- Training length is 1.3M images x num_epochs x tokens/image regardless of the dataset; bigG_config converts a token budget.
- TopK needs activation_fn_kwargs={"k": ...}; per-token layer_norm input normalization is off (it clashes with the geometric-median b_dec init and would hide token norms from the SAE); W&B stays on because checkpointing calls wandb.log unconditionally.
"""

import dataclasses
from collections.abc import Callable, Iterable, Iterator

import torch
from torch.utils.data import DataLoader, Subset

from vit_prisma.sae import VisionModelSAERunnerConfig, VisionSAETrainer
from vit_prisma.sae.training.activations_store import VisionActivationsStore

MODEL_NAME = "open-clip:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
LN_POST = "hook_ln_final"  # Prisma's name for open_clip visual.ln_post output, all 257 tokens
N_BLOCKS, D_MODEL, N_PATCHES = 48, 1664, 256
PRISMA_IMAGES_PER_EPOCH = 1_300_000  # hard-coded in VisionModelSAERunnerConfig.total_training_images


@dataclasses.dataclass
class HookedSAEConfig(VisionModelSAERunnerConfig):
    """VisionModelSAERunnerConfig that can also point at hooks outside blocks.* (e.g. hook_ln_final).

    The base class always builds hook_point as f"blocks.{hook_point_layer}.{layer_subtype}". Checkpoints pickle this class, so prisma_bigg must be importable when loading them.
    """

    hook_point_name: str | None = None
    val_n_batches: int = 4  # fixed val batches scored at every metrics step (0 = off); --set val_n_batches=N

    @property
    def hook_point(self) -> str:
        if self.hook_point_name is not None:
            return self.hook_point_name
        return f"blocks.{self.hook_point_layer}.{self.layer_subtype}"


def target_fields(target: str | int, n_blocks: int = N_BLOCKS) -> dict:
    """Config fields for a target: "ln_post" (final tokens) or a block index (that block's hook_resid_post)."""
    if target == "ln_post":
        # hook_point_layer = n_blocks matches saev's numbering after the clip.py patch (resblocks, then ln_post).
        return dict(hook_point_name=LN_POST, hook_point_layer=n_blocks, layer_subtype="ln_post")
    assert isinstance(target, int) and 0 <= target < n_blocks, f"bad target {target!r}"
    return dict(hook_point_name=None, hook_point_layer=target, layer_subtype="hook_resid_post")


def bigG_config(
    target: str | int = "ln_post",
    *,
    total_tokens: int,
    checkpoint_path: str,
    k: int = 64,
    expansion_factor: int = 16,
    train_batch_size: int = 4096,
    lr: float = 1e-3,
    **overrides,
) -> HookedSAEConfig:
    """Config for bigG patch tokens. Any VisionModelSAERunnerConfig field can be overridden by keyword."""
    kwargs = dict(
        model_name=MODEL_NAME,
        **target_fields(target),
        d_in=D_MODEL,
        context_size=N_PATCHES,  # kept tokens per image; the on-the-fly buffer requires this with use_patches_only
        use_patches_only=True,
        expansion_factor=expansion_factor,
        activation_fn_str="topk",
        activation_fn_kwargs={"k": k},
        normalize_activations="none",
        b_dec_init_method="geometric_median",
        train_batch_size=train_batch_size,
        lr=lr,
        lr_scheduler_name="cosineannealingwarmup",
        lr_warm_up_steps=1000,
        n_batches_in_buffer=100,  # images route: ~100 x 32 images mixed per buffer (~5.5 GB on GPU for bigG)
        store_batch_size=32,
        use_cached_activations=False,
        log_to_wandb=True,
        wandb_project="bigg-prisma-sae",
        n_checkpoints=3,
        checkpoint_path=checkpoint_path,
        _device="cuda",
        _dtype="float32",
    )
    kwargs.update(overrides)
    if "num_epochs" not in overrides:
        # Prisma counts context_size - 1 tokens per image when use_patches_only is set.
        tokens_per_image = kwargs["context_size"] - int(kwargs["use_patches_only"])
        kwargs["num_epochs"] = total_tokens / (PRISMA_IMAGES_PER_EPOCH * tokens_per_image)
    return HookedSAEConfig(**kwargs)


@torch.no_grad()
def get_hook_acts(model, imgs_bchw: torch.Tensor, hook: str, layer: int) -> torch.Tensor:
    """Activations at `hook` for every token, shape (batch, tokens, d)."""
    stop = layer + 1 if hook.startswith("blocks.") else None  # only block hooks can stop early
    _, cache = model.run_with_cache(imgs_bchw, names_filter=[hook], stop_at_layer=stop)
    return cache[hook]


def _close(it) -> None:
    close = getattr(it, "close", None)
    if close is not None:
        close()


class ImageHookStore(VisionActivationsStore):
    """Prisma's on-the-fly store, able to read non-block hooks such as hook_ln_final."""

    def get_activations(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        assert not self.cfg.cls_token_only and not self.cfg.is_transcoder, "not supported here"
        acts_btd = get_hook_acts(self.model, batch_tokens, self.cfg.hook_point, self.cfg.hook_point_layer)
        n_tok = self.cfg.context_size + int(self.cfg.use_patches_only)
        assert acts_btd.shape[1] == n_tok, f"{self.cfg.hook_point} has {acts_btd.shape[1]} tokens, expected {n_tok}"
        return acts_btd[:, :, None, :]  # (batch, tokens, n_layers=1, d); get_buffer drops CLS

    def close(self) -> None:
        _close(self.image_dataloader_iter)
        _close(getattr(self, "image_dataloader_eval_iter", None))


def _batch_acts(batch) -> torch.Tensor:
    """The (batch, d) activation tensor from a saev batch (dict with "act") or a plain tensor."""
    if isinstance(batch, dict):
        return batch["act"]
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


class SaevStore:
    """Stands in for Prisma's activation store, fed by a saev loader.

    Prisma's trainer only uses next_batch() -> (train_batch_size, 1, d_in) and storage_buffer -> (N, 1, d_in), the latter once for the b_dec geometric median.
    """

    def __init__(self, make_loader: Callable[[], Iterable], cfg: VisionModelSAERunnerConfig, n_init_batches: int = 8):
        self.make_loader, self.cfg, self.n_passes = make_loader, cfg, 0
        self._it = iter(make_loader())
        first = torch.cat([self._next_acts() for _ in range(n_init_batches)])
        assert first.shape[-1] == cfg.d_in, f"saev d={first.shape[-1]}, cfg.d_in={cfg.d_in}"
        self.storage_buffer = first[:, None, :]
        self._buf = first  # train on the init tokens too

    def _next_acts(self) -> torch.Tensor:
        try:
            batch = next(self._it)
        except StopIteration:
            self.n_passes += 1
            self._it = iter(self.make_loader())
            batch = next(self._it)
        return _batch_acts(batch).to(self.cfg.device, self.cfg.dtype, non_blocking=True)

    def next_batch(self) -> torch.Tensor:
        bsz = self.cfg.train_batch_size
        while self._buf.shape[0] < bsz:
            self._buf = torch.cat([self._buf, self._next_acts()])
        out, self._buf = self._buf[:bsz], self._buf[bsz:]
        return out[:, None, :]

    def close(self) -> None:
        _close(self._it)  # saev's loader generator shuts its workers down in a finally block
        self._buf = self._buf[:0]


class _CleanupMixin:
    """The base run() leaves the loader and the W&B run open, which breaks the next run in the same process."""

    def run(self):
        try:
            return super().run()
        finally:
            self.activations_store.close()
            if self.cfg.log_to_wandb:
                import wandb

                wandb.finish()


class _SaevMetricsMixin:
    """At every Prisma metrics step (wandb_log_frequency), also log saev's metrics to the same W&B step.

    metrics/normalized_mse, metrics/sse_sae, metrics/sse_baseline: the current train batch, as in saev's train loop (SSE about the batch mean, fp64).
    val/normalized_mse, val/l0: a fixed val set of cfg.val_n_batches batches from `make_val`, loaded once, scored with evaluate_sae.
    """

    make_val: Callable[[], Iterable] | None = None
    _val_batches: list | None = None

    def _log_metrics(self, sparse_autoencoder, hyperparams, optimizer, sae_in, sae_out, *args):
        super()._log_metrics(sparse_autoencoder, hyperparams, optimizer, sae_in, sae_out, *args)
        import wandb

        step = args[-2]  # args ends with (..., l0, n_training_steps, n_training_tokens)
        x, xh = sae_in.double(), sae_out.double()
        sse, sst = (xh - x).pow(2).sum().item(), (x - x.mean(0)).pow(2).sum().item()
        metrics = {"metrics/normalized_mse": sse / sst, "metrics/sse_sae": sse, "metrics/sse_baseline": sst}
        if self.make_val is not None and self.cfg.val_n_batches > 0:
            if self._val_batches is None:
                it = iter(self.make_val())
                try:
                    self._val_batches = [_batch_acts(b).to(self.cfg.device, self.cfg.dtype) for _, b in zip(range(self.cfg.val_n_batches), it)]
                finally:
                    _close(it)
            ev = evaluate_sae(sparse_autoencoder, lambda: self._val_batches, n_batches=len(self._val_batches))
            sparse_autoencoder.train()
            metrics.update({"val/normalized_mse": ev["normalized_mse"], "val/l0": ev["l0"]})
        wandb.log(metrics, step=step)


class ImagesSAETrainer(_CleanupMixin, _SaevMetricsMixin, VisionSAETrainer):
    """Runs the ViT on images during training. dataset items must be (image_tensor, label)."""

    def __init__(self, cfg: HookedSAEConfig, model, dataset, eval_dataset=None, make_val: Callable[[], Iterable] | None = None):
        self.make_val = make_val
        super().__init__(cfg, model, dataset, eval_dataset)

    def initialize_activations_store(self, dataset, eval_dataset):
        return ImageHookStore(self.cfg, self.model, dataset, eval_dataset=eval_dataset, num_workers=self.cfg.num_workers)


class SaevSAETrainer(_CleanupMixin, _SaevMetricsMixin, VisionSAETrainer):
    """Reads activations from saev shards instead of running the ViT."""

    def __init__(self, cfg: HookedSAEConfig, make_loader: Callable[[], Iterable], make_val: Callable[[], Iterable] | None = None):
        self._make_loader, self.make_val = make_loader, make_val
        super().__init__(cfg, model=None, dataset=[], eval_dataset=[])

    def initialize_activations_store(self, dataset, eval_dataset):
        return SaevStore(self._make_loader, self.cfg)


def saev_loader(shards_dpath: str, layer: int, batch_size: int, seed: int = 17) -> Callable[[], Iterable]:
    """Factory for saev's shuffled loader over patch tokens (CLS excluded)."""
    import pathlib

    import saev.data.shuffled as shuffled

    def make_loader():
        cfg = shuffled.Config(shards=pathlib.Path(shards_dpath), layer=layer, tokens="content", batch_size=batch_size, drop_last=True, seed=seed)
        return shuffled.DataLoader(cfg)

    return make_loader


def _stack_images(items) -> torch.Tensor:
    return torch.stack([x[0] for x in items])


def image_token_batches(model, dataset, cfg: VisionModelSAERunnerConfig, n_images: int = 2000, batch_images: int = 32, num_workers: int = 8) -> Callable[[], Iterator[dict]]:
    """Factory yielding {"act": (batch_images * tokens, d)} target tokens from a fixed, evenly spaced subset of `dataset`."""
    n = min(n_images, len(dataset))
    idx = torch.linspace(0, len(dataset) - 1, n).long().tolist()
    subset = Subset(dataset, idx)

    def make_batches():
        loader = DataLoader(subset, batch_size=batch_images, shuffle=False, num_workers=num_workers, collate_fn=_stack_images)
        for imgs in loader:
            acts_btd = get_hook_acts(model, imgs.to(cfg.device, cfg.dtype), cfg.hook_point, cfg.hook_point_layer)
            if cfg.use_patches_only:
                acts_btd = acts_btd[:, 1:]
            yield {"act": acts_btd.reshape(-1, acts_btd.shape[-1])}

    return make_batches


@torch.no_grad()
def evaluate_sae(sae, make_batches: Callable[[], Iterable], n_batches: int = 50, dense_thresh: float = 0.01) -> dict:
    """Reconstruction and sparsity metrics, defined as in saev's eval:

    normalized_mse = SSE(x - x_hat) / SSE(x - mean(x)) over all evaluated tokens; l0 = mean active latents per token; n_dead = latents that never fired; n_dense = latents firing on more than dense_thresh of tokens (saev uses 1%).
    """
    sae.eval()
    device, dtype = sae.cfg.device, sae.cfg.dtype
    sse = sum_x2 = l0_sum = 0.0
    sum_x, n_tok = 0.0, 0
    fire_counts = torch.zeros(sae.cfg.d_sae, device=device, dtype=torch.float64)
    it = iter(make_batches())
    try:
        for _, batch in zip(range(n_batches), it):
            x = _batch_acts(batch).to(device, dtype)
            out, feats, *_ = sae(x)
            xd = x.double()
            sse += (out.double() - xd).pow(2).sum().item()
            sum_x = sum_x + xd.sum(0)
            sum_x2 += xd.pow(2).sum().item()
            fired = feats > 0
            l0_sum += fired.sum().item()
            fire_counts += fired.sum(0).double()
            n_tok += x.shape[0]
    finally:
        _close(it)
    assert n_tok > 0, "no evaluation batches"
    sst = sum_x2 - sum_x.pow(2).sum().item() / n_tok
    freq = fire_counts / n_tok
    d_sae = sae.cfg.d_sae
    n_dead, n_dense = int((freq == 0).sum()), int((freq > dense_thresh).sum())
    return dict(normalized_mse=sse / sst, l0=l0_sum / n_tok, n_dead=n_dead, dead_pct=100 * n_dead / d_sae, n_dense=n_dense, dense_pct=100 * n_dense / d_sae, n_tokens=n_tok)
