# Prisma-SAE: SAE on OpenCLIP ViT-bigG/14 final patch tokens (Open Images v7) with ViT-Prisma

Verified against Prisma-Multimodal/ViT-Prisma @ 64daf14 (2026-10-06), Imageomics/saev @ 9715c45, open_clip_torch 3.3.0. Re-check if versions differ. Everything below was tested on CPU in a standalone uv project built with the commands in "Repo setup", using a random-weight open_clip ViT-B-32 fixture (same VisionTransformer class as bigG) and real saev-format shards. Not yet run on real bigG weights or a GPU.

## What this is
- A standalone repo (Prisma-SAE), separate from SAETraining on purpose, so nothing here can break the saev setup. Same target tensor and data as SAETraining's `contrib/bigg_openimages/`. Prisma replaces saev's trainer: Prisma's `VisionSAETrainer` (TopK or ReLU+L1 SAE, Adam, Prisma checkpoints and W&B logging).
- Two data routes, one trainer, one eval:
  - `shards`: train on the saev shards SAETraining already wrote (layer 48 = ln_post). No ViT compute. Default, and the cheap choice for sweeps.
  - `images`: run bigG on the fly over the Open Images thumbnails. No shards and no scratch budget, but every run pays a full bigG forward per training token.
- ImageNet is not needed and buys nothing. Prisma's ImageNet tooling is (a) `load_dataset` for `imagenet1k`, which we don't use (we pass datasets in), and (b) `SparsecoderEval`, a zero-shot "CE recovered" score. For final patch tokens that score is meaningless: the CLIP image embedding is `proj(ln_post(x)[:, 0])`, so patch tokens never reach it. Use Open Images.

## Boundary with SAETraining (critical)
- SAETraining is a read-only input. From here: read its shards (saev memmaps are opened with `mode="r"`), its thumbnails and its val folder. Never write, move or delete anything in SAETraining's repo or data dirs, never run its scripts from here, and never edit its `env.sh`.
- Prisma-SAE never writes shards. New or regenerated shards are made in SAETraining with its patched saev.
- Nothing is imported from the SAETraining checkout. The shards loader comes from upstream saev, installed into this repo's own env at the same commit SAETraining tracks (9715c45). Reading shards only needs upstream code, because the clip.py ln_post patch changes what gets written, not the shard format (float32 memmaps).
  - Exception: if SAETraining's `src/saev/data/shards.py` or the loaders were changed (the fp16 / token-subsampling idea in its notes) and shards were written that way, upstream can't read them. Then depend on SAETraining pinned to a commit instead (`uv add "saev @ git+https://github.com/Arnav-Aggarwal13/SAETraining@<commit>"`, needs GitHub auth if private). That builds a copy into this env without touching the checkout.
- All outputs live under Prisma-SAE's own paths: checkpoints, `results.jsonl`, W&B offline runs, Slurm logs.

## Project files (repo root)
- `CLAUDE.md`: this file.
- `pyproject.toml`, `uv.lock`: this repo's own uv env (see "Repo setup").
- `prisma_bigg.py`: library. `HookedSAEConfig` (config that can name `hook_ln_final`), `target_fields`, `bigG_config`, `ImageHookStore` + `ImagesSAETrainer` (images route), `SaevStore` + `SaevSAETrainer` (shards route), `saev_loader`, `image_token_batches`, `evaluate_sae`. All Prisma workarounds live here as subclasses; Prisma itself is not patched.
- `train_prisma.py`: CLI. One run, or `--sweep FILE`; evaluates each run and appends a row to `results.jsonl` in `--ckpt-dir`.
- `check_tokens_prisma.py`: asserts Prisma's `hook_ln_final[:, 1:]` == open_clip `output_tokens` (and == saev shard tokens with `--shards`). Must PASS before training.
- `sweeps/first_sweep.py`: the lr/k/expansion sweep from "Starting config" below, in `--sweep` format.
- TO CREATE:
  - `env.sh`: this repo's own paths. Copy the VALUES from SAETraining's `env.sh` (don't `source` that file, so later edits there can't silently change runs here):
    - `TRAIN_IMAGES`, `VAL_IMAGES`: thumbnail and val folders
    - `SHARDS_ROOT`, `TRAIN_HASH`, `VAL_HASH`: train shards are `$SHARDS_ROOT/$TRAIN_HASH`
    - `HF_HOME`: the same cache, so the bigG weights aren't downloaded twice
    - `RUNS_DIR` (e.g. `~/prisma_runs`), `WANDB_DIR` (e.g. `$RUNS_DIR/wandb`)
  - `slurm/check_tokens.sbatch` (1 GPU, ~20 GB VRAM) and `slurm/train.sbatch`. Take partition/account/resource lines from SAETraining's `slurm/train.sbatch` (read it, don't edit it). Each `source env.sh` and runs `uv run python <script>.py ...`. Set `WANDB_MODE=offline` if compute nodes have no internet. Overrides via `--export=ALL,VAR=...`.
  - `logs/`: must exist before `sbatch`.
  - `.gitignore`: `.venv/`, `__pycache__/`, `logs/`, `wandb/`, `*.pt`, `*.bin`, `results.jsonl`.
- Superseded: the earlier `prisma_saev_bridge.py` / `train_bigG_saev_prisma.ipynb`. They trained on the right tensor from the shards but labeled it `blocks.48.hook_resid_post`, which doesn't exist. Don't copy them in.

## Repo setup
- Create the repo and its env (exact commands verified; resolves 161 packages, open_clip_torch 3.3.0):
  ```
  mkdir Prisma-SAE && cd Prisma-SAE && git init
  uv init --bare --python 3.12 --name prisma-sae
  uv python pin 3.12
  uv add "vit-prisma @ git+https://github.com/Prisma-Multimodal/ViT-Prisma@64daf1425f7e5311844213fd0fb6f4127a30cd16" \
         "saev @ git+https://github.com/Imageomics/saev@9715c45ffc5aa822bfdac9610b62664503ed1a71"
  ```
  Then copy in the files above. `uv init --bare` makes a virtual project (no package build), which is what we want for a flat folder of scripts.
- GitHub: `origin` = Arnav-Aggarwal13/Prisma-SAE (private is fine). No upstream remote; Prisma and saev are pinned dependencies, not forks. Bump either by changing the `rev` in `pyproject.toml`, re-running the token check, and updating the "Verified against" line.
- Do not edit Prisma's or saev's installed code. If a Prisma change becomes unavoidable, clone ViT-Prisma next to this repo, `uv add --editable ../ViT-Prisma`, and record the patch here.
- ALWAYS run from the repo root: `uv run python <script>.py`. Scripts put their own directory on `sys.path`, so `import prisma_bigg` works.
- Code style: asserts over if/raise, ascii only, no em dashes, short code (same conventions as saev).
- Model download: `load_hooked_model` fetches `open_clip_pytorch_model.bin` (~10 GB, both towers) from `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` into the HF cache. open_clip may have cached only the safetensors for saev, so expect one more download. Offline nodes: pass `--model-local-path` to the .bin with `open_clip_config.json` beside it (Prisma reads the config from the same folder).
- Data never goes in git.

## Goal
Train an SAE on the 256 x 1664 patch tokens that `model.visual(img)` returns with `output_tokens=True` (ln_post output, unprojected), one token per training example, CLS excluded. Identical to the saev setup.

## Which Prisma activation is the target (critical)
- Prisma's `HookedViT.forward`: embed, `ln_pre`, 48 blocks, then `x = ln_final(residual)` on ALL tokens, `hook_ln_final(x)` ([B, 257, 1664]), then CLS, then head (= proj), then normalize.
- `ln_final.w/.b` are converted from open_clip `visual.ln_post.weight/.bias` (`weight_conversion.py`). `load_hooked_model` defaults to `fold_ln=False`, `center_writing_weights=False`, so nothing is folded into it.
- Target = `hook_ln_final[:, 1:]` (`ln_final.hook_normalized` is the same tensor). It is NOT `blocks.47.hook_resid_post`, which is the residual stream before ln_post (= saev's unpatched layer 47). For the record, Prisma `blocks.{i}.hook_resid_post` == open_clip `resblocks[i]` output (same per-block code; checked on the last block below).
- Tested on the fixture through Prisma's real OpenCLIP loading path: `hook_ln_final[:, 1:]` vs open_clip `output_tokens` max abs diff 2.6e-6 (fp32 noise); `blocks.11.hook_resid_post` vs `resblocks[-1]` output 3.3e-6.
- Prisma's config cannot name it. `VisionModelSAERunnerConfig.hook_point` is always `f"blocks.{hook_point_layer}.{layer_subtype}"` and silently ignores assignment. `HookedSAEConfig.hook_point_name = "hook_ln_final"` fixes this. `target_fields("ln_post")` sets it with `hook_point_layer=48`, matching saev's numbering after the clip.py patch.
- Prisma's stock on-the-fly store calls `run_with_cache(..., stop_at_layer=layer + 1)`, which returns before `ln_final`, so `hook_ln_final` never fires. `ImageHookStore` runs the full forward for non-block hooks.
- Preprocessing: Prisma `get_clip_val_transforms()` == open_clip's bigG preprocess (resize shortest side 224 bicubic, center crop 224, OpenAI mean/std). `check_tokens_prisma.py` compares pixels.

## Token check (must PASS before training)
- `uv run python check_tokens_prisma.py --images $TRAIN_IMAGES --shards $SHARDS_ROOT/$TRAIN_HASH --saev-layer 48` (GPU, two copies of the bigG vision tower, ~20 GB).
- Checks, on 8 images spread across the ImageFolder:
  1. pixels: Prisma transform vs open_clip preprocess (rel tol 1e-5)
  2. Prisma `hook_ln_final[:, 1:]` vs open_clip `output_tokens`, fp32 with TF32 off (rel tol 1e-4)
  3. saev shard tokens vs Prisma (rel tol 1e-2, since the shards were written with TF32)
- Images are matched to shard examples by torchvision ImageFolder index. saev's `ImgFolderDataset` subclasses torchvision ImageFolder on the same root, and the content-token global index is `example * 256 + token`. The script asserts the folder's image count == `metadata.json` `n_examples`, so it catches a folder changed after sharding.
- Prints PASS/FAIL per check and exits 1 on failure. On the fixture: PASS (rel err ~5e-7); with noise added to one shard: FAIL (rel err 0.28).

## Data routes
### shards (existing saev shards, default)
- Uses saev's own shuffled loader (`saev.data.shuffled`, `tokens="content"`, `layer=48`) through `saev_loader`. `SaevStore` re-chunks batches to `train_batch_size` and restarts the loader at the end of each pass (counted in `n_passes`).
- Never use Prisma's own cached-activation path (`use_cached_activations=True`):
  - `_load_cached_activations` resets to `0.pt` on every refill. In a test it served only the first 16,000 tokens of a 100k-token cache, forever.
  - It wants Prisma's `.pt` format, not saev's `.bin` memmaps.
- `train_prisma.py --route shards --train-shards $SHARDS_ROOT/$TRAIN_HASH --val-shards $SHARDS_ROOT/$VAL_HASH --saev-layer 48 ...`. It asserts `d_model`, the 256-token count, and that layer 48 is in both shards' `metadata.json`.
- Token budget is capped by what is sharded (~128M tokens at ~500k images). More tokens than that means more than one epoch.

### images (bigG on the fly)
- The thumbnail layout `<root>/<id[:2]>/<id>.jpg` works as a torchvision ImageFolder as-is. The 2-char prefix dirs act as classes and the labels are ignored. The NSD quarantine dir must be outside the root. Eval uses the held-out val folder.
- Prisma's buffer: each refill runs bigG on `n_batches_in_buffer/2 x store_batch_size` images, mixes them with the retained half, and serves half. Prisma's default (20 x 32) keeps only ~640 images in play, so batches are dominated by a few images. `bigG_config` sets `n_batches_in_buffer=100` (~3,200 images, ~5.5 GB of fp32 tokens on GPU, a few times that at peak). Raise it if VRAM allows.
- With `use_patches_only`, Prisma's buffer needs `context_size` == kept tokens (256). 257 crashes ("expanded size 257 must match 256"). `bigG_config` sets 256.
- Cost: every 4096-token batch is 16 images through all 48 blocks, the same compute as writing shards, paid again per run and per sweep entry. Time a short run (small `--total-tokens`) before committing GPU hours. TF32 is enabled, as in the shard writer.
- Storage: thumbnails only (~20 KB/img), so no scratch cap. All streamed thumbnails are usable, beyond the ~500k images the shards hold.
- If the GPU is starved by JPEG reads from pool (HDD), copy the thumbnail root to a new folder on scratch (~40 GB per 2M images; a copy, never a move of SAETraining's data) and raise `num_workers` to match the job's CPUs.
- `load_hooked_model` loads the full open_clip state dict (both towers, ~10 GB) on CPU before converting, so request enough job RAM.

## Storage on MIT ORCD Engaging
- Inputs stay where SAETraining put them (read-only from here): thumbnails on pool, shards on scratch. This repo, its `.venv` and `$RUNS_DIR` live on home. The separate `.venv` is a second copy of torch and friends (~7 GB on home). Check usage in `~/orcd/.quota`.
- Prisma checkpoints: each `n_images_<N>.pt` holds fp32 `W_enc` + `W_dec`, ~355 MB at 16x (26624 x 1664 x 2 x 4 B), ~710 MB at 32x. `bigG_config` writes `n_checkpoints=3` per run (~1.1 GB per 16x run). Keep runs on home (backed up, 200 GB) and prune old checkpoints.
- W&B offline runs go to `$WANDB_DIR` (default `./wandb`); point it under `$RUNS_DIR`, not the repo.
- For CPU jobs add: `#SBATCH --account=mit_amf_standard_cpu` and `#SBATCH --qos=mit_amf_standard_cpu`
- For GPU jobs add: `#SBATCH --account=mit_amf_standard_gpu` and `#SBATCH --qos=mit_amf_standard_gpu`

## Training (`train_prisma.py`, which wraps `VisionSAETrainer.run`)
- Per step (Prisma `train_sae.py`):
  1. renormalize decoder rows to unit norm
  2. forward, then backward
  3. clip grad norm to 1.0
  4. remove the gradient component parallel to each decoder row
  5. Adam step, then scheduler step
- Input normalization: `bigG_config` sets `normalize_activations="none"`, matching saev (raw ln_post tokens). Prisma's default `"layer_norm"` has two problems:
  - It standardizes each token before subtracting `b_dec`, but `b_dec` is initialized to the geometric median of the raw tokens. In a toy test it had not recovered after 400 steps: explained variance ~0, vs 0.29 with normalization off.
  - It also hands each token's mean/std to the reconstruction for free, so its MSE isn't comparable to saev's.
- Activation options:
  - `topk`: per-token TopK, ReLU on the top-k values. Needs `activation_fn_kwargs={"k": ...}`; `bigG_config` always sets it, because the bare default fails an assert.
  - `relu` with L1 (`l1_coefficient`, default 2e-4).
  - gated.
- Not available: BatchTopK, AuxK, Matryoshka, JumpReLU (the config's Literal lists `jumprelu`, but the trainer raises). Ghost grads (`use_ghost_grads`) exist for the standard arch only and were designed for ReLU.
  - The closest match to the saev starting config (BatchTopK k=64, 16x, `n_prefixes=1`) is TopK k=64, 16x.
  - Without AuxK, expect more dead latents than saev at the same settings. Watch `n_dead`.
- Loss: elementwise squared error divided by each token's L2 norm about the batch mean (norm, not squared norm), averaged. Plus L1 for ReLU. Logged values are not comparable to saev's `loss/mse`.
- Init: `W_enc` and `W_dec` independent Kaiming, decoder rows unit norm. `b_dec` is the geometric median of the first buffer (8 saev batches on the shards route). saev used data init and `b_dec = 0`.
- Optimizer: Adam (default betas, no weight decay). `bigG_config` default lr 1e-3 (Prisma's default; saev used 4e-4); in a toy comparison 4e-4 learned much slower at the same step count. Schedule `cosineannealingwarmup` with 1000 warmup steps. Its floor is effectively 0, not lr/10: `lr_end=lr/10` is passed but used as a multiplier.
- Batch: `train_batch_size` 4096 by default (saev used 16384). Use `--batch-size 16384` for a closer match.
- Length: Prisma computes `total_training_tokens = 1.3M x num_epochs x (context_size - 1)` whatever the dataset. `bigG_config` derives `num_epochs` from `--total-tokens`, so steps = total_tokens // batch.
- Checkpoints: written at `n_checkpoints` evenly spaced points plus the end, to `<ckpt-dir>/<8 hex>-<wandb_project>/`:
  - `config.json`, `n_images_<N>.pt`, `n_images_<N>_log_feature_sparsity.pt`
  - N = tokens / context_size, not images
- Loading a checkpoint: `import prisma_bigg` first (the checkpoint pickles `HookedSAEConfig`), then `SparseAutoencoder.load_from_pretrained(path)`. Without the import it fails with "No module named 'prisma_bigg'". `config.json` keeps `hook_point_name`, but plain `VisionModelSAERunnerConfig.load_config` drops it.
- W&B:
  - Keep `log_to_wandb=True` whenever `n_checkpoints > 0`. `checkpoint()` calls `wandb.log` unconditionally, and wandb raises without `init`.
  - Use `WANDB_MODE=offline` plus `wandb sync` later if nodes lack internet.
  - Checkpoints are never uploaded as W&B artifacts: the artifact name is built empty and the ValueError is swallowed. Disk is the only copy.
- Sweeps:
  - `--sweep FILE`, where FILE defines `make_cfgs() -> list[dict]` of `bigG_config` keyword overrides (e.g. `{"k": 32}`, `{"expansion_factor": 32}`, `{"lr": 3e-4}`, or any `VisionModelSAERunnerConfig` field).
  - Runs are sequential, each with its own data stream (unlike saev's parallel-on-one-stream sweeps).
  - `--set key=value` overrides a field for every run (e.g. `--set n_batches_in_buffer=200`).
  - Prisma's base `run()` leaves the W&B run and the loader open, which breaks the next run in the process. `_CleanupMixin` closes both.

## Metrics
Prisma logs to W&B every `wandb_log_frequency` (10) steps, on the current training batch. Use these for trends only:
- `losses/mse_loss`: Prisma's loss above.
- `metrics/explained_variance`: per-token `1 - ||x - x_hat||^2 / ||x - batch_mean||^2`, averaged over tokens. Not `1 - normalized_mse`.
- `metrics/l0`: = k for TopK.
- `sparsity/dead_features`: latents that haven't fired in `dead_feature_window` = 5000 steps (20M tokens at batch 4096; saev uses 10M). Set `--set dead_feature_window=2441` for 10M at 4096.
- `metrics/mean_log10_feature_sparsity` and its histogram: every `feature_sampling_window` (1000) steps.
- `details/current_learning_rate`.
- Added by `_SaevMetricsMixin`, logged at the same steps (every `wandb_log_frequency`), same definitions as saev:
  - `metrics/normalized_mse`, `metrics/sse_sae`, `metrics/sse_baseline`: current train batch, SSE about the batch mean (saev's train-loop `metrics/normalized_mse`).
  - `val/normalized_mse`, `val/l0`: a fixed val set (`val_n_batches` = 4 batches, 16,384 tokens at batch 4096, loaded once from the val shards) via `evaluate_sae`. Costs a few percent of step time; `--set val_n_batches=0` turns it off, higher gives a less noisy curve.

Eval: `train_prisma.py` runs `evaluate_sae` on val after each run and writes it to `results.jsonl`. Judge runs by these:
- `normalized_mse` = SSE(x - x_hat) / SSE(x - mean) over all eval tokens. Same definition as saev's `eval/normalized_mse`, so the numbers are comparable to the saev runs on the same val shards.
- `l0`, `n_dead`/`dead_pct` (never fired in the eval set), `n_dense`/`dense_pct` (fires on >1% of tokens, same threshold as saev).
- Default eval is 50 batches (~205k tokens at batch 4096), so rare latents look dead more often than in saev's 10M-token val. Raise `--eval-batches` for reported numbers.
- Same rule as saev: a run is only better if it wins `normalized_mse` at matched `l0`; then compare dead/dense.
- Don't use Prisma's `SparsecoderEval`. It is ImageNet-only zero-shot CE (meaningless here, see top), it crashes for non-DINO models with `is_clip=False`, and it doesn't compute explained variance.
- Rough targets (from the saev notes, not guarantees): normalized_mse < 0.15 good, 0.15-0.25 usable, > 0.3 poor (k=64, 16x).

## Starting config and what to iterate on
- Start: route `shards`, TopK k=64, 16x (d_sae 26624), lr 1e-3, batch 4096, `--total-tokens` = sharded tokens (<= ~1 epoch). For 128M tokens that is 31,250 steps.
- Sweep in this order:
  1. lr {3e-4, 1e-3} first, since Prisma's loss scale differs from saev's
  2. k {32, 64, 128}
  3. expansion {16, 32}
- High dead fraction (no AuxK here): lower lr or raise batch size, or try `relu` + L1 with ghost grads. If it stays high, that's a reason to prefer saev's trainer for this model.
- Poor MSE: raise k or expansion.
- Compare with saev: same val shards, `normalized_mse` at matched `l0`.
- Real test: swap SAE reconstructions for true tokens in the downstream pipeline and measure degradation.
- Example: `uv run python train_prisma.py --route shards --train-shards $SHARDS_ROOT/$TRAIN_HASH --val-shards $SHARDS_ROOT/$VAL_HASH --saev-layer 48 --ckpt-dir $RUNS_DIR --total-tokens 128000000 --sweep sweeps/first_sweep.py`

## Known rough edges in Prisma (at 64daf14; the code here avoids all of them)
- Cached-activation loader rereads `0.pt` forever. The cache writer crashes with `use_patches_only` or `cls_token_only`.
- Bare default config (topk, no k) fails an assert.
- On-the-fly buffer with `use_patches_only` and `context_size=257` crashes.
- `layer_norm` input normalization vs raw-space geometric-median `b_dec`.
- `log_to_wandb=False` with checkpoints crashes at the first checkpoint.
- `VisionSAETrainer.load_dataset` works only for `imagenet1k`/`cifar10`.
- `val()` is unreachable. `n_validation_runs`, `from_pretrained_path` (so no resume) and `dead_feature_threshold` are unused.
- `load_hooked_model(pretrained=False)` crashes (`state_dict` undefined). `cfg.num_patch` raises (no `import math`). A list-valued `hook_point_layer` is half-wired. Saved transcoders can't be reloaded.
- Repo demos/tests are mostly stale. `demos/2_Train_SAE.ipynb` (on-the-fly CLIP-B/32, ImageNet) does run: tested on CPU with random weights, it trained, checkpointed and reloaded.


## Arnav's Personalization
- Any code you write, summarize in the chat what it does and why
- Any code that runs for a non trivial amoutn of time, add progres bars or print statements in the log to track progress


## Found on the first GPU run
- `load_hooked_model` calls `move_model_modules_to_device()`, which uses `model.cfg.device` (defaults to cpu) and moves the patch embedding back to CPU after `model.to(device)`. Crash: "Input type (cuda) and weight type (cpu)". We pass `move_to_device=False` in `check_tokens_prisma.py` and the images route of `train_prisma.py`. Not seen on the CPU fixture.
- The cached bigG safetensors loads offline via `$MODEL_LOCAL` (see `env.sh`), no .bin download needed.

## Analysis (`analyze_sae.py`, `slurm/analyze.sbatch`)
- Best model so far: `$RUNS_DIR/full_tied/5dc3a416-bigg-prisma-sae/n_images_496128.pt` (TopK k=128, 16x, tied init `encoder_transpose_decoder`, batch 16384, 127M tokens; val nmse 0.219, 0.0% dead). The k=64 base of the same sweep: nmse 0.271. Tied init was the big win over the first sweep (93% dead -> ~0%).
- `analyze_sae.py` encodes the val shards once (no bigG), then writes per-latent stats, top-K (activation, image, patch), `summary.json`, `frequency_hist.png` and heatmap grids under `--out`. Prisma's own eval/feature tools are ImageNet-oriented and partly broken (`highest_activating_tokens` is commented out), so they are not used.
- Prisma's `load_from_pretrained` needs `current_cfg={"_device": ...}` to load on another device; `load_sae` handles it.
- Re-render other latents without re-encoding: `sbatch --export=ALL,SKIP_ENCODE=1,FEATURES=12+345 slurm/analyze.sbatch`.
- Grids: 5 latents per PNG, 8 images per latent, 20 latents per group = 4 PNGs per group (`<group>_0..3.png`). Heat is drawn only on patches with activation >= `--heat-floor` (0.1) x the latent's max; weaker patches show the plain image (before, zero-activation patches were drawn in jet's dark blue at 55% alpha and tinted the whole image purple).
- Re-rendering needs no GPU: `sbatch --export=ALL,CKPT=...,OUT=<analysis dir with features.npz> slurm/rerender_cpu.sbatch` (stage 2 only, CPU job; `FEATURES=12+345` for extra latents).

## MindEye2 prior_out (`apply_sae.py`, `slurm/apply_sae.sbatch`)
- Applies a Prisma SAE to MindEye2's cached `base_all_prior_out.pt` ((1000, 256, 1664) fp16, raw, never resampled) token by token, in order, and writes `<cond>_all_prior_out.pt` (fp16, same shape/row order), `<cond>_image_idx.npy` and `<cond>_sae_record.json` to `~/orcd/scratch/mindeye/outputs/<cond><wandb run id>/`. Never writes into `base/`. `--oracle` applies it to `base_clip_target.pt` (cond `sae_pre_oracle`).
- Built-in checks (asserts): output finite and (1000, 256, 1664); flat per-image cosine retrieval of x from SAE(x) >= 99% (row order); encode+decode == `sae.forward`; re-running 16k tokens with another batch size changes nothing beyond fp rounding.
- Compute is fp32 with TF32 off. The W&B run id is looked up by the checkpoint folder name (= run display name); `--wandb-run-id` overrides. `--dir-sep` puts text between `<cond>` and the id (spec: none).
- 8-image CPU smoke test (k=128, 16x SAE): nmse 0.177, token cosine 0.936, output token norm 0.936 of input (46.2 -> 43.3), order retrieval 100%.
