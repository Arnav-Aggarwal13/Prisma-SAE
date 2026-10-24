"""Rough number of val images each latent fires on, from the saved features.npz of analyze_sae.py (no model, no GPU).

  uv run python image_count_estimate.py DIR [DIR ...]      # DIR = an analysis dir holding features.npz

features.npz only has token-level frequency, so an image count can only be bounded:
  lower = ceil(tokens_fired / 256)  (every image the latent fires in has it on all 256 patches)
  upper = min(tokens_fired, n_images)  (one firing patch per image)
  est   = tokens_fired / (64 / distinct images among the latent's top-64 tokens), clipped to [lower, upper]. Heuristic: the strongest
          firings cluster inside images more than weak ones do, so this leans low. The exact count needs a re-encode pass.
Writes DIR/image_count_bounds.npz (per latent: tokens_fired, lower, upper, est, n_img_top64) and prints percentile tables.
"""

import sys

import numpy as np

N_IMG, N_TOK, PATCHES = 19901, 5094656, 256  # val set: 19,901 images x 256 patches


def main(dirs: list[str]):
    assert dirs, __doc__
    for d in dirs:
        z = np.load(f"{d}/features.npz")
        assert int(z["n_tok"]) == N_TOK, f"{d}: n_tok {int(z['n_tok'])} != {N_TOK}; not a full-val analysis"
        fired = np.rint(z["freq"] * N_TOK)
        lower, upper = np.ceil(fired / PATCHES), np.minimum(fired, N_IMG)
        n_top = np.array([len(set(z["top_ex"][:, f])) for f in range(z["top_ex"].shape[1])])
        est = np.clip(fired / (z["top_ex"].shape[0] / n_top), lower, upper)
        np.savez(f"{d}/image_count_bounds.npz", tokens_fired=fired, lower=lower, upper=upper, est=est, n_img_top64=n_top)
        pct = lambda a: "  ".join(f"{v:8.0f}" for v in np.percentile(a, [10, 50, 90]))
        print(f"== {d}  ({len(fired)} latents)  columns: p10 / p50 / p90 over latents", flush=True)
        print(f"  tokens fired       {pct(fired)}")
        print(f"  images, lower      {pct(lower)}")
        print(f"  images, estimate   {pct(est)}   ({100 * np.median(est) / N_IMG:.1f}% of val images at the median)")
        print(f"  images, upper      {pct(upper)}")
        print(f"  distinct imgs in top-64 tokens  {pct(n_top)}")


if __name__ == "__main__":
    main(sys.argv[1:])
