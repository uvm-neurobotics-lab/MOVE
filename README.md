# MOVE – Many-objective Optimization via Voting for Elites

MOVE is a neuroevolution system for image generation that combines CPPN-based
genotypes with many-objective optimisation and gradient-based fine-tuning.
The algorithm was introduced by Jackson Dean and Nick Cheney in the paper
[*Many-objective Optimization via Voting for Elites*](https://arxiv.org/abs/2307.02661).

This repository contains the research implementation together with ready-to-run
configurations, CLIP-based semantic objectives, and utilities for analysing
experiment output.

---

## Highlights

- **CPPN evolution** with configurable activation libraries and topology
  mutation rates.
- **Many-objective optimization** across perceptual metrics (LPIPS, DISTS,
  SSIM, style/content, …) and CLIP similarity.
- **Optional SGD fine-tuning** (AdamW) with early stopping per genome.
- **Feature caching & AMP** support for faster CLIP/perceptual evaluation.
- **Reproducible experiment driver** powered by JSON configuration files.

---

## Requirements

- Python 3.10 or newer (3.11 recommended).
- CUDA-capable GPU (MOVE will fall back to CPU but runs considerably slower).
- Conda is recommended for reproducing the paper environment.

PyTorch, TorchVision, LPIPS, and other dependencies are pinned inside
`environment.yml`.

---

## Quickstart

1. **Create and activate the environment** (once):

   ```bash
   conda env create -f environment.yml
   conda activate move
   ```

2. **Run MOVE with the default configuration**
   ```bash
   python -m MOVE --config default.json
   ```

   Output is written to `output/default/<run-id>/` with intermediate checkpoints
   and fitness logs.

3. **Swap in your own target image** while reusing the default settings:

   ```bash
   python -m move --config default.json --target path/to/your_image.png
   ```

4. **Run CLIP-guided text objectives** with the handy starter config:

   ```bash
   python -m MOVE --config clip-test.json
   ```

   CLIP embeddings and partial prompts are generated automatically; results are
   saved under `output/clip-test/<run-id>/`.

At any time you can inspect the available options:

```bash
python -m MOVE --help
```

---

## Command-line reference

| Flag | Description |
| --- | --- |
| `-c`, `--config` | Path to a JSON configuration (defaults to `default.json`). |
| `-t`, `--target` | Override the target image or CLIP text prompt directly from the CLI. |
| `-g`, `--generations` | Force a number of generations (overrides config). |
| `-p`, `--population` | Override population size (`num_cells`, `num_children`). |
| `-o`, `--output` | Custom output directory root. |
| `-d`, `--device` | Torch device string (default: first CUDA GPU if available). |
| `-sgd`, `--sgd` | Set the number of SGD fine-tuning steps per generation. |
| `-ff`, `--num_fourier_features` | Enable Fourier feature inputs with the given count. |
| `-hn`, `--num_hidden_nodes` | Override initial hidden node count. |
| `-pr`, `--profile` | Capture a cProfile trace for the run. |
| `-pl`, `--parallel` | Launch evolutionary batches on background threads (experimental). |
| `-r`, `--resume` | Resume a previous run directory (expects `config.json`). |
| `-sc`, `--stop-condition` | Override the stopping criterion (`evals`, `batches`, `passes`, …). |
| `-sv`, `--stop-value` | Stop once the chosen condition reaches this value. |

All flags map onto the JSON configuration schema; command-line values win over
file-based values.

---

## Configuration files

MOVE experiments are described in JSON. The minimal structure contains a
`controls` object for global defaults and an optional list of `conditions` for
batched sweeps:

```json
{
  "name": "MOVE",
  "controls": {
    "target": "data/apple.png",
    "sgd_steps": 10,
    "num_cells": 25,
    "stop_condition": "passes",
    "stop_condition_value": 50000
  },
  "conditions": [
    {"default": {"sgd_strat": "sgd"}}
  ]
}
```

- `controls` keys correspond to attributes on `MOVEConfig`
  (see `MOVE/move_config.py` for the full catalogue).
- Each entry in `conditions` defines a named variant and the overrides that
  should apply when running that variant.
- When a run starts, MOVE materializes `output/<condition-name>/<run-id>/` and
  persists checkpoints, fitness tables, rendered images, and lineage metadata.

Helpful starter configs:

- `default.json` – replicates the paper’s Apple image reconstruction target.
- `clip-test.json` – runs CLIP text-to-image optimization with partial prompts.
- `clip-test/` – directory containing example CLIP experiments and targets.

---

## Using CLIP objectives

MOVE’s CLIP integration lets you optimize CPPNs directly against text prompts.
When `clip_text_target` is present (either in a config or via
`python -m move --target "your prompt"`), the traditional image-based
objectives are replaced with a bundle of CLIP similarity objectives:

1. MOVE samples one or more noisy text embeddings with
   `clip_num_variants` (defaults to 1). Each variant becomes an objective.
2. If `clip_include_partials` is `True`, MOVE extracts distinct keywords from
   the prompt and tracks them as additional objectives.
3. During evolution the map contains a cell for each objective, encouraging
   diverse imagery that satisfies different parts of the prompt.

Key configuration knobs:

| Field | Purpose |
| --- | --- |
| `clip_text_target` | Base prompt string. Set automatically when you pass a string to `--target`. |
| `clip_num_variants` | Number of noisy re-embeddings to use (diversifies CLIP guidance). |
| `clip_noise_scale` | Magnitude of Gaussian noise applied to each variant. |
| `clip_noise_anneal` & friends | Enables annealing the noise scale during the run (`clip_noise_final_scale`, `clip_noise_anneal_start`, `clip_noise_anneal_end`, `clip_noise_anneal_power`). |
| `clip_include_partials` | Enable token-level objectives based on the prompt. |
| `clip_disable_partials` | Force-disable token-level objectives (even if `clip_include_partials` is `True`). Useful when you provide a list of full prompts and want MOVE cells to combine only full-prompt objectives. |
| `clip_partial_min_length` | Minimum character length for partial prompts. |
| `clip_partial_stopwords` | Words to ignore when extracting partial prompts (defaults to a short English list). |
| `clip_max_partial_prompts` | Hard cap on the number of partial prompts. |
| `clip_microbatch_size` | Number of images to embed at once when generating CLIP activations (helps with GPU memory). |
| `clip_augmentations` | Number of additional random crops per candidate used when embedding for CLIP. Averaged scores make guidance less brittle (defaults to 4). |
| `clip_aug_min_scale` / `clip_aug_max_scale` | Range of crop scales sampled for each extra view (fraction of the original image size). |
| `clip_aug_flip_prob` | Probability of applying a horizontal flip to each augmented view. |
| `clip_aug_jitter_std` | Standard deviation of additive Gaussian noise applied to augmented views (in `[0,1]`). |
| `clip_random_seed` | Seed for reproducible embedding variants. |

You can also add CLIP-driven scores to standard MOVE runs (with a fixed image
target) by mixing the new fitness functions into your `objective_functions`
list:

- `"clip_similarity"` – compares each candidate’s CLIP embedding to the CLIP
  embedding of the current target image.
- `"clip_a-red-dog"`, `"clip_space-ship"`, etc. – any string beginning with
  `clip_` creates a text objective where hyphens/underscores become spaces. The
  example `clip_a-red-dog` evaluates cosine similarity to the phrase “a red dog”.

These objectives sit alongside existing metrics such as MSE or PSNR, making it
easy to blend low-level reconstruction with high-level semantics without
enabling the full CLIP mode.

Notes:

- The CLIP objectives rely on the OpenAI `clip` Python package. It’s already
  included in `environment.yml`; if you run outside Conda, install it with
  `pip install git+https://github.com/openai/CLIP.git`.
- Feature caching and automatic mixed precision (AMP) are active by default to
  keep CLIP scoring efficient. You can adjust `clip_microbatch_size` to control
  the embedding batch size, and `use_amp` to disable mixed precision if you run
  into stability issues. If prompts feel unstable, increase `clip_augmentations`
  or narrow the crop range (`clip_aug_min_scale` closer to `clip_aug_max_scale`) to
  smooth CLIP feedback at the cost of a small runtime increase.
- Per-objective results (full variants, partial prompts, and their scores) are
  saved alongside other fitness logs in the run directory. This makes it easy to
  inspect which prompts each elite specialized for.

See `clip-test.json` for a compact example that exercises most of the CLIP
options.

---

## Outputs & artifacts

Each run directory contains:

- `config.json` – the fully-resolved configuration used for the run.
- `lineages.json` – parentage data per MAP-Elites cell.
- `checkpoints/` – serialized evolutionary state for resuming.
- `images/` – periodic renderings of elites.
- `metrics/` – CSV and NumPy dumps of raw and normalized fitness history.

Utility scripts in `scripts/` and notebooks in `analysis/` demonstrate how to
visualize these artifacts.

---


## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| Immediate CUDA OOM | Lower `batch_size`/`num_cells` or disable `use_amp` in the config. |
| CLIP metrics stay at 0 | Ensure the run is using a text config (`clip_text_target` or `--target "your prompt"`). |
| Run exits with “target not found” | Supply `--target path/to/img.png` or edit the config’s `target`. |
| Resume fails to load checkpoint | Pass the original run directory to `--resume`; MOVE will ingest `config.json` automatically. |

If you bump into an unexpected issue, enable verbose logs with `--verbose` and
open an issue with the resulting stack trace plus your configuration.

---

## Citation

If you use MOVE in research, please cite the original paper:

```
@inproceedings{dean2023move,
  title={Many-objective Optimization via Voting for Elites},
  author={Dean, Jackson and Cheney, Nick},
  booktitle={GECCO},
  year={2023}
}
```
