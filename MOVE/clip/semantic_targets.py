"""Utilities for generating CLIP text embeddings and prompt variants.

This module encapsulates the logic used by MOVE to build sets of CLIP targets
from a single prompt.  The workflow is shared by the evolutionary loop and the
CLI helpers:

1. Embed the base prompt with both ViT-B/32 and RN50 CLIP encoders.
2. Sample optional noisy variants that encourage exploration of nearby concepts.
3. Extract partial prompts (keywords) so individual MAP-Elites cells can focus
    on specific aspects of the text description.

The helpers defined here are intentionally lightweight and side-effect free so
they can be reused by notebooks and analysis scripts without pulling in the
entire MOVE stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from .clip_model import embed_text

DEFAULT_STOP_WORDS = frozenset({"a", "an", "the", "of", "and", "or", "with", "that", "on", "in", "for", "to"})


@dataclass(frozen=True)
class ClipSemanticConfig:
    """Configuration for sampling CLIP text embeddings.

    Attributes
    ----------
    text:
        Prompt text to encode.
    num_variants:
        Number of additional noisy embeddings (including the base embedding).
        Values <= 1 disable noise sampling.
    noise_scale:
        Standard deviation of the Gaussian direction added to the base
        embedding for each variant.  Higher values encourage more exploration
        but can reduce fidelity.
    seed:
        Optional random seed for deterministic sampling.
    device:
        Desired device on which to place the resulting embeddings.
    """

    text: str
    num_variants: int
    noise_scale: float = 0.2
    seed: Optional[int] = None
    device: Union[torch.device, str, None] = None


def _rng(config: ClipSemanticConfig) -> torch.Generator:
    """Return a deterministic CPU RNG based on the config seed."""

    generator = torch.Generator(device="cpu")
    if config.seed is not None:
        generator.manual_seed(config.seed)
    return generator


def _sample_noise_like(reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Draw Gaussian noise with the same shape as ``reference``."""

    noise = torch.randn(
        reference.shape,
        generator=generator,
        dtype=reference.dtype,
        device="cpu",
    )
    return noise.to(reference.device)


@torch.no_grad()
def generate_clip_targets(config: ClipSemanticConfig) -> List[torch.Tensor]:
    """Generate the base CLIP embedding plus optional noisy variants.

    The base embedding favours deterministic behaviour; variants are sampled by
    adding orthogonal Gaussian perturbations scaled by ``config.noise_scale``.
    Each embedding is normalised to unit length.
    """

    base_embedding = embed_text(config.text, device=config.device).float()
    base_embedding = torch.nn.functional.normalize(base_embedding, dim=0)
    targets: List[torch.Tensor] = [base_embedding]
    generator = _rng(config)

    if config.num_variants <= 1:
        return targets

    for _ in range(config.num_variants - 1):
        noise = _sample_noise_like(base_embedding, generator)
        noise = noise - torch.dot(noise, base_embedding) * base_embedding
        if torch.allclose(noise, torch.zeros_like(noise)):
            noise = _sample_noise_like(base_embedding, generator)
        noise = torch.nn.functional.normalize(noise, dim=0)
        perturbed = base_embedding + config.noise_scale * noise
        perturbed = torch.nn.functional.normalize(perturbed, dim=0)
        targets.append(perturbed)
    return targets


def generate_partial_prompts(
    text: str,
    *,
    min_length: int = 3,
    stop_words: Optional[set[str]] = None,
    n_tokens: int = 1,
    max_partial_prompts: Optional[int] = None,
) -> List[str]:
    """Derive canonical partial prompts from ``text``.

    Tokens shorter than ``min_length`` or appearing in ``stop_words`` are
    filtered out. ``n_tokens`` controls how many tokens are joined for each
    partial (e.g. 1 => single token, 2 => bigrams). Uniqueness is enforced at
    the full partial level (case-insensitive) while preserving original casing
    from the first occurrence.
    """

    min_length = max(1, int(min_length))
    n_tokens = max(1, int(n_tokens))
    normalized_stop_words = None
    if stop_words:
        normalized_stop_words = {word.lower() for word in stop_words}

    filtered_tokens: List[str] = []
    for match in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text):
        token = match.group(0).strip("'\"")
        lower_token = token.lower()
        if len(lower_token) < min_length:
            continue
        if normalized_stop_words and lower_token in normalized_stop_words:
            continue
        filtered_tokens.append(token)

    if not filtered_tokens:
        return []

    if n_tokens == 1:
        span_tokens: List[List[str]] = [[tok] for tok in filtered_tokens]
    else:
        if len(filtered_tokens) < n_tokens:
            return []
        span_tokens = [filtered_tokens[i : i + n_tokens] for i in range(len(filtered_tokens) - n_tokens + 1)]

    seen = set()
    partials: List[str] = []
    for span in span_tokens:
        key = " ".join(token.lower() for token in span)
        if key in seen:
            continue
        seen.add(key)
        partials.append(" ".join(span))
        if max_partial_prompts is not None and len(partials) >= max_partial_prompts:
            break
    return partials
