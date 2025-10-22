"""CLIP similarity objectives used by MOVE's evolutionary loop.

The helpers in this module expose lightweight wrappers that turn frozen CLIP
embeddings into callable fitness objectives compatible with the MOVE fitness
API.  They support mixed-precision micro-batching and integrate with the global
feature cache to avoid redundant CLIP forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import math

import torch

import torch.nn.functional as F

from .clip_model import embed_images, cosine_similarity
from ..fitness.feature_cache import cached_result
@dataclass(frozen=True)
class ClipAugmentations:
    """Configuration for CLIP image augmentations to stabilize similarity scores."""

    num_random_crops: int = 0
    min_crop_scale: float = 0.7
    max_crop_scale: float = 1.0
    flip_prob: float = 0.0
    jitter_std: float = 0.0

    def __post_init__(self) -> None:  # type: ignore[override]
        object.__setattr__(self, "num_random_crops", max(0, int(self.num_random_crops)))
        min_scale = float(self.min_crop_scale)
        max_scale = float(self.max_crop_scale)
        if not math.isfinite(min_scale) or min_scale <= 0:
            min_scale = 0.1
        if not math.isfinite(max_scale) or max_scale <= 0:
            max_scale = 1.0
        if max_scale < min_scale:
            max_scale = min_scale
        object.__setattr__(self, "min_crop_scale", min_scale)
        object.__setattr__(self, "max_crop_scale", max_scale)
        flip = float(self.flip_prob)
        object.__setattr__(self, "flip_prob", min(1.0, max(0.0, flip)))
        jitter = float(self.jitter_std)
        if not math.isfinite(jitter) or jitter < 0.0:
            jitter = 0.0
        object.__setattr__(self, "jitter_std", jitter)


@dataclass(eq=False)
class ClipSimilarityObjective:
    """Callable CLIP objective that scores candidates against a text embedding."""

    embedding: torch.Tensor
    identifier: str
    microbatch_size: int = 0
    augmentations: Optional[ClipAugmentations] = None

    def __post_init__(self) -> None:
        if not isinstance(self.embedding, torch.Tensor):
            raise TypeError("embedding must be a torch.Tensor")
        if self.embedding.ndim != 1:
            raise ValueError("embedding must be a 1D tensor")
        if isinstance(self.identifier, str):
            self.__name__ = self.identifier
        else:
            self.__name__ = "clip_similarity"
        self.embedding = torch.nn.functional.normalize(self.embedding.detach(), dim=0)
        if self.microbatch_size is None:
            self.microbatch_size = 0
        if self.microbatch_size < 0:
            raise ValueError("microbatch_size must be non-negative")

    def to(self, device: Union[torch.device, str]) -> "ClipSimilarityObjective":
        """Move the reference embedding to ``device`` and return ``self``."""

        self.embedding = self.embedding.to(device)
        return self

    def __call__(self, candidates: torch.Tensor, _unused_target: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute CLIP cosine similarity for ``candidates``.

        ``_unused_target`` is part of the generic MOVE fitness signature and is
        ignored for CLIP objectives.
        """

        image_embeddings = self._get_image_embeddings(candidates)
        return cosine_similarity(image_embeddings, self.embedding)

    def _get_image_embeddings(self, candidates: torch.Tensor) -> torch.Tensor:
        """Embed candidate images, respecting the configured micro-batch size."""

        micro = int(self.microbatch_size) if self.microbatch_size else candidates.shape[0]
        micro = max(1, micro)
        view_multiplier = 1

        if self.augmentations and self.augmentations.num_random_crops > 0:
            augmented_views = self._build_augmented_views(candidates)
            view_multiplier = len(augmented_views)

            def build_full_embeddings() -> torch.Tensor:
                stacked = torch.cat(augmented_views, dim=0)
                embeddings = embed_images(stacked, device=self.embedding.device)
                embeddings = embeddings.view(view_multiplier, candidates.shape[0], -1)
                return embeddings.mean(dim=0)

            slot = ("clip:image_embeddings/full", micro, view_multiplier)
            return cached_result(slot, candidates, build_full_embeddings)

        def build_full_embeddings() -> torch.Tensor:
            chunks: List[torch.Tensor] = []
            for chunk in candidates.split(micro):
                chunk = chunk.contiguous()
                chunk_embeddings = embed_images(chunk, device=self.embedding.device)
                chunks.append(chunk_embeddings)
            if len(chunks) == 1:
                return chunks[0]
            return torch.cat(chunks, dim=0)

        slot = ("clip:image_embeddings/full", micro, view_multiplier)
        return cached_result(slot, candidates, build_full_embeddings)

    def _build_augmented_views(self, candidates: torch.Tensor) -> List[torch.Tensor]:
        assert self.augmentations is not None
        views: List[torch.Tensor] = [candidates]
        for _ in range(self.augmentations.num_random_crops):
            views.append(self._random_resized_crop(candidates))
        return views

    def _random_resized_crop(self, images: torch.Tensor) -> torch.Tensor:
        assert self.augmentations is not None
        B, C, H, W = images.shape
        device = images.device
        result = torch.empty_like(images)
        min_scale = self.augmentations.min_crop_scale
        max_scale = self.augmentations.max_crop_scale
        for idx in range(B):
            scale = float(torch.empty(1, device=device).uniform_(min_scale, max_scale).item())
            crop_h = max(1, min(H, int(round(H * scale))))
            crop_w = max(1, min(W, int(round(W * scale))))
            if crop_h == H:
                top = 0
            else:
                top = int(torch.randint(0, H - crop_h + 1, (1,), device=device).item())
            if crop_w == W:
                left = 0
            else:
                left = int(torch.randint(0, W - crop_w + 1, (1,), device=device).item())
            crop = images[idx : idx + 1, :, top : top + crop_h, left : left + crop_w]
            resized = F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)
            result[idx] = resized[0]

        if self.augmentations.flip_prob > 0.0:
            flip_mask = torch.rand((B,), device=device) < self.augmentations.flip_prob
            if flip_mask.any():
                result[flip_mask] = torch.flip(result[flip_mask], dims=[-1])

        if self.augmentations.jitter_std > 0.0:
            noise = torch.randn_like(result) * self.augmentations.jitter_std
            result = torch.clamp(result + noise, 0.0, 1.0)

        return result.contiguous()

    def __hash__(self) -> int:
        return hash(self.identifier)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClipSimilarityObjective):
            return False
        return self.identifier == other.identifier


def build_clip_objectives(
    embeddings: Iterable[torch.Tensor],
    prefix: str = "clip",
    *,
    microbatch_size: int = 0,
    augmentations: Optional[ClipAugmentations] = None,
) -> List[ClipSimilarityObjective]:
    """Wrap each embedding in a :class:`ClipSimilarityObjective` instance."""

    objectives: List[ClipSimilarityObjective] = []
    for idx, embedding in enumerate(embeddings):
        name = f"{prefix}_{idx:02d}"
        objectives.append(
            ClipSimilarityObjective(
                embedding=embedding,
                identifier=name,
                microbatch_size=microbatch_size,
                augmentations=augmentations,
            )
        )
    return objectives
