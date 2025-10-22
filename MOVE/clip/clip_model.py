"""Shared CLIP model utilities for MOVE.

These helpers lazily load the OpenAI CLIP models (ViT-B/32 and RN50), keep them
frozen for inference, and expose convenience functions for embedding text and
images.  Keeping the loading logic centralised avoids repeated downloads and
ensures consistent preprocessing across objectives.
"""

import functools
from typing import Tuple, Union

import torch
import torch.nn.functional as F

try:
    import clip  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "The 'clip' package is required for CLIP-based objectives. Install it with `pip install git+https://github.com/openai/CLIP.git`."
    ) from exc


_CLIP_MODELS = {}


def _canonical_device(device: Union[torch.device, str, None]) -> torch.device:
    """Return a concrete :class:`torch.device` for ``device`` or a sensible default."""

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_models(device: torch.device) -> Tuple["clip.model.CLIP", "clip.model.CLIP"]:
    """Load and cache the ViT-B/32 and RN50 CLIP encoders on ``device``."""

    key = str(device)
    if key not in _CLIP_MODELS:
        vit, _ = clip.load("ViT-B/32", device=device, jit=False)
        rn50, _ = clip.load("RN50", device=device, jit=False)
        vit.eval()
        rn50.eval()
        for param in vit.parameters():
            param.requires_grad_(False)
        for param in rn50.parameters():
            param.requires_grad_(False)
        _CLIP_MODELS[key] = (vit, rn50)
    return _CLIP_MODELS[key]


def get_clip_models(device: Union[torch.device, str, None] = None) -> Tuple["clip.model.CLIP", "clip.model.CLIP", torch.device]:
    """Return the cached CLIP models and the resolved device."""

    resolved_device = _canonical_device(device)
    vit, rn50 = _load_models(resolved_device)
    return vit, rn50, resolved_device


@functools.lru_cache(maxsize=None)
def tokenize_text(text: str) -> torch.Tensor:
    """Tokenize ``text`` once and cache the resulting IDs on the default device."""

    _, _, device = get_clip_models(None)
    return clip.tokenize(text).to(device)


@torch.no_grad()
def embed_text(text: str, device: Union[torch.device, str, None] = None) -> torch.Tensor:
    """Encode ``text`` with both CLIP encoders and concatenate the features."""

    vit, rn50, resolved_device = get_clip_models(device)
    tokenized = clip.tokenize(text).to(resolved_device)
    vit_features = vit.encode_text(tokenized)
    rn50_features = rn50.encode_text(tokenized)
    combined = torch.cat([vit_features, rn50_features], dim=-1)
    combined = combined.to(dtype=torch.float32)
    return combined.squeeze(0)


def embed_images(images: torch.Tensor, device: Union[torch.device, str, None] = None) -> torch.Tensor:
    """Embed ``images`` (NCHW, [0,1]) with CLIP and concatenate encoder features."""

    vit, rn50, resolved_device = get_clip_models(device)
    if images.device != resolved_device:
        images = images.to(resolved_device)
    images = images.clamp(0.0, 1.0)
    if images.shape[-2:] != (224, 224):
        images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=resolved_device, dtype=images.dtype).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=resolved_device, dtype=images.dtype).view(1, 3, 1, 1)
    normalized = (images - clip_mean) / clip_std
    vit_features = vit.encode_image(normalized)
    rn50_features = rn50.encode_image(normalized)
    combined = torch.cat([vit_features, rn50_features], dim=-1)
    return combined.to(dtype=torch.float32)


def cosine_similarity(image_embeddings: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
    """Return CLIP cosine similarity scores for image and text embeddings."""

    if text_embedding.ndim == 1:
        text_embedding = text_embedding.unsqueeze(0)
    image_norm = torch.nn.functional.normalize(image_embeddings, dim=-1)
    text_norm = torch.nn.functional.normalize(text_embedding, dim=-1)
    return torch.matmul(image_norm, text_norm.T).squeeze(-1)
