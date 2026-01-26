"""Shared CLIP model utilities for MOVE.

These helpers lazily load the OpenAI CLIP models (ViT-B/32 and RN50), keep them
frozen for inference, and expose convenience functions for embedding text and
images.  Keeping the loading logic centralised avoids repeated downloads and
ensures consistent preprocessing across objectives.
"""

import functools
import logging
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
_CLIP_COMPILED = {}
_CLIP_COMPILE_ACTIVE_KEY = None
_CLIP_MODEL_NAMES = ("ViT-B/32", "RN50")


def _canonical_device(device: Union[torch.device, str, None]) -> torch.device:
    """Return a concrete :class:`torch.device` for ``device`` or a sensible default."""

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_models(device: torch.device) -> Tuple["clip.model.CLIP", "clip.model.CLIP"]:
    """Load and cache the ViT-B/32 and RN50 CLIP encoders on ``device``."""

    vit_name, rn50_name = _CLIP_MODEL_NAMES
    key = (str(device), vit_name, rn50_name)
    if key not in _CLIP_MODELS:
        vit, _ = clip.load(vit_name, device=device, jit=False)
        rn50, _ = clip.load(rn50_name, device=device, jit=False)
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


def set_clip_model_names(vit_name: str, rn50_name: str) -> None:
    """Update CLIP model names and clear cached models/compilations if changed."""

    global _CLIP_MODEL_NAMES, _CLIP_MODELS, _CLIP_COMPILED, _CLIP_COMPILE_ACTIVE_KEY
    vit_name = str(vit_name)
    rn50_name = str(rn50_name)
    if _CLIP_MODEL_NAMES == (vit_name, rn50_name):
        return
    _CLIP_MODEL_NAMES = (vit_name, rn50_name)
    _CLIP_MODELS = {}
    _CLIP_COMPILED = {}
    _CLIP_COMPILE_ACTIVE_KEY = None


def maybe_compile_clip_models(
    config,
    device: Union[torch.device, str, None] = None,
) -> None:
    """Optionally torch.compile CLIP encoders for faster inference."""

    if not getattr(config, "clip_compile_models", False):
        return
    if not hasattr(torch, "compile"):
        logging.info("torch.compile unavailable; skipping CLIP compilation.")
        return

    vit, rn50, resolved_device = get_clip_models(device)
    mode = str(getattr(config, "clip_compile_mode", "reduce-overhead"))
    backend = getattr(config, "clip_compile_backend", None)
    dynamic = bool(getattr(config, "clip_compile_dynamic", False))
    fullgraph = bool(getattr(config, "clip_compile_fullgraph", False))
    key = (str(resolved_device), mode, backend, dynamic, fullgraph)
    global _CLIP_COMPILE_ACTIVE_KEY
    if key in _CLIP_COMPILED:
        _CLIP_COMPILE_ACTIVE_KEY = key
        return

    compiled = {}

    logging.info(
        "Compiling CLIP encoders (mode=%s, backend=%s, dynamic=%s, fullgraph=%s)",
        mode,
        backend,
        dynamic,
        fullgraph,
    )

    def _compile(name, fn):
        try:
            if backend in ("aot_eager", "nvfuser"):
                compiled_fn = torch.compile(fn, dynamic=dynamic, fullgraph=fullgraph, backend=backend)
            else:
                compiled_fn = torch.compile(fn, dynamic=dynamic, mode=mode, fullgraph=fullgraph, backend=backend)
            compiled[name] = compiled_fn
        except Exception as exc:
            logging.warning("Failed to compile CLIP %s; using eager. Error: %s", name, exc)

    _compile("vit_text", vit.encode_text)
    _compile("rn50_text", rn50.encode_text)
    _compile("vit_image", vit.encode_image)
    _compile("rn50_image", rn50.encode_image)

    _CLIP_COMPILED[key] = compiled
    _CLIP_COMPILE_ACTIVE_KEY = key

    if compiled:
        logging.info("CLIP compilation complete: %s", ", ".join(sorted(compiled.keys())))
    else:
        logging.warning("CLIP compilation produced no compiled functions; using eager encoders.")


def _get_compiled(device: torch.device, config) -> dict:
    if config is not None:
        if not getattr(config, "clip_compile_models", False):
            return {}
        mode = str(getattr(config, "clip_compile_mode", "reduce-overhead"))
        backend = getattr(config, "clip_compile_backend", None)
        dynamic = bool(getattr(config, "clip_compile_dynamic", False))
        fullgraph = bool(getattr(config, "clip_compile_fullgraph", False))
        key = (str(device), mode, backend, dynamic, fullgraph)
        return _CLIP_COMPILED.get(key, {})
    if _CLIP_COMPILE_ACTIVE_KEY is None:
        return {}
    if _CLIP_COMPILE_ACTIVE_KEY[0] != str(device):
        return {}
    return _CLIP_COMPILED.get(_CLIP_COMPILE_ACTIVE_KEY, {})


@functools.lru_cache(maxsize=None)
def tokenize_text(text: str) -> torch.Tensor:
    """Tokenize ``text`` once and cache the resulting IDs on the default device."""

    _, _, device = get_clip_models(None)
    return clip.tokenize(text).to(device)


@torch.no_grad()
def embed_text(text: str, device: Union[torch.device, str, None] = None, config=None) -> torch.Tensor:
    """Encode ``text`` with both CLIP encoders and concatenate the features."""

    vit, rn50, resolved_device = get_clip_models(device)
    tokenized = clip.tokenize(text).to(resolved_device)
    compiled = _get_compiled(resolved_device, config) if config is not None else {}
    vit_fn = compiled.get("vit_text", vit.encode_text)
    rn50_fn = compiled.get("rn50_text", rn50.encode_text)
    vit_features = vit_fn(tokenized)
    rn50_features = rn50_fn(tokenized)
    combined = torch.cat([vit_features, rn50_features], dim=-1)
    combined = combined.to(dtype=torch.float32)
    return combined.squeeze(0)


def embed_images(images: torch.Tensor, device: Union[torch.device, str, None] = None, config=None) -> torch.Tensor:
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
    compiled = _get_compiled(resolved_device, config) if config is not None else {}
    vit_fn = compiled.get("vit_image", vit.encode_image)
    rn50_fn = compiled.get("rn50_image", rn50.encode_image)
    vit_features = vit_fn(normalized)
    rn50_features = rn50_fn(normalized)
    combined = torch.cat([vit_features, rn50_features], dim=-1)
    return combined.to(dtype=torch.float32)


def cosine_similarity(image_embeddings: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
    """Return CLIP cosine similarity scores for image and text embeddings."""

    if text_embedding.ndim == 1:
        text_embedding = text_embedding.unsqueeze(0)
    image_norm = torch.nn.functional.normalize(image_embeddings, dim=-1)
    text_norm = torch.nn.functional.normalize(text_embedding, dim=-1)
    return torch.matmul(image_norm, text_norm.T).squeeze(-1)
