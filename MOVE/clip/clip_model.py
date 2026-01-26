"""Shared CLIP model utilities for MOVE.

These helpers lazily load the OpenAI CLIP models (ViT-B/32 and RN50), keep them
frozen for inference, and expose convenience functions for embedding text and
images.  Keeping the loading logic centralised avoids repeated downloads and
ensures consistent preprocessing across objectives.
"""

import functools
import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

try:
    import clip  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    clip = None

try:  # Optional dependency when using OpenCLIP
    import open_clip  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    open_clip = None


_CLIP_MODELS = {}
_CLIP_COMPILED = {}
_CLIP_COMPILE_ACTIVE_KEY = None
_CLIP_MODEL_NAMES = ("ViT-B/32", "RN50")
_CLIP_PROVIDER = "openai"
_CLIP_OPENCLIP_PRETRAINED = "openai"
_CLIP_OPENCLIP_PRETRAINED_SECOND = None
_CLIP_VERBOSE = False
_OPENCLIP_TOKENIZERS = {}


def _set_openclip_logging(verbose: bool) -> None:
    if open_clip is None:
        return
    level = logging.INFO if verbose else logging.WARNING
    logger_dict = logging.Logger.manager.loggerDict
    for name, logger in logger_dict.items():
        if isinstance(logger, logging.Logger) and name.startswith("open_clip"):
            logger.setLevel(level)
            if not verbose:
                logger.propagate = False
    logging.getLogger("open_clip").setLevel(level)
    if not verbose:
        logging.getLogger("open_clip").propagate = False


def _get_openclip_tokenizer(model_name: str):
    cached = _OPENCLIP_TOKENIZERS.get(model_name)
    if cached is not None:
        return cached
    _set_openclip_logging(_CLIP_VERBOSE)
    tokenizer = open_clip.get_tokenizer(model_name)
    _OPENCLIP_TOKENIZERS[model_name] = tokenizer
    return tokenizer


def _canonical_device(device: Union[torch.device, str, None]) -> torch.device:
    """Return a concrete :class:`torch.device` for ``device`` or a sensible default."""

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _normalize_openclip_name(model_name: str) -> str:
    """Convert OpenAI-style names to OpenCLIP names (best-effort)."""

    name = str(model_name)
    if "@336px" in name:
        name = name.replace("/", "-").replace("@336px", "-336")
    else:
        name = name.replace("/", "-")
    return name


def _resolve_model_name(model_name: Optional[str], provider: str) -> Optional[str]:
    if model_name is None:
        return None
    if provider == "openclip":
        return _normalize_openclip_name(model_name)
    return model_name


def _resolve_openclip_pretrained(model_name: str, pretrained: Union[str, None]) -> Union[str, None]:
    if not pretrained:
        return None
    if open_clip is None:
        raise ImportError(
            "OpenCLIP is not installed. Install it with `pip install open_clip_torch`."
        )
    if not hasattr(open_clip, "list_pretrained"):
        return pretrained
    try:
        tags = open_clip.list_pretrained(model_name)
    except Exception:
        tags = None
    if tags is None:
        try:
            tags = open_clip.list_pretrained()
        except Exception:
            return pretrained

    available: list[str] = []
    if isinstance(tags, dict):
        available = list(tags.keys())
    elif isinstance(tags, (list, tuple)):
        for item in tags:
            if isinstance(item, str):
                if ":" in item:
                    model, tag = item.split(":", 1)
                    if model == model_name:
                        available.append(tag)
                else:
                    available.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                if item[0] == model_name:
                    available.append(item[1])

    if not available:
        logging.warning(
            "No pretrained tags found for OpenCLIP model '%s'; using default weights.",
            model_name,
        )
        return None

    if available and pretrained not in available:
        unique = sorted(set(available))
        logging.warning(
            "OpenCLIP pretrained tag '%s' not found for model '%s'. Available tags: %s",
            pretrained,
            model_name,
            ", ".join(unique),
        )
        fallback = unique[0]
        logging.warning("Defaulting OpenCLIP pretrained tag to '%s'.", fallback)
        return fallback
    return pretrained


def _load_models(device: torch.device) -> Tuple[object, Optional[object]]:
    """Load and cache the ViT-B/32 and RN50 CLIP encoders on ``device``."""

    vit_name, second_name = _CLIP_MODEL_NAMES
    provider = _CLIP_PROVIDER
    pretrained = _CLIP_OPENCLIP_PRETRAINED
    pretrained_second = _CLIP_OPENCLIP_PRETRAINED_SECOND
    resolved_vit = _resolve_model_name(vit_name, provider)
    resolved_second = _resolve_model_name(second_name, provider)
    key = (str(device), provider, pretrained, pretrained_second, resolved_vit, resolved_second)
    if key not in _CLIP_MODELS:
        if provider == "openclip":
            if open_clip is None:
                raise ImportError(
                    "OpenCLIP is not installed. Install it with `pip install open_clip_torch`."
                )
            resolved_pretrained = _resolve_openclip_pretrained(resolved_vit, pretrained)
            vit, _, _ = open_clip.create_model_and_transforms(
                resolved_vit,
                pretrained=resolved_pretrained,
            )
            if resolved_second is None:
                rn50 = None
            else:
                resolved_pretrained = _resolve_openclip_pretrained(
                    resolved_second,
                    pretrained_second if pretrained_second is not None else pretrained,
                )
                rn50, _, _ = open_clip.create_model_and_transforms(
                    resolved_second,
                    pretrained=resolved_pretrained,
                )
            vit = vit.to(device)
            if rn50 is not None:
                rn50 = rn50.to(device)
        else:
            if clip is None:
                raise ImportError(
                    "The 'clip' package is required for OpenAI CLIP models. Install it with `pip install git+https://github.com/openai/CLIP.git`."
                )
            vit, _ = clip.load(resolved_vit, device=device, jit=False)
            if resolved_second is None:
                rn50 = None
            else:
                rn50, _ = clip.load(resolved_second, device=device, jit=False)
        vit.eval()
        if rn50 is not None:
            rn50.eval()
        for param in vit.parameters():
            param.requires_grad_(False)
        if rn50 is not None:
            for param in rn50.parameters():
                param.requires_grad_(False)
        _CLIP_MODELS[key] = (vit, rn50)
    else:
        if _CLIP_VERBOSE:
            logging.info("Reusing cached CLIP models for key=%s", key)
    return _CLIP_MODELS[key]


def get_clip_models(device: Union[torch.device, str, None] = None) -> Tuple[object, Optional[object], torch.device]:
    """Return the cached CLIP models and the resolved device."""

    resolved_device = _canonical_device(device)
    vit, rn50 = _load_models(resolved_device)
    return vit, rn50, resolved_device


def set_clip_model_names(vit_name: str, rn50_name: Optional[str]) -> None:
    """Update CLIP model names and clear cached models/compilations if changed."""

    global _CLIP_MODEL_NAMES, _CLIP_MODELS, _CLIP_COMPILED, _CLIP_COMPILE_ACTIVE_KEY, _OPENCLIP_TOKENIZERS
    vit_name = str(vit_name)
    if rn50_name is None:
        rn50 = None
    else:
        rn50 = str(rn50_name).strip()
        if rn50.lower() in ("none", "null", ""):
            rn50 = None
    if rn50 == vit_name:
        rn50 = None
    if _CLIP_MODEL_NAMES == (vit_name, rn50):
        return
    _CLIP_MODEL_NAMES = (vit_name, rn50)
    _CLIP_MODELS = {}
    _CLIP_COMPILED = {}
    _CLIP_COMPILE_ACTIVE_KEY = None
    _OPENCLIP_TOKENIZERS = {}


def set_clip_provider(
    provider: str,
    openclip_pretrained: Union[str, None] = None,
    openclip_pretrained_second: Union[str, None] = None,
    verbose: bool = False,
) -> None:
    """Set the CLIP provider ("openai" or "openclip") and optional pretrained tags."""

    global _CLIP_PROVIDER, _CLIP_OPENCLIP_PRETRAINED, _CLIP_OPENCLIP_PRETRAINED_SECOND, _CLIP_VERBOSE
    global _CLIP_MODELS, _CLIP_COMPILED, _CLIP_COMPILE_ACTIVE_KEY
    provider = str(provider).strip().lower()
    if provider not in ("openai", "openclip"):
        raise ValueError("clip_provider must be 'openai' or 'openclip'")
    pretrained = None if openclip_pretrained is None else str(openclip_pretrained)
    pretrained_second = None if openclip_pretrained_second is None else str(openclip_pretrained_second)
    if (
        _CLIP_PROVIDER,
        _CLIP_OPENCLIP_PRETRAINED,
        _CLIP_OPENCLIP_PRETRAINED_SECOND,
        bool(_CLIP_VERBOSE),
    ) == (provider, pretrained, pretrained_second, bool(verbose)):
        return
    _CLIP_PROVIDER = provider
    _CLIP_OPENCLIP_PRETRAINED = pretrained
    _CLIP_OPENCLIP_PRETRAINED_SECOND = pretrained_second
    _CLIP_VERBOSE = bool(verbose)
    _CLIP_MODELS = {}
    _CLIP_COMPILED = {}
    _CLIP_COMPILE_ACTIVE_KEY = None
    _set_openclip_logging(_CLIP_VERBOSE)


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
    key = (
        str(resolved_device),
        _CLIP_PROVIDER,
        _CLIP_OPENCLIP_PRETRAINED,
        _CLIP_OPENCLIP_PRETRAINED_SECOND,
        _resolve_model_name(_CLIP_MODEL_NAMES[0], _CLIP_PROVIDER),
        _resolve_model_name(_CLIP_MODEL_NAMES[1], _CLIP_PROVIDER),
        mode,
        backend,
        dynamic,
        fullgraph,
    )
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
    _compile("vit_image", vit.encode_image)
    if rn50 is not None:
        _compile("rn50_text", rn50.encode_text)
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
        key = (
            str(device),
            _CLIP_PROVIDER,
            _CLIP_OPENCLIP_PRETRAINED,
            _CLIP_OPENCLIP_PRETRAINED_SECOND,
            _resolve_model_name(_CLIP_MODEL_NAMES[0], _CLIP_PROVIDER),
            _resolve_model_name(_CLIP_MODEL_NAMES[1], _CLIP_PROVIDER),
            mode,
            backend,
            dynamic,
            fullgraph,
        )
        return _CLIP_COMPILED.get(key, {})
    if _CLIP_COMPILE_ACTIVE_KEY is None:
        return {}
    if _CLIP_COMPILE_ACTIVE_KEY[0] != str(device):
        return {}
    return _CLIP_COMPILED.get(_CLIP_COMPILE_ACTIVE_KEY, {})


@functools.lru_cache(maxsize=None)
def _cached_tokenize(provider: str, model_name: str, text: str) -> torch.Tensor:
    """Tokenize ``text`` once and cache the resulting IDs on the default device."""

    _, _, device = get_clip_models(None)
    if provider == "openclip":
        if open_clip is None:
            raise ImportError(
                "OpenCLIP is not installed. Install it with `pip install open_clip_torch`."
            )
        tokenizer = _get_openclip_tokenizer(model_name)
        return tokenizer(text).to(device)
    if clip is None:
        raise ImportError(
            "The 'clip' package is required for OpenAI CLIP models. Install it with `pip install git+https://github.com/openai/CLIP.git`."
        )
    return clip.tokenize(text).to(device)


def tokenize_text(text: str) -> torch.Tensor:
    """Tokenize ``text`` once and cache the resulting IDs on the default device."""

    provider = _CLIP_PROVIDER
    vit_name = _resolve_model_name(_CLIP_MODEL_NAMES[0], provider)
    return _cached_tokenize(provider, vit_name, text)


@torch.no_grad()
def embed_text(text: str, device: Union[torch.device, str, None] = None, config=None) -> torch.Tensor:
    """Encode ``text`` with both CLIP encoders and concatenate the features."""

    vit, rn50, resolved_device = get_clip_models(device)
    if _CLIP_PROVIDER == "openclip":
        vit_name = _resolve_model_name(_CLIP_MODEL_NAMES[0], _CLIP_PROVIDER)
        tokenized = _cached_tokenize(_CLIP_PROVIDER, vit_name, text).to(resolved_device)
    else:
        if clip is None:
            raise ImportError(
                "The 'clip' package is required for OpenAI CLIP models. Install it with `pip install git+https://github.com/openai/CLIP.git`."
            )
        tokenized = clip.tokenize(text).to(resolved_device)
    compiled = _get_compiled(resolved_device, config) if config is not None else {}
    vit_fn = compiled.get("vit_text", vit.encode_text)
    vit_features = vit_fn(tokenized)
    if rn50 is None:
        combined = vit_features
    else:
        rn50_fn = compiled.get("rn50_text", rn50.encode_text)
        rn50_features = rn50_fn(tokenized)
        combined = torch.cat([vit_features, rn50_features], dim=-1)
    combined = combined.to(dtype=torch.float32)
    return combined.squeeze(0)


def _get_image_preprocess(model: object) -> Tuple[Tuple[int, int], Tuple[float, float, float], Tuple[float, float, float]]:
    size = (224, 224)
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    if _CLIP_PROVIDER == "openclip" and hasattr(model, "visual"):
        visual = model.visual
        if hasattr(visual, "image_size"):
            img_size = visual.image_size
            if isinstance(img_size, (tuple, list)) and len(img_size) >= 2:
                size = (int(img_size[0]), int(img_size[1]))
            elif isinstance(img_size, int):
                size = (int(img_size), int(img_size))
        if hasattr(visual, "image_mean"):
            try:
                mean = tuple(float(x) for x in visual.image_mean)
            except Exception:
                pass
        if hasattr(visual, "image_std"):
            try:
                std = tuple(float(x) for x in visual.image_std)
            except Exception:
                pass
    return size, mean, std


def _normalize_images(
    images: torch.Tensor,
    size: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> torch.Tensor:
    images = images.clamp(0.0, 1.0)
    if images.shape[-2:] != size:
        images = F.interpolate(images, size=size, mode="bicubic", align_corners=False)
    mean_t = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean_t) / std_t


def embed_images(images: torch.Tensor, device: Union[torch.device, str, None] = None, config=None) -> torch.Tensor:
    """Embed ``images`` (NCHW, [0,1]) with CLIP and concatenate encoder features."""

    vit, rn50, resolved_device = get_clip_models(device)
    if images.device != resolved_device:
        images = images.to(resolved_device)

    vit_size, vit_mean, vit_std = _get_image_preprocess(vit)
    normalized_vit = _normalize_images(images, vit_size, vit_mean, vit_std)
    if rn50 is not None:
        rn50_size, rn50_mean, rn50_std = _get_image_preprocess(rn50)
        if (vit_size, vit_mean, vit_std) == (rn50_size, rn50_mean, rn50_std):
            normalized_rn50 = normalized_vit
        else:
            normalized_rn50 = _normalize_images(images, rn50_size, rn50_mean, rn50_std)

    compiled = _get_compiled(resolved_device, config) if config is not None else {}
    vit_fn = compiled.get("vit_image", vit.encode_image)
    vit_features = vit_fn(normalized_vit)
    if rn50 is None:
        combined = vit_features
    else:
        rn50_fn = compiled.get("rn50_image", rn50.encode_image)
        rn50_features = rn50_fn(normalized_rn50)
        combined = torch.cat([vit_features, rn50_features], dim=-1)
    return combined.to(dtype=torch.float32)


def cosine_similarity(image_embeddings: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
    """Return CLIP cosine similarity scores for image and text embeddings."""

    if text_embedding.ndim == 1:
        text_embedding = text_embedding.unsqueeze(0)
    image_norm = torch.nn.functional.normalize(image_embeddings, dim=-1)
    text_norm = torch.nn.functional.normalize(text_embedding, dim=-1)
    return torch.matmul(image_norm, text_norm.T).squeeze(-1)
