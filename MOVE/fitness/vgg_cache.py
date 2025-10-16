"""Helpers for caching VGG feature extractor outputs across fitness metrics."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch

from .feature_cache import cached_result
from .backbones import FEATURE_EXTRACTOR


def _normalize_layers(layers: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(set(int(layer) for layer in layers)))


def _ensure_feature_extractor_device(device: torch.device) -> None:
    current_device = next(FEATURE_EXTRACTOR.parameters()).device
    if current_device != device:
        FEATURE_EXTRACTOR.to(device)
    FEATURE_EXTRACTOR.eval()


def _run_vgg_features(tensor: torch.Tensor, layers: Sequence[int]) -> Tuple[torch.Tensor, ...]:
    _ensure_feature_extractor_device(tensor.device)
    outputs = []
    x = tensor
    layer_set = set(layers)
    for idx, module in enumerate(FEATURE_EXTRACTOR):
        x = module(x)
        if idx in layer_set:
            outputs.append(x)
    return tuple(outputs)


def get_vgg_features(
    tensor: torch.Tensor,
    layers: Sequence[int],
    *,
    persistent: bool = False,
    detach: bool = False,
    store_on_cpu: bool = False,
    cache_tensor: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, ...]:
    """Return cached VGG16 feature maps for ``tensor`` at ``layers`` indices.

    Args:
        tensor: Input tensor already in the desired scaling space.
        layers: Iterable of layer indices from ``torchvision.models.vgg16().features``
            whose outputs should be returned.
        persistent: Store the cached result in the global cache so it can be reused
            across batches (useful for static targets). When ``True`` consider
            setting ``store_on_cpu`` to avoid holding GPU memory indefinitely.
        detach: Detach the cached tensors from the autograd graph (recommended for
            targets). When ``False`` gradients flow to the input tensor.
        store_on_cpu: Move cached tensors to CPU before storing. When retrieving on
            GPU a device-specific copy will be cached in the scoped cache.
        cache_tensor: Optional tensor whose storage identity should be used as the
            cache key. This is useful when ``tensor`` is the result of lightweight
            transformations (e.g. normalization) but the cache should survive
            across repeated constructions of that tensor from a shared source.
    """

    layers_tuple = _normalize_layers(layers)
    base_slot = ("vgg16", layers_tuple, "detached" if detach else "grad")
    storage_slot = base_slot + (("cpu",) if store_on_cpu else ("device",))
    key_tensor = cache_tensor if cache_tensor is not None else tensor

    def build_features() -> Tuple[torch.Tensor, ...]:
        feats = _run_vgg_features(tensor, layers_tuple)
        if detach:
            feats = tuple(f.detach() for f in feats)
        if store_on_cpu:
            feats = tuple(f.to("cpu") for f in feats)
        return feats

    features = cached_result(storage_slot, key_tensor, build_features, persistent=persistent)

    if store_on_cpu and tensor.device.type != "cpu":
        gpu_slot = base_slot + (tensor.device.type,)

        def to_gpu() -> Tuple[torch.Tensor, ...]:
            return tuple(f.to(tensor.device) for f in features)

        features = cached_result(gpu_slot, key_tensor, to_gpu)

    return features