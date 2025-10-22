"""Cached perceptual helpers for MOVE style and content losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .feature_cache import cached_result
from .vgg_cache import get_vgg_features
from .lpips import SHIFT as _IMAGENET_MEAN, SCALE as _IMAGENET_STD

# Indices inside torchvision.models.vgg16().features matching the classic
# Gatys style transfer configuration. ``STYLE_LAYERS`` corresponds to the
# first two convolutions in blocks 1 and 2 plus the first convolution in block 3.
STYLE_LAYERS = (0, 2, 5, 7, 10)
CONTENT_LAYERS = (19,)  # conv4_2
_UNION_LAYERS = tuple(sorted(set(STYLE_LAYERS + CONTENT_LAYERS)))


def _imagenet_normalize(tensor: torch.Tensor) -> torch.Tensor:
    shift = _IMAGENET_MEAN.view(1, -1, 1, 1).to(device=tensor.device, dtype=tensor.dtype)
    scale = _IMAGENET_STD.view(1, -1, 1, 1).to(device=tensor.device, dtype=tensor.dtype)
    return (tensor - shift) / scale


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    r"""Compute the Gram matrix for each sample in ``x``.

    Args:
        x: Tensor of shape ``(N, C, H, W)``.

    Returns:
        Tensor of shape ``(N, C, C)`` containing the normalized Gram matrices.
    """

    n, c, h, w = x.shape
    features = x.view(n, c, h * w)
    gram = torch.matmul(features, features.transpose(-1, -2))
    return gram.div(c * h * w)


def _perceptual_features(
    tensor: torch.Tensor,
    *,
    layers: tuple[int, ...],
    detach: bool = False,
    persistent: bool = False,
    store_on_cpu: bool = False,
    cache_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Return cached VGG16 features for ``tensor`` at ``layers`` indices.

    The forward pass is shared across all perceptual metrics via a superset of
    required layers, so repeated calls for different metrics reuse the same
    cached activations.
    """

    key_tensor = cache_tensor if cache_tensor is not None else tensor

    norm_slot = ("imagenet", "normalized", "cpu" if store_on_cpu else "device")

    def _build_normalized() -> torch.Tensor:
        norm = _imagenet_normalize(tensor)
        if store_on_cpu and norm.device.type != "cpu":
            norm = norm.to("cpu")
        return norm

    normalized = cached_result(norm_slot, key_tensor, _build_normalized, persistent=persistent)

    if store_on_cpu and tensor.device.type != "cpu":
        device_slot = ("imagenet", "normalized", tensor.device.type, tensor.device.index)

        def _to_device() -> torch.Tensor:
            return normalized.to(tensor.device)

        normalized = cached_result(device_slot, key_tensor, _to_device)

    union_features = get_vgg_features(
        normalized,
        _UNION_LAYERS,
        persistent=persistent,
        detach=detach,
        store_on_cpu=store_on_cpu,
        cache_tensor=key_tensor,
    )

    feature_map = {layer: feat for layer, feat in zip(_UNION_LAYERS, union_features)}
    return tuple(feature_map[layer] for layer in layers)


class StyleLoss(nn.Module):
    """LPIPS-inspired style loss with aggressive feature caching."""

    def __init__(self, _model, device, target, weight: float = 1.0):
        super().__init__()
        self.device = torch.device(device)
        self.weight = float(weight)

        self._target_tensor: torch.Tensor | None = None
        self._target_grams_cpu: tuple[torch.Tensor, ...] | None = None
        self._target_value: torch.Tensor | None = None

        self.setup(target)

    @property
    def _style_layers(self) -> tuple[int, ...]:
        return STYLE_LAYERS

    def _target_grams_for_device(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        assert self._target_tensor is not None
        assert self._target_grams_cpu is not None

        if self._target_grams_cpu[0].device == device:
            return self._target_grams_cpu

        return cached_result(
            ("style", "target_grams", device.type, device.index),
            self._target_tensor,
            lambda: tuple(g.to(device) for g in self._target_grams_cpu),
            persistent=False,
        )

    def setup(self, target: torch.Tensor) -> None:
        raw_target = target
        target_value = raw_target.to(device=self.device, dtype=torch.float32)
        self._target_tensor = raw_target
        self._target_value = target_value

        target_features = _perceptual_features(
            target_value,
            layers=self._style_layers,
            detach=True,
            persistent=True,
            store_on_cpu=True,
            cache_tensor=raw_target,
        )
        self._target_grams_cpu = tuple(gram_matrix(feat) for feat in target_features)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raw_target = target
        target_value = raw_target.to(device=self.device, dtype=torch.float32)
        if (
            self._target_tensor is None
            or self._target_value is None
            or self._target_value.shape != target_value.shape
            or not torch.isclose(self._target_value, target_value, rtol=1e-3, atol=1e-3).all()
        ):
            self.setup(target)

        raw_input = input
        input = raw_input.to(device=self.device, dtype=torch.float32)
        input_features = _perceptual_features(
            input,
            layers=self._style_layers,
            detach=False,
            persistent=False,
            store_on_cpu=False,
            cache_tensor=raw_input,
        )

        target_grams = self._target_grams_for_device(input.device)
        style_score = torch.zeros(input.shape[0], device=input.device, dtype=input.dtype)

        for gram_input, gram_target in zip(input_features, target_grams):
            g_input = gram_matrix(gram_input)
            g_target = gram_target
            if g_target.dim() == 2:
                g_target = g_target.unsqueeze(0)
            if g_target.shape[0] == 1 and g_input.shape[0] > 1:
                g_target = g_target.expand(g_input.shape[0], -1, -1)
            diff = (g_input - g_target).pow(2).mean(dim=(1, 2))
            style_score += diff

        style_score = torch.sqrt(torch.sqrt(style_score * self.weight))
        return style_score


class ContentLoss(nn.Module):
    """Content loss that reuses MOVE's perceptual caches."""

    def __init__(self, device, target, weight: float = 1.0):
        super().__init__()
        self.device = torch.device(device)
        self.weight = float(weight)

        self._target_tensor: torch.Tensor | None = None
        self._target_features_cpu: tuple[torch.Tensor, ...] | None = None
        self._target_value: torch.Tensor | None = None

        self.setup(target)

    @property
    def _content_layers(self) -> tuple[int, ...]:
        return CONTENT_LAYERS

    def _target_features_for_device(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        assert self._target_tensor is not None
        assert self._target_features_cpu is not None

        if self._target_features_cpu[0].device == device:
            return self._target_features_cpu

        return cached_result(
            ("content", "target_features", device.type, device.index),
            self._target_tensor,
            lambda: tuple(feat.to(device) for feat in self._target_features_cpu),
            persistent=False,
        )

    def setup(self, target: torch.Tensor) -> None:
        raw_target = target
        target_value = raw_target.to(device=self.device, dtype=torch.float32)
        self._target_tensor = raw_target
        self._target_value = target_value

        target_features = _perceptual_features(
            target_value,
            layers=self._content_layers,
            detach=True,
            persistent=True,
            store_on_cpu=True,
            cache_tensor=raw_target,
        )
        self._target_features_cpu = target_features

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raw_target = target
        target_value = raw_target.to(device=self.device, dtype=torch.float32)
        if (
            self._target_tensor is None
            or self._target_value is None
            or self._target_value.shape != target_value.shape
            or not torch.isclose(self._target_value, target_value, rtol=1e-3, atol=1e-3).all()
        ):
            self.setup(target)

        raw_input = input
        input = raw_input.to(device=self.device, dtype=torch.float32)
        input_features = _perceptual_features(
            input,
            layers=self._content_layers,
            detach=False,
            persistent=False,
            store_on_cpu=False,
            cache_tensor=raw_input,
        )

        target_features = self._target_features_for_device(input.device)
        content_loss = torch.zeros(input.shape[0], device=input.device, dtype=input.dtype)

        for feat_input, feat_target in zip(input_features, target_features):
            if feat_target.dim() == 3:  # single target sample
                feat_target = feat_target.unsqueeze(0)
            if feat_target.shape[0] == 1 and feat_input.shape[0] > 1:
                feat_target = feat_target.expand(feat_input.shape[0], -1, -1, -1)
            diff = (feat_input - feat_target).pow(2).mean(dim=(1, 2, 3))
            content_loss += diff

        return content_loss * self.weight
