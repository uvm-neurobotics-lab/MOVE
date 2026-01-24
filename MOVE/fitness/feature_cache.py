"""Shared feature caching utilities for MOVE fitness metrics.

This module exposes a lightweight scoped cache that lets heavy feature
extractors (such as VGG16) avoid recomputing activations when the same
input tensor is evaluated multiple times within a batch.  It also offers
an optional persistent layer for tensors that rarely change (for example
benchmark targets) so downstream metrics can skip redundant work across
batches.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, Optional, Tuple

import torch

try:
    _dynamo_disable = torch._dynamo.disable
except Exception:  # pragma: no cover - torch._dynamo may be unavailable
    def _dynamo_disable(fn):
        return fn

TensorKey = Tuple[int, Tuple[int, ...], torch.device, torch.dtype]
CacheKey = Tuple[Hashable, TensorKey]


@dataclass
class _Cache:
    store: Dict[CacheKey, Any]

    def __init__(self) -> None:
        self.store = {}

    @staticmethod
    def tensor_key(tensor: torch.Tensor) -> TensorKey:
        return (
            int(tensor.data_ptr()),
            tuple(tensor.shape),
            tensor.device,
            tensor.dtype,
        )

    def get(self, slot: Hashable, tensor: torch.Tensor, builder: Callable[[], Any]) -> Any:
        key = (slot, self.tensor_key(tensor))
        if key not in self.store:
            self.store[key] = builder()
        return self.store[key]

    def set(self, slot: Hashable, tensor: torch.Tensor, value: Any) -> Any:
        key = (slot, self.tensor_key(tensor))
        self.store[key] = value
        return value

    def pop_many(self, slots: Iterable[Hashable], tensor: torch.Tensor) -> None:
        key = self.tensor_key(tensor)
        for slot in slots:
            self.store.pop((slot, key), None)

    def clear(self) -> None:
        self.store.clear()


_LOCAL_CACHE: Optional[_Cache] = None
_PERSISTENT_CACHE: _Cache = _Cache()


@contextmanager
def feature_cache_scope():
    """Create a scoped cache for repeated feature extraction.

    The scoped cache is reset when the context exits so per-batch
    intermediate activations do not leak across batches.
    """

    global _LOCAL_CACHE
    previous = _LOCAL_CACHE
    cache = _Cache()
    _LOCAL_CACHE = cache
    try:
        yield cache
    finally:
        cache.clear()
        _LOCAL_CACHE = previous


@_dynamo_disable
def cached_result(
    slot: Hashable,
    tensor: Optional[torch.Tensor],
    builder: Callable[[], Any],
    *,
    persistent: bool = False,
) -> Any:
    """Return a cached result for ``tensor`` if available, else build it.

    Args:
        slot: Stable identifier for the cached computation (e.g. "lpips:features").
        tensor: Tensor used as the cache key. When ``None`` no caching occurs.
        builder: Callable that produces the value if it wasn't cached yet.
        persistent: When ``True`` the value is stored in the long-lived cache
            that survives across scopes; otherwise it is scoped to the current
            ``feature_cache_scope``.
    """

    if tensor is None:
        return builder()

    cache = _PERSISTENT_CACHE if persistent else _LOCAL_CACHE
    if cache is None:
        return builder()
    return cache.get(slot, tensor, builder)


@_dynamo_disable
def store_result(
    slot: Hashable,
    tensor: Optional[torch.Tensor],
    value: Any,
    *,
    persistent: bool = False,
) -> Any:
    """Force-store a value in the cache and return it."""

    if tensor is None:
        return value

    cache = _PERSISTENT_CACHE if persistent else _LOCAL_CACHE
    if cache is None:
        return value
    return cache.set(slot, tensor, value)


@_dynamo_disable
def invalidate_slots(slots: Iterable[Hashable], tensor: Optional[torch.Tensor]) -> None:
    """Invalidate cached entries for ``tensor`` in the given slots."""

    if tensor is None:
        return

    for cache in (_LOCAL_CACHE, _PERSISTENT_CACHE):
        if cache is None:
            continue
        cache.pop_many(slots, tensor)


@_dynamo_disable
def clear_persistent_cache() -> None:
    """Remove all entries from the persistent cache."""

    _PERSISTENT_CACHE.clear()
