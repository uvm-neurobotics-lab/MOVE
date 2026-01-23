"""Gradient-based fine-tuning utilities used throughout MOVE.

This module centralizes the stochastic gradient descent (SGD) helpers that
polish CPPN genomes after evolutionary updates.  The helpers manage image
pre-processing, feature caching, mixed-precision autocasting, and per-genome
early stopping so the evolutionary loop can stay focused on high-level logic.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from .fitness.feature_cache import feature_cache_scope
from .norm import norm_tensor
from .util import is_canonical_image_batch

__all__ = ["sgd_weights", "sgd_weights_imaml", "prep_images"]

_TARGET_PREP_CACHE: Dict[Tuple[int, Tuple[int, ...], torch.device], torch.Tensor] = {}

try:
    torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

class _CompiledForwardCache:
    """Cache torch.compile'd forward graphs for CPPN genomes during SGD."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, torch.device, str, Tuple], Callable[[torch.Tensor], torch.Tensor]] = {}
        self._last_signature: Dict[int, Tuple] = {}
        self._disabled: Set[int] = set()

    @staticmethod
    def _signature(genome) -> Tuple:
        nodes = getattr(genome, "nodes", {})
        enabled = getattr(genome, "enabled_connections", [])
        node_items = tuple(
            (str(k), type(v.activation).__name__)
            for k, v in sorted(nodes.items(), key=lambda kv: str(kv[0]))
        )
        conn_items = tuple(sorted(str(c) for c in enabled))
        return (len(nodes), len(enabled), node_items, conn_items)

    def get(self, genome, config) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        if not getattr(config, "sgd_use_compiled_forward", False):
            return None
        if not hasattr(torch, "compile"):
            return None
        genome_id = int(genome.id)
        if genome_id in self._disabled:
            return None
        device = torch.device(config.device) if config.device is not None else torch.device("cpu")
        if device.type != "cuda":
            return None
        mode = str(getattr(config, "sgd_compile_mode", "reduce-overhead"))
        signature = self._signature(genome)
        last_sig = self._last_signature.get(genome_id)
        if last_sig is None:
            self._last_signature[genome_id] = signature
            return None
        if last_sig != signature:
            self._last_signature[genome_id] = signature
            return None

        key = (genome_id, device, mode, signature)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        def forward_fn(x: torch.Tensor) -> torch.Tensor:
            return genome(
                x,
                force_recalculate=True,
                channel_first=True,
                use_graph=True,
            )

        try:
            compiled = torch.compile(
                forward_fn,
                    dynamic=bool(getattr(config, "sgd_compile_dynamic", True)),
                mode=mode,
                fullgraph=bool(getattr(config, "sgd_compile_fullgraph", False)),
            )
        except Exception:
            return None

        self._cache[key] = compiled
        return compiled

    def disable(self, genome) -> None:
        genome_id = int(genome.id)
        self._disabled.add(genome_id)
        keys_to_drop = [key for key in self._cache if key[0] == genome_id]
        for key in keys_to_drop:
            self._cache.pop(key, None)


_SGD_FORWARD_CACHE = _CompiledForwardCache()


class EarlyStopping:
    """Keep per-genome patience counters for the SGD optimiser."""

    def __init__(
        self,
        patience: int = 1,
        min_delta: float = 0.0,
        n_genomes: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.min_loss = torch.inf

        self.counter_tensor = torch.zeros(n_genomes, device=device)
        self.min_loss_tensor = torch.full((n_genomes,), torch.inf, device=device)

    def check_stop(self, loss: float) -> bool:
        """Return ``True`` when the aggregate loss stops improving."""

        if loss < (self.min_loss + self.min_delta):
            self.min_loss = loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def mask_stop(
        self, loss: torch.Tensor, indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return a boolean mask for genomes that should stop updating."""

        loss = loss.flatten()
        if indices is None:
            indices = torch.arange(loss.shape[0], device=self.counter_tensor.device)
        else:
            indices = indices.to(self.counter_tensor.device)

        improve = loss < (self.min_loss_tensor[indices] + self.min_delta)
        self.counter_tensor[indices] = torch.where(
            improve,
            torch.zeros_like(self.counter_tensor[indices]),
            self.counter_tensor[indices] + 1,
        )
        self.min_loss_tensor[indices] = torch.where(
            improve, loss, self.min_loss_tensor[indices]
        )
        return self.counter_tensor[indices] >= self.patience


def _ensure_batched_rgb(imgs: torch.Tensor) -> torch.Tensor:
    """Promote arbitrary tensors to the ``(N, 3, H, W)`` MOVE format."""

    if imgs.ndim == 2:
        imgs = imgs.unsqueeze(0).unsqueeze(0)
    elif imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.ndim != 4:
        raise ValueError(f"Unsupported image tensor with shape {tuple(imgs.shape)}")
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    elif imgs.shape[1] != 3 and imgs.shape[-1] == 3:
        imgs = imgs.permute(0, 3, 1, 2)
    elif imgs.shape[1] not in (1, 3):
        imgs = imgs[:, :3]
    return imgs.contiguous()


def _resize_to_min(imgs: torch.Tensor) -> torch.Tensor:
    """Ensure every image has at least 33×33 pixels before scoring."""

    if imgs.shape[-2] < 33 or imgs.shape[-1] < 33:
        imgs = F.interpolate(imgs, size=(33, 33), mode="bilinear", align_corners=False)
    return imgs


def prep_images(imgs: torch.Tensor, config, *, copy: bool = False) -> torch.Tensor:
    """Normalise arbitrary tensors into canonical MOVE image batches."""
    if not torch.isfinite(imgs).all():
        raise ValueError("Non-finite values detected in image batch")

    if is_canonical_image_batch(imgs):
        result = imgs.clone() if copy else imgs
        # Apply channels_last memory format for better GPU performance
        if getattr(config, "use_channels_last", True) and result.device.type == "cuda":
            if not result.is_contiguous(memory_format=torch.channels_last):
                result = result.contiguous(memory_format=torch.channels_last)
            return result
        if not result.is_contiguous():
            result = result.contiguous()
        return result

    working = imgs.clone() if copy else imgs
    working = _ensure_batched_rgb(working)
    if working.dtype != torch.float32:
        working = working.to(dtype=torch.float32)
    working = _resize_to_min(working)
    working = torch.clamp(working, 0.0, 1.0)
    if not torch.isfinite(working).all():
        raise ValueError("Non-finite values detected after preprocessing")
    
    # Apply channels_last memory format for better GPU performance
    if getattr(config, "use_channels_last", True) and working.device.type == "cuda":
        if not working.is_contiguous(memory_format=torch.channels_last):
            working = working.contiguous(memory_format=torch.channels_last)
        return working
    if not working.is_contiguous():
        working = working.contiguous()
    return working


def _prepare_target_cache(target: torch.Tensor, config) -> torch.Tensor:
    """Cache and reuse the prepared target tensor across SGD steps."""

    cache_key = (int(target.data_ptr()), tuple(target.shape), target.device)
    cached = _TARGET_PREP_CACHE.get(cache_key)
    if cached is None or cached.shape != target.shape:
        cached = prep_images(target, config, copy=True).detach()
        # Use pinned memory for faster CPU-GPU transfers if enabled
        if getattr(config, "use_pinned_memory", True) and cached.device.type == 'cpu':
            cached = cached.pin_memory()
        _TARGET_PREP_CACHE[cache_key] = cached
    return cached


def _ensure_tensor_on_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    return tensor.to(device=device, non_blocking=True)


def _render_population_eager(genomes_batch, inputs_batch, feed_fn, config=None) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    for idx, genome in enumerate(genomes_batch):
        feed = feed_fn(inputs_batch, idx)
        # Ensure feed matches genome device/dtype/format to avoid implicit copies during forward
        try:
            if isinstance(feed, torch.Tensor):
                target_device = getattr(genome, "device", None) or (torch.device(config.device) if config is not None else feed.device)
                if feed.dtype != torch.float32:
                    feed = feed.to(dtype=torch.float32)
                if feed.device != target_device:
                    feed = feed.to(device=target_device, non_blocking=True)
                if config is not None and getattr(config, "use_channels_last", False) and feed.device.type == "cuda" and feed.dim() == 4:
                    if not feed.is_contiguous(memory_format=torch.channels_last):
                        feed = feed.contiguous(memory_format=torch.channels_last)
        except Exception:
            pass
        out = genome(
            feed,
            force_recalculate=True,
            channel_first=True,
            use_graph=True,
        )
        # Convert once to channels_last when configured and on CUDA to avoid repeated layout thrash
        try:
            if config is not None and getattr(config, "use_channels_last", False) and out.device.type == "cuda" and out.dim() == 4:
                if not out.is_contiguous(memory_format=torch.channels_last):
                    out = out.contiguous(memory_format=torch.channels_last)
        except Exception:
            pass
        outputs.append(out)
    return torch.stack(outputs)


def _render_population_compiled(genomes_batch, inputs_batch, feed_fn, config) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    for idx, genome in enumerate(genomes_batch):
        feed = feed_fn(inputs_batch, idx)
        # Prepare feed to match genome device/dtype/format to avoid implicit host-device transfers
        try:
            if isinstance(feed, torch.Tensor):
                target_device = getattr(genome, "device", None) or (torch.device(config.device) if config is not None else feed.device)
                if feed.dtype != torch.float32:
                    feed = feed.to(dtype=torch.float32)
                if feed.device != target_device:
                    feed = feed.to(device=target_device, non_blocking=True)
                if getattr(config, "use_channels_last", False) and feed.device.type == "cuda" and feed.dim() == 4:
                    if not feed.is_contiguous(memory_format=torch.channels_last):
                        feed = feed.contiguous(memory_format=torch.channels_last)
        except Exception:
            pass
        compiled = _SGD_FORWARD_CACHE.get(genome, config)
        if compiled is None:
            outputs.append(
                genome(
                    feed,
                    force_recalculate=True,
                    channel_first=True,
                    use_graph=True,
                )
            )
        else:
            try:
                out = compiled(feed)
                # Convert to channels_last once if requested, to avoid layout thrash
                if getattr(config, "use_channels_last", False) and out.device.type == "cuda" and out.dim() == 4:
                    if not out.is_contiguous(memory_format=torch.channels_last):
                        out = out.contiguous(memory_format=torch.channels_last)
                outputs.append(out)
            except Exception:
                _SGD_FORWARD_CACHE.disable(genome)
                outputs.append(
                    genome(
                        feed,
                        force_recalculate=True,
                        channel_first=True,
                        use_graph=True,
                    )
                )
    return torch.stack(outputs)

def _evaluate_function_masked(
    fn,
    fn_name: str,
    imgs: torch.Tensor,
    target: torch.Tensor,
    mask_row: Optional[torch.Tensor],
    *,
    use_cuda_amp: bool,
    amp_whitelist: Set[str],
    normalizer,
    norm,
    config,
) -> torch.Tensor:
    """Evaluate ``fn`` on ``imgs`` while respecting an optional mask."""

    active_indices: Optional[torch.Tensor] = None
    if mask_row is not None:
        if mask_row.dtype != torch.bool:
            mask_row = mask_row.to(dtype=torch.bool)
        active_indices = mask_row.nonzero(as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return imgs.new_zeros(imgs.shape[0], dtype=torch.float32)
        if active_indices.numel() == imgs.shape[0]:
            active_indices = None

    if active_indices is not None:
        fn_imgs = imgs.index_select(0, active_indices)
        fn_target = target.index_select(0, active_indices)
    else:
        fn_imgs = imgs
        fn_target = target

    def _compute(autocast_enabled: bool) -> torch.Tensor:
        amp_ctx = (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if autocast_enabled
            else nullcontext()
        )
        with amp_ctx:
            result = fn(fn_imgs, fn_target)
        if result.dim() == 0:
            result = result.unsqueeze(0)
        return result

    autocast_enabled = use_cuda_amp and fn_name in amp_whitelist
    fitness = _compute(autocast_enabled)
    if not torch.isfinite(fitness).all():
        fitness = _compute(False)
        if fitness.dim() == 0:
            fitness = fitness.unsqueeze(0)

    fitness = fitness.to(device=imgs.device, dtype=torch.float32, non_blocking=True)

    if active_indices is not None:
        norm_input = fitness
    else:
        norm_input = fitness

    if normalizer is not None:
        normed = normalizer(fn, norm_input)
    else:
        normed = norm_tensor(norm_input, norm, fn_name, clamp=True, warn=False)

    normed = torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)

    normed = normed.to(dtype=torch.float32)
    if active_indices is None:
        return normed

    full = imgs.new_zeros(imgs.shape[0], dtype=torch.float32)
    full.index_copy_(0, active_indices, normed)
    return full


def sgd_weights(
    genomes,
    mask,
    inputs,
    target,
    fns,
    norm,
    config,
    early_stop: int = 3,
    record_loss: Optional[torch.Tensor] = None,
    skip_pbar: bool = False,
    current_gen: int = 0,
    unequal_shape: bool = False,
    record_passes: Optional[List[int]] = None,
    normalizer=None,
) -> int:
    """Fine-tune a batch of CPPN genomes with AdamW-based SGD."""

    device = torch.device(config.device) if config.device is not None else torch.device("cpu")
    if isinstance(inputs, torch.Tensor):
        inputs = _ensure_tensor_on_device(inputs, device)
    if isinstance(target, torch.Tensor):
        target = _ensure_tensor_on_device(target, device)

    if getattr(config, "sgd_no_branch", False):
        return sgd_weights_no_branch(
            genomes,
            mask,
            inputs,
            target,
            fns,
            norm,
            config,
            early_stop,
            record_loss,
            skip_pbar,
            current_gen,
            unequal_shape,
            record_passes,
            normalizer,
        )

    lr = float(getattr(config, "sgd_learning_rate", 0.0))
    raw_steps = getattr(config, "sgd_steps", 0)
    if isinstance(raw_steps, str):
        logging.warning(
            "String-based SGD schedules are no longer supported in sgd_weights; "
            "skipping fine-tuning for this batch."
        )
        return 0

    sgd_steps = int(raw_steps)
    if sgd_steps <= 0:
        return 0

    mask_tensor = mask
    if mask_tensor is not None:
        fns = [fn for idx, fn in enumerate(fns) if mask_tensor[idx].any()]
        mask_tensor = mask_tensor[mask_tensor.any(dim=1)]

    parameter_groups: List[Dict[str, object]] = []
    group_to_genome: List[int] = []
    for genome_idx, (_, _, genome) in enumerate(genomes):
        if getattr(config, "device", None) is not None:
            genome.to(device)
        params = list(genome.parameters())
        if not params:
            continue
        parameter_groups.append({"params": params, "lr": getattr(genome, "sgd_lr", lr)})
        group_to_genome.append(genome_idx)
        for param in params:
            param.requires_grad_(True)

    if not fns or not parameter_groups:
        logging.debug("No fitness functions or trainable parameters available; skipping SGD step.")
        return 0

    has_cuda_params = any(
        isinstance(group.get("params"), list) and any(param.is_cuda for param in group["params"])
        for group in parameter_groups
    )
    has_cpu_params = any(
        isinstance(group.get("params"), list) and any(not param.is_cuda for param in group["params"])
        for group in parameter_groups
    )

    fused_requested = getattr(config, "use_fused_optimizer", True)
    use_fused = (
        fused_requested
        and config.device.type == "cuda"
        and has_cuda_params
        and not has_cpu_params
    )
    if fused_requested and not use_fused:
        reason = "parameters remain on CPU" if has_cpu_params else "no CUDA parameters available"
        logging.debug("Disabling fused AdamW for SGD because %s.", reason)

    # Use fused AdamW on CUDA
    try:
        optimizer = torch.optim.AdamW(parameter_groups, lr=lr, weight_decay=config.sgd_l2_reg, fused=use_fused)
    except (TypeError, RuntimeError):
        # Fallback for older PyTorch versions or CPU
        optimizer = torch.optim.AdamW(parameter_groups, lr=lr, weight_decay=config.sgd_l2_reg)
    
    param_snapshots: List[List[torch.Tensor]] = [
        [param.detach().clone() for param in group["params"]]
        for group in parameter_groups
    ]

    for group in parameter_groups:
        for param in group["params"]:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
    stopping = EarlyStopping(
        patience=early_stop or sgd_steps,
        min_delta=getattr(config, "sgd_early_stop_delta", 0.0),
        n_genomes=len(genomes),
        device=config.device,
    )

    progress = trange(sgd_steps, disable=skip_pbar or sgd_steps <= 5, position=1, leave=True, desc=f"{current_gen}: GD")
    stop_mask = torch.zeros(len(genomes), dtype=torch.bool, device=config.device)
    prepared_target = _prepare_target_cache(target, config)

    inputs_detached = inputs.detach() if isinstance(inputs, torch.Tensor) else inputs
    record_tracker = record_passes if record_passes is not None else [0, 0]

    amp_whitelist = set(getattr(config, "sgd_amp_whitelist", {"lpips", "dists"}))
    use_amp = bool(getattr(config, "use_amp", True))
    device_type = torch.device(config.device).type if config.device is not None else "cpu"

    # Determine whether any parameter lives on CUDA; AMP/GradScaler only works there.
    use_cuda_amp = use_amp and device_type == "cuda" and has_cuda_params and not has_cpu_params

    if use_amp and device_type == "cuda" and has_cpu_params:
        logging.warning(
            "Disabling CUDA AMP for SGD step because some parameters remain on CPU; move genomes to CUDA to re-enable."
        )

    # GradScaler for safe mixed precision training when using CUDA autocast
    scaler = torch.cuda.amp.GradScaler() if use_cuda_amp else None

    steps_executed = 0

    # Pre-allocate tensors for reuse across SGD steps
    max_active = len(genomes)
    per_genome_loss_buffer = torch.zeros(max_active, device=config.device, dtype=torch.float32)
    loss_accum_buffer = torch.zeros(1, device=config.device, dtype=torch.float32)

    # Cache frequently accessed config values (avoid repeated getattr calls in hot loop)
    clip_microbatch_size = int(getattr(config, "clip_microbatch_size", 0))
    sgd_clamp_grad = getattr(config, "sgd_clamp_grad", None)
    max_weight = getattr(config, "max_weight", None)

    inputs_per_genome = isinstance(inputs_detached, torch.Tensor) and inputs_detached.dim() > 3

    def _feed_fn(batch_inputs, idx):
        return batch_inputs[idx] if inputs_per_genome else batch_inputs

    if config is not None and getattr(config, "sgd_use_compiled_forward", False):
        render_fn = lambda genomes_batch, inputs_batch: _render_population_compiled(
            genomes_batch, inputs_batch, _feed_fn, config
        )
    else:
        render_fn = lambda genomes_batch, inputs_batch: _render_population_eager(
            genomes_batch, inputs_batch, _feed_fn, config
        )

    if mask_tensor is not None:
        mask_fn = lambda slice_idx: mask_tensor[:, slice_idx]
        chunk_mask_fn = lambda active_mask, chunk_slice: active_mask[:, chunk_slice]
        mask_row_fn = lambda chunk_mask, fn_idx: chunk_mask[fn_idx]
    else:
        mask_fn = lambda slice_idx: None
        chunk_mask_fn = lambda active_mask, chunk_slice: None
        mask_row_fn = lambda chunk_mask, fn_idx: None

    inputs_slice_fn = (lambda slice_idx: inputs_detached[slice_idx]) if inputs_per_genome else (lambda slice_idx: inputs_detached)

    unscale_fn = (lambda: scaler.unscale_(optimizer)) if scaler is not None else (lambda: None)
    backward_fn = (lambda loss: scaler.scale(loss).backward()) if scaler is not None else (lambda loss: loss.backward())
    step_fn = (lambda: (scaler.step(optimizer), scaler.update())) if scaler is not None else (lambda: optimizer.step())
    clamp_fn = (
        lambda: torch.nn.utils.clip_grad_norm_(
            [param for group in parameter_groups for param in group["params"]],
            sgd_clamp_grad,
        )
        if sgd_clamp_grad
        else (lambda: None)
    )
    weight_clamp_fn = (
        lambda: [param.data.clamp_(-max_weight, max_weight)
                 for group in parameter_groups for param in group["params"]]
        if max_weight
        else (lambda: None)
    )
    postfix_fn = progress.set_postfix_str if hasattr(progress, "set_postfix_str") else (lambda _: None)

    for step_idx in progress:
        active_indices = (~stop_mask).nonzero(as_tuple=True)[0]
        if active_indices.numel() == 0:
            break

        steps_executed = step_idx + 1
        active_genomes = [genomes[i][2] for i in active_indices.tolist()]
        active_inputs = inputs_slice_fn(active_indices)
        active_target = prepared_target[active_indices]
        active_mask = mask_fn(active_indices)

        optimizer.zero_grad(set_to_none=True)

        total_active = len(active_genomes)
        per_genome_loss = per_genome_loss_buffer[:total_active]
        per_genome_loss.zero_()
        loss_accum = loss_accum_buffer
        loss_accum.zero_()
        microbatch = clip_microbatch_size or total_active

        for offset in range(0, total_active, microbatch):
            chunk_slice = slice(offset, min(offset + microbatch, total_active))
            chunk_indices = active_indices[chunk_slice]
            chunk_genomes = [genomes[i][2] for i in chunk_indices.tolist()]

            chunk_inputs = inputs_slice_fn(chunk_slice)
            chunk_target = active_target[chunk_slice]
            chunk_mask = chunk_mask_fn(active_mask, chunk_slice)

            imgs = render_fn(chunk_genomes, chunk_inputs)
            imgs = prep_images(imgs, config)
            if not chunk_target.is_contiguous():
                chunk_target = chunk_target.contiguous()

            normed = imgs.new_zeros((imgs.shape[0], len(fns)), dtype=torch.float32)
            with feature_cache_scope():
                for fn_idx, fn in enumerate(fns):
                    fn_name = getattr(fn, "__name__", "")
                    mask_row = mask_row_fn(chunk_mask, fn_idx)
                    normed_fit = _evaluate_function_masked(
                        fn,
                        fn_name,
                        imgs,
                        chunk_target,
                        mask_row,
                        use_cuda_amp=use_cuda_amp,
                        amp_whitelist=amp_whitelist,
                        normalizer=normalizer,
                        norm=norm,
                        config=config,
                    )
                    normed[:, fn_idx] = normed_fit

            normed = torch.where(torch.isfinite(normed), normed, torch.zeros_like(normed))

            loss_per_example = (1.0 - normed).mean(dim=1)
            loss_per_example = torch.where(
                torch.isfinite(loss_per_example),
                loss_per_example,
                torch.ones_like(loss_per_example),
            )
            loss_mean = loss_per_example.mean().to(dtype=torch.float32)

            record_tracker[0] += len(chunk_genomes)
            chunk_weight = loss_per_example.numel() / max(1, total_active)
            scaled_loss = loss_mean * chunk_weight
            backward_fn(scaled_loss)
            record_tracker[1] += len(chunk_genomes)

            per_genome_loss[chunk_slice] = loss_per_example.detach()
            loss_accum += loss_mean.detach() * chunk_weight

        loss_value = loss_accum.squeeze(0)

        if record_loss is not None:
            record_loss[step_idx] = loss_value.item()

        invalid_indices: Set[int] = set()

        unscale_fn()
        clamp_fn()

        sanitized_grad_genomes: Set[int] = set()
        for group_idx, group in enumerate(parameter_groups):
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if not torch.isfinite(grad).all():
                    sanitized_grad_genomes.add(group_to_genome[group_idx])
                    grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        if sanitized_grad_genomes:
            logging.debug(
                "Sanitized non-finite gradients for genomes %s during SGD.",
                sorted(int(idx) for idx in sanitized_grad_genomes),
            )

        for group_idx, group in enumerate(parameter_groups):
            genome_idx = group_to_genome[group_idx]
            grads_finite = True
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if not torch.isfinite(grad).all():
                    grads_finite = False
                    break
            if not grads_finite:
                invalid_indices.add(genome_idx)
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.detach().zero_()

        step_fn()

        for group_idx, group in enumerate(parameter_groups):
            genome_idx = group_to_genome[group_idx]
            params = group["params"]
            has_nonfinite = any(not torch.isfinite(param).all() for param in params)
            if has_nonfinite:
                sanitized = False
                for param in params:
                    if not torch.isfinite(param).all():
                        param.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        sanitized = True
                if sanitized and all(torch.isfinite(param).all() for param in params):
                    logging.debug(
                        "Sanitized non-finite parameter data for genome %s during SGD step.",
                        int(genome_idx),
                    )
                    param_snapshots[group_idx] = [param.detach().clone() for param in params]
                    continue

                invalid_indices.add(genome_idx)
                for param, backup in zip(params, param_snapshots[group_idx]):
                    param.data.copy_(backup)
                    if param in optimizer.state:
                        optimizer.state[param].clear()
                continue
            param_snapshots[group_idx] = [param.detach().clone() for param in params]

        if invalid_indices:
            stop_mask[list(invalid_indices)] = True
            logging.warning(
                "Skipping SGD updates for genomes %s due to non-finite gradients or parameters",
                sorted(int(idx) for idx in invalid_indices),
            )

        weight_clamp_fn()

        stop_updates = stopping.mask_stop(per_genome_loss, indices=active_indices)
        stop_mask[active_indices] |= stop_updates

        if stop_mask.all():
            break

        n_params = sum(len(list(gen.parameters())) for gen in active_genomes)
        postfix_fn(
            f"loss={loss_value.item():.4f}, {n_params}p, {len(active_genomes)}/{len(genomes)} gs"
        )
    progress.close()

    return steps_executed


def sgd_weights_no_branch(
    genomes,
    mask,
    inputs,
    target,
    fns,
    norm,
    config,
    early_stop: int = 3,
    record_loss: Optional[torch.Tensor] = None,
    skip_pbar: bool = False,
    current_gen: int = 0,
    unequal_shape: bool = False,
    record_passes: Optional[List[int]] = None,
    normalizer=None,
) -> int:
    """SGD loop without branch-based early exits (fixed-step execution)."""

    device = torch.device(config.device) if config.device is not None else torch.device("cpu")
    if isinstance(inputs, torch.Tensor):
        inputs = _ensure_tensor_on_device(inputs, device)
    if isinstance(target, torch.Tensor):
        target = _ensure_tensor_on_device(target, device)

    lr = float(getattr(config, "sgd_learning_rate", 0.0))
    raw_steps = getattr(config, "sgd_steps", 0)
    if isinstance(raw_steps, str):
        logging.warning(
            "String-based SGD schedules are no longer supported in sgd_weights; "
            "skipping fine-tuning for this batch."
        )
        return 0

    sgd_steps = int(raw_steps)
    if sgd_steps <= 0:
        return 0

    mask_tensor = mask
    if mask_tensor is not None:
        fns = [fn for idx, fn in enumerate(fns) if mask_tensor[idx].any()]
        mask_tensor = mask_tensor[mask_tensor.any(dim=1)]

    parameter_groups: List[Dict[str, object]] = []
    group_to_genome: List[int] = []
    for genome_idx, (_, _, genome) in enumerate(genomes):
        if getattr(config, "device", None) is not None:
            genome.to(device)
        params = list(genome.parameters())
        if not params:
            continue
        parameter_groups.append({"params": params, "lr": getattr(genome, "sgd_lr", lr)})
        group_to_genome.append(genome_idx)
        for param in params:
            param.requires_grad_(True)

    if not fns or not parameter_groups:
        logging.debug("No fitness functions or trainable parameters available; skipping SGD step.")
        return 0

    has_cuda_params = any(
        isinstance(group.get("params"), list) and any(param.is_cuda for param in group["params"])
        for group in parameter_groups
    )
    has_cpu_params = any(
        isinstance(group.get("params"), list) and any(not param.is_cuda for param in group["params"])
        for group in parameter_groups
    )

    fused_requested = getattr(config, "use_fused_optimizer", True)
    use_fused = (
        fused_requested
        and config.device.type == "cuda"
        and has_cuda_params
        and not has_cpu_params
    )
    if fused_requested and not use_fused:
        reason = "parameters remain on CPU" if has_cpu_params else "no CUDA parameters available"
        logging.debug("Disabling fused AdamW for SGD because %s.", reason)

    try:
        optimizer = torch.optim.AdamW(parameter_groups, lr=lr, weight_decay=config.sgd_l2_reg, fused=use_fused)
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(parameter_groups, lr=lr, weight_decay=config.sgd_l2_reg)

    param_snapshots: List[List[torch.Tensor]] = [
        [param.detach().clone() for param in group["params"]]
        for group in parameter_groups
    ]

    progress = trange(sgd_steps, disable=skip_pbar or sgd_steps <= 5, position=1, leave=True, desc=f"{current_gen}: GD")
    prepared_target = _prepare_target_cache(target, config)

    inputs_detached = inputs.detach() if isinstance(inputs, torch.Tensor) else inputs
    record_tracker = record_passes if record_passes is not None else [0, 0]

    amp_whitelist = set(getattr(config, "sgd_amp_whitelist", {"lpips", "dists"}))
    use_amp = bool(getattr(config, "use_amp", True))
    device_type = torch.device(config.device).type if config.device is not None else "cpu"
    use_cuda_amp = use_amp and device_type == "cuda" and has_cuda_params and not has_cpu_params

    scaler = torch.cuda.amp.GradScaler() if use_cuda_amp else None

    profile_enabled = bool(getattr(config, "sgd_profile", False))
    profile_steps = max(1, int(getattr(config, "sgd_profile_steps", 1)))
    profile_warmup = max(0, int(getattr(config, "sgd_profile_warmup", 0)))
    profile_start = max(0, int(getattr(config, "sgd_profile_start_step", 0)))
    profile_record_shapes = bool(getattr(config, "sgd_profile_record_shapes", False))
    profile_with_stack = bool(getattr(config, "sgd_profile_with_stack", False))
    profile_memory = bool(getattr(config, "sgd_profile_memory", False))
    profile_with_flops = bool(getattr(config, "sgd_profile_with_flops", False))
    profile_row_limit = max(5, int(getattr(config, "sgd_profile_row_limit", 40)))
    profiler = None
    profiler_ctx = nullcontext()
    profiler_step = lambda: None
    record_fn = (lambda _: nullcontext())

    if profile_enabled and hasattr(torch, "profiler"):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device_type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=profile_start,
                warmup=profile_warmup,
                active=profile_steps,
                repeat=1,
            ),
            record_shapes=profile_record_shapes,
            profile_memory=profile_memory,
            with_stack=profile_with_stack,
            with_flops=profile_with_flops,
        )
        profiler_ctx = profiler
        profiler_step = profiler.step
        record_fn = torch.profiler.record_function
        try:
            logging.warning("SGD profiling enabled: will record %d active step(s) after warmup; summary saved to sgd_profile_<pid>.txt", profile_steps)
        except Exception:
            pass

    steps_executed = 0

    max_active = len(genomes)
    per_genome_loss_buffer = torch.zeros(max_active, device=config.device, dtype=torch.float32)
    loss_accum_buffer = torch.zeros(1, device=config.device, dtype=torch.float32)

    clip_microbatch_size = int(getattr(config, "clip_microbatch_size", 0))
    sgd_clamp_grad = getattr(config, "sgd_clamp_grad", None)
    max_weight = getattr(config, "max_weight", None)

    inputs_per_genome = isinstance(inputs_detached, torch.Tensor) and inputs_detached.dim() > 3

    def _feed_fn(batch_inputs, idx):
        return batch_inputs[idx] if inputs_per_genome else batch_inputs

    if config is not None and getattr(config, "sgd_use_compiled_forward", False):
        render_fn = lambda genomes_batch, inputs_batch: _render_population_compiled(
            genomes_batch, inputs_batch, _feed_fn, config
        )
    else:
        render_fn = lambda genomes_batch, inputs_batch: _render_population_eager(
            genomes_batch, inputs_batch, _feed_fn, config
        )

    if mask_tensor is not None:
        mask_fn = lambda slice_idx: mask_tensor[:, slice_idx]
        chunk_mask_fn = lambda active_mask, chunk_slice: active_mask[:, chunk_slice]
        mask_row_fn = lambda chunk_mask, fn_idx: chunk_mask[fn_idx]
    else:
        mask_fn = lambda slice_idx: None
        chunk_mask_fn = lambda active_mask, chunk_slice: None
        mask_row_fn = lambda chunk_mask, fn_idx: None

    inputs_slice_fn = (lambda slice_idx: inputs_detached[slice_idx]) if inputs_per_genome else (lambda slice_idx: inputs_detached)

    unscale_fn = (lambda: scaler.unscale_(optimizer)) if scaler is not None else (lambda: None)
    backward_fn = (lambda loss: scaler.scale(loss).backward()) if scaler is not None else (lambda loss: loss.backward())
    step_fn = (lambda: (scaler.step(optimizer), scaler.update())) if scaler is not None else (lambda: optimizer.step())
    clamp_fn = (
        lambda: torch.nn.utils.clip_grad_norm_(
            [param for group in parameter_groups for param in group["params"]],
            sgd_clamp_grad,
        )
        if sgd_clamp_grad
        else (lambda: None)
    )
    weight_clamp_fn = (
        lambda: [param.data.clamp_(-max_weight, max_weight)
                 for group in parameter_groups for param in group["params"]]
        if max_weight
        else (lambda: None)
    )
    postfix_fn = progress.set_postfix_str if hasattr(progress, "set_postfix_str") else (lambda _: None)
    record_loss_fn = (lambda idx, value: record_loss.__setitem__(idx, value)) if record_loss is not None else (lambda idx, value: None)

    full_indices = torch.arange(len(genomes), device=config.device)
    active_mask = mask_fn(full_indices)
    active_inputs = inputs_slice_fn(full_indices)
    active_target = prepared_target[full_indices]

    with profiler_ctx:
        for step_idx in progress:
            steps_executed = step_idx + 1
            active_genomes = [genomes[i][2] for i in full_indices.tolist()]

            with record_fn("sgd.zero_grad"):
                optimizer.zero_grad(set_to_none=True)

            total_active = len(active_genomes)
            per_genome_loss = per_genome_loss_buffer[:total_active]
            per_genome_loss.zero_()
            loss_accum = loss_accum_buffer
            loss_accum.zero_()
            microbatch = clip_microbatch_size or total_active

            for offset in range(0, total_active, microbatch):
                chunk_slice = slice(offset, min(offset + microbatch, total_active))
                chunk_genomes = [genomes[i][2] for i in full_indices[chunk_slice].tolist()]

                chunk_inputs = inputs_slice_fn(chunk_slice)
                chunk_target = active_target[chunk_slice]
                chunk_mask = chunk_mask_fn(active_mask, chunk_slice)

                with record_fn("sgd.render"):
                    imgs = render_fn(chunk_genomes, chunk_inputs)
                    imgs = prep_images(imgs, config)
                    if not chunk_target.is_contiguous():
                        if not chunk_target.is_contiguous():
                            chunk_target = chunk_target.contiguous()

                with record_fn("sgd.fitness"):
                    normed = imgs.new_zeros((imgs.shape[0], len(fns)), dtype=torch.float32)
                    with feature_cache_scope():
                        for fn_idx, fn in enumerate(fns):
                            fn_name = getattr(fn, "__name__", "")
                            mask_row = mask_row_fn(chunk_mask, fn_idx)
                            normed_fit = _evaluate_function_masked(
                                fn,
                                fn_name,
                                imgs,
                                chunk_target,
                                mask_row,
                                use_cuda_amp=use_cuda_amp,
                                amp_whitelist=amp_whitelist,
                                normalizer=normalizer,
                                norm=norm,
                                config=config,
                            )
                            normed[:, fn_idx] = normed_fit

                    normed = torch.where(torch.isfinite(normed), normed, torch.zeros_like(normed))

                loss_per_example = (1.0 - normed).mean(dim=1)
                loss_per_example = torch.where(
                    torch.isfinite(loss_per_example),
                    loss_per_example,
                    torch.ones_like(loss_per_example),
                )
                loss_mean = loss_per_example.mean().to(dtype=torch.float32)

                record_tracker[0] += len(chunk_genomes)
                chunk_weight = loss_per_example.numel() / max(1, total_active)
                scaled_loss = loss_mean * chunk_weight
                with record_fn("sgd.backward"):
                    backward_fn(scaled_loss)
                record_tracker[1] += len(chunk_genomes)

                per_genome_loss[chunk_slice] = loss_per_example.detach()
                loss_accum += loss_mean.detach() * chunk_weight

            loss_value = loss_accum.squeeze(0)
            record_loss_fn(step_idx, loss_value.item())

            with record_fn("sgd.optim"):
                unscale_fn()
                clamp_fn()

                for group_idx, group in enumerate(parameter_groups):
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

                step_fn()

                for group_idx, group in enumerate(parameter_groups):
                    params = group["params"]
                    backups = param_snapshots[group_idx]
                    for param, backup in zip(params, backups):
                        param.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        finite_mask = torch.isfinite(param.data)
                        param.data.copy_(torch.where(finite_mask, param.data, backup))
                    param_snapshots[group_idx] = [param.detach().clone() for param in params]

                weight_clamp_fn()

            n_params = sum(len(list(gen.parameters())) for gen in active_genomes)
            postfix_fn(
                f"loss={loss_value.item():.4f}, {n_params}p, {len(active_genomes)}/{len(genomes)} gs"
            )

            profiler_step()

    progress.close()

    if profiler is not None:
        sort_key = "self_cuda_time_total" if device_type == "cuda" else "self_cpu_time_total"
        table = profiler.key_averages().table(sort_by=sort_key, row_limit=profile_row_limit)
        # Log at warning so it's visible by default and also persist to a file for inspection
        logging.warning("SGD profile summary (compiled loop):\n%s", table)
        try:
            pid = os.getpid()
            out_name = f"sgd_profile_{pid}.txt"
            with open(out_name, "w") as fh:
                fh.write("SGD profile summary (compiled loop):\n")
                fh.write(table)
            logging.warning("Wrote SGD profile summary to %s", out_name)
        except Exception:
            logging.exception("Failed to write SGD profile summary to file")
    return steps_executed


def sgd_weights_imaml(
    genomes,
    mask,
    inputs,
    target,
    fns,
    norm,
    config,
    early_stop: int = 3,
    record_loss: Optional[torch.Tensor] = None,
    skip_pbar: bool = False,
    current_gen: int = 0,
    unequal_shape: bool = False,
    record_passes: Optional[List[int]] = None,
    normalizer=None,
) -> int:
    """Compute IMAML-style weight updates by task-specific fine-tuning."""

    if mask is None:
        raise ValueError("IMAML requires a cell-function mask")

    record_tracker = record_passes if record_passes is not None else [0, 0]

    param_deltas: List[List[torch.Tensor]] = []
    total_steps = 0
    losses: List[torch.Tensor] = []

    for task_idx, task in enumerate(fns):
        cloned = [(None, None, genome.clone(config, new_id=False)) for _, _, genome in genomes]
        params_before = [
            [param.detach().clone() for param in clone[2].parameters()]
            for clone in cloned
        ]

        logging.info("IMAML task %d: %s", task_idx, getattr(task, "__name__", str(task)))

        total_steps += sgd_weights(
            cloned,
            None,
            inputs,
            target,
            [task],
            norm,
            config,
            early_stop,
            record_loss,
            skip_pbar,
            current_gen,
            unequal_shape,
            record_tracker,
            normalizer,
        )

        if record_loss is not None:
            losses.append(record_loss.clone())

        params_after = [
            [param.detach().clone() for param in clone[2].parameters()]
            for clone in cloned
        ]

        deltas_for_task: List[torch.Tensor] = []
        for before, after in zip(params_before, params_after):
            task_delta = [a - b for a, b in zip(after, before)]
            deltas_for_task.append(task_delta)
        param_deltas.append(deltas_for_task)

    if record_loss is not None and losses:
        record_loss[:] = torch.stack(losses).mean(dim=0)

    delta_mag = 0.0
    total_index = 0

    for genome_idx in range(len(genomes)):
        for cell_mask in mask.T:
            weighted = []
            for fn_index, deltas_for_task in enumerate(param_deltas):
                weighted.append([delta * cell_mask[fn_index] for delta in deltas_for_task[genome_idx]])
            avg_delta = [torch.stack(layer).mean(dim=0) for layer in zip(*weighted)]
            delta_mag += sum(torch.norm(layer_delta) for layer_delta in avg_delta)

            if total_index < len(genomes):
                genomes[total_index] = (
                    total_index,
                    None,
                    genomes[genome_idx][2].clone(config, new_id=False),
                )
            else:
                genomes.append(
                    (
                        total_index,
                        None,
                        genomes[genome_idx][2].clone(config, new_id=False),
                    )
                )

            for param, delta in zip(genomes[total_index][2].parameters(), avg_delta):
                param.data += delta
            total_index += 1

    if len(genomes) > 0:
        logging.info(
            "IMAML average parameter delta magnitude: %.6f",
            delta_mag / len(genomes),
        )

    return total_steps
