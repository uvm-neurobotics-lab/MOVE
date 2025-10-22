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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from .fitness.feature_cache import feature_cache_scope
from .norm import norm_tensor
from .util import is_canonical_image_batch

__all__ = ["sgd_weights", "sgd_weights_imaml", "prep_images"]

_TARGET_PREP_CACHE: Dict[Tuple[int, Tuple[int, ...], torch.device], torch.Tensor] = {}


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

    assert torch.isfinite(imgs).all(), "NaNs detected in candidate images"
    if is_canonical_image_batch(imgs):
        result = imgs.clone() if copy else imgs
        return result.contiguous()

    working = imgs.clone() if copy else imgs
    working = _ensure_batched_rgb(working)
    if working.dtype != torch.float32:
        working = working.to(dtype=torch.float32)
    working = _resize_to_min(working)
    working = torch.nan_to_num(working, nan=0.0, posinf=1.0, neginf=0.0)
    working = torch.clamp(working, 0.0, 1.0)
    return working.contiguous()


def _prepare_target_cache(target: torch.Tensor, config) -> torch.Tensor:
    """Cache and reuse the prepared target tensor across SGD steps."""

    cache_key = (int(target.data_ptr()), tuple(target.shape), target.device)
    cached = _TARGET_PREP_CACHE.get(cache_key)
    if cached is None or cached.shape != target.shape:
        cached = prep_images(target, config, copy=True).detach()
        _TARGET_PREP_CACHE[cache_key] = cached
    return cached


def _render_population(
    genomes_batch, inputs_batch, unequal_shape: bool
) -> torch.Tensor:
    """Evaluate a set of genomes into image tensors."""

    outputs: List[torch.Tensor] = []
    for idx, genome in enumerate(genomes_batch):
        if (
            unequal_shape
            and isinstance(inputs_batch, torch.Tensor)
            and inputs_batch.dim() > 3
        ):
            feed = inputs_batch[idx]
        else:
            feed = inputs_batch
        outputs.append(
            genome(
                feed,
                force_recalculate=True,
                channel_first=True,
                use_graph=True,
            )
        )
    return torch.stack(outputs)


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
    for _, _, genome in genomes:
        params = list(genome.parameters())
        if not params:
            continue
        parameter_groups.append({"params": params, "lr": getattr(genome, "sgd_lr", lr)})
        for param in params:
            param.requires_grad_(True)

    if not fns or not parameter_groups:
        logging.debug("No fitness functions or trainable parameters available; skipping SGD step.")
        return 0

    optimizer = torch.optim.AdamW(parameter_groups, lr=lr, weight_decay=config.sgd_l2_reg)
    stopping = EarlyStopping(
        patience=early_stop or sgd_steps,
        min_delta=getattr(config, "sgd_early_stop_delta", 0.0),
        n_genomes=len(genomes),
        device=config.device,
    )

    progress = trange(sgd_steps, disable=skip_pbar or sgd_steps <= 5)
    stop_mask = torch.zeros(len(genomes), dtype=torch.bool, device=config.device)
    prepared_target = _prepare_target_cache(target, config)

    inputs_detached = inputs.detach() if isinstance(inputs, torch.Tensor) else inputs
    record_tracker = record_passes if record_passes is not None else [0, 0]

    amp_whitelist = {"lpips", "dists"}
    use_amp = bool(getattr(config, "use_amp", True))
    device_type = torch.device(config.device).type if config.device is not None else "cpu"
    use_cuda_amp = use_amp and device_type == "cuda"

    steps_executed = 0

    for step_idx in progress:
        active_indices = (~stop_mask).nonzero(as_tuple=True)[0]
        if active_indices.numel() == 0:
            break

        steps_executed = step_idx + 1
        active_genomes = [genomes[i][2] for i in active_indices.tolist()]
        if isinstance(inputs_detached, torch.Tensor) and inputs_detached.dim() > 3:
            active_inputs = inputs_detached[active_indices]
        else:
            active_inputs = inputs_detached
        active_target = prepared_target[active_indices]
        active_mask = mask_tensor[:, active_indices] if mask_tensor is not None else None

        optimizer.zero_grad()

        total_active = len(active_genomes)
        per_genome_loss = torch.zeros(total_active, device=config.device, dtype=torch.float32)
        loss_accum = torch.zeros(1, device=config.device, dtype=torch.float32)
        microbatch = int(getattr(config, "clip_microbatch_size", 0) or total_active)

        for offset in range(0, total_active, microbatch):
            chunk_slice = slice(offset, min(offset + microbatch, total_active))
            chunk_indices = active_indices[chunk_slice]
            chunk_genomes = [genomes[i][2] for i in chunk_indices.tolist()]

            if isinstance(active_inputs, torch.Tensor) and active_inputs.dim() > 3:
                chunk_inputs = active_inputs[chunk_slice]
            else:
                chunk_inputs = active_inputs

            chunk_target = active_target[chunk_slice]
            chunk_mask = active_mask[:, chunk_slice] if active_mask is not None else None

            imgs = _render_population(chunk_genomes, chunk_inputs, unequal_shape)
            imgs = prep_images(imgs, config)
            chunk_target = chunk_target.contiguous()

            components: List[torch.Tensor] = []
            with feature_cache_scope():
                for fn in fns:
                    fn_name = getattr(fn, "__name__", "")
                    amp_ctx = (
                        torch.cuda.amp.autocast(dtype=torch.float16)
                        if use_cuda_amp and fn_name in amp_whitelist
                        else nullcontext()
                    )
                    with amp_ctx:
                        fitness = fn(imgs, chunk_target)
                    if fitness.dim() == 0:
                        fitness = fitness.unsqueeze(0)
                    if not torch.isfinite(fitness).all():
                        disable_ctx = (
                            torch.cuda.amp.autocast(enabled=False)
                            if use_cuda_amp and fn_name in amp_whitelist
                            else nullcontext()
                        )
                        with disable_ctx:
                            fitness = fn(imgs, chunk_target)
                            if fitness.dim() == 0:
                                fitness = fitness.unsqueeze(0)
                    fitness = fitness.to(device=imgs.device, dtype=torch.float32, non_blocking=True)
                    if normalizer is not None:
                        normed_fit = normalizer(fn, fitness)
                    else:
                        normed_fit = norm_tensor(fitness, norm, fn.__name__, clamp=True, warn=False)
                    components.append(normed_fit.to(dtype=torch.float32))

            if components:
                normed = torch.stack(components, dim=1)
            else:
                normed = torch.empty((imgs.shape[0], 0), device=imgs.device)

            if chunk_mask is not None:
                normed = normed * chunk_mask.T

            loss_per_example = (1.0 - normed).mean(dim=1)
            loss_mean = loss_per_example.mean().to(dtype=torch.float32)

            record_tracker[0] += len(chunk_genomes)

            chunk_weight = loss_per_example.numel() / max(1, total_active)
            (loss_mean * chunk_weight).backward()

            record_tracker[1] += len(chunk_genomes)

            per_genome_loss[chunk_slice] = loss_per_example.detach()
            loss_accum += loss_mean.detach() * chunk_weight

        loss_value = loss_accum.squeeze(0)

        if record_loss is not None:
            record_loss[step_idx] = loss_value.item()

        if getattr(config, "sgd_clamp_grad", None):
            torch.nn.utils.clip_grad_norm_(parameter_groups, config.sgd_clamp_grad)

        optimizer.step()

        if getattr(config, "max_weight", None):
            for group in parameter_groups:
                for param in group["params"]:
                    param.data.clamp_(-config.max_weight, config.max_weight)

        stop_updates = stopping.mask_stop(per_genome_loss, indices=active_indices)
        stop_mask[active_indices] |= stop_updates

        if stop_mask.all():
            break

        if hasattr(progress, "set_postfix_str"):
            n_params = sum(len(list(gen.parameters())) for gen in active_genomes)
            progress.set_postfix_str(
                f"loss={loss_value.item():.4f}, {n_params}p, {len(active_genomes)}/{len(genomes)} gs"
            )

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
