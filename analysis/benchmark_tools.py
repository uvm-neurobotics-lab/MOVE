"""Utilities for benchmarking MOVE runtime hot paths.

These helpers create self-contained MOVE configurations that can be run
without depending on external experiment artifacts.  They are designed
so notebooks and scripts can reuse identical benchmarking logic when
profiling or validating runtime optimisations.
"""
from __future__ import annotations

import shutil
import statistics
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from MOVE.move import MOVE
from MOVE.move_config import MOVEConfig


@dataclass
class BenchmarkResult:
    """Summary of a benchmark run."""

    name: str
    device: str
    durations: List[float]
    fit_shape: Tuple[int, ...]
    norm_shape: Tuple[int, ...]
    agg_shape: Tuple[int, ...]
    extras: Optional[Dict[str, object]] = None

    @property
    def mean(self) -> float:
        return statistics.mean(self.durations)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.durations) if len(self.durations) > 1 else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.durations)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "device": self.device,
            "durations": self.durations,
            "mean": self.mean,
            "stdev": self.stdev,
            "median": self.median,
            "fit_shape": self.fit_shape,
            "norm_shape": self.norm_shape,
            "agg_shape": self.agg_shape,
            "extras": self.extras,
        }

    def __str__(self) -> str:  # pragma: no cover - human readable helper
        return (
            f"BenchmarkResult(name={self.name!r}, device={self.device}, "
            f"mean={self.mean * 1e3:.2f}ms, stdev={self.stdev * 1e3:.2f}ms, "
            f"runs={len(self.durations)}, fit_shape={self.fit_shape}, "
            f"norm_shape={self.norm_shape}, agg_shape={self.agg_shape})"
        )


@contextmanager
def _temporary_output_dir(prefix: str = "benchmark"):
    """Create and cleanup a temporary MOVE output directory."""

    root = Path("tmp_results") / prefix
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"run_{uuid.uuid4().hex}"
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _ensure_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_benchmark_config(
    *,
    device: Optional[torch.device | str] = None,
    num_cells: int = 16,
    objective_functions: Sequence[str] = ("mse", "psnr", "gmsd"),
    with_sgd: bool = False,
    sgd_steps: int = 10,
) -> MOVEConfig:
    """Create a MOVE configuration tailored for micro-benchmarks."""

    cfg = MOVEConfig()
    cfg.device = _ensure_device(device)
    cfg.num_cells = num_cells
    cfg.batch_size = num_cells
    cfg.initial_batch_size = num_cells
    cfg.total_offspring = num_cells
    cfg.objective_functions = list(objective_functions)
    cfg.allow_jumps = torch.inf
    cfg.target = torch.rand(33, 33, 3, device=cfg.device)
    cfg.output_dir = "__benchmark_placeholder__"
    cfg.experiment_condition = "benchmark"
    cfg.low_mem = True
    cfg.record_frequency_batch = 1
    cfg.dry_run = False
    cfg.show_progress = False
    cfg.with_grad = with_sgd
    if with_sgd:
        sgd_steps = max(1, sgd_steps)
        cfg.sgd_steps = sgd_steps
    else:
        cfg.sgd_steps = 0
    if with_sgd:
        cfg.grad_every = 1
        cfg.sgd_early_stop = sgd_steps
        cfg.sgd_learning_rate = 0.01
    else:
        cfg.grad_every = 1
    cfg.norm_df_path = "data/target_fitness_fn_ranges.csv"
    cfg.target_path = "synthetic"
    return cfg


def _build_move(cfg: MOVEConfig) -> MOVE:
    """Instantiate MOVE with an isolated output directory."""

    with _temporary_output_dir() as out_dir:
        cfg.output_dir = out_dir
        move = MOVE(cfg)
    return move


def _make_population(move: MOVE, size: int, *, seed: int) -> List[Tuple[int, int, object]]:
    torch.manual_seed(seed)
    population = []
    for idx in range(size):
        genome = move.genome_type(move.config)
        population.append((idx, idx, genome))
    return population


def benchmark_measure_fitness(
    move: MOVE,
    *,
    batch_size: Optional[int] = None,
    repeats: int = 5,
    seed: int = 0,
    label: str = "measure_fitness",
) -> BenchmarkResult:
    """Benchmark ``MOVE.measure_fitness`` using synthetic genomes."""

    batch_size = batch_size or move.config.batch_size
    durations: List[float] = []
    fit_shape: Optional[Tuple[int, ...]] = None
    norm_shape: Optional[Tuple[int, ...]] = None
    agg_shape: Optional[Tuple[int, ...]] = None

    population = _make_population(move, batch_size, seed=seed)
    genomes = [g for _, _, g in population]
    if move.config.device.type == "cuda":
        torch.cuda.synchronize(move.config.device)

    # Warm-up pass (not timed) to trigger lazy initialisation.
    warm_imgs = move.activate_population(genomes)
    move.measure_fitness(population, warm_imgs)
    if move.config.device.type == "cuda":
        torch.cuda.synchronize(move.config.device)

    for run in range(repeats):
        imgs = move.activate_population(genomes)
        start = time.perf_counter()
        fit_children, fc_normed, agg = move.measure_fitness(population, imgs)
        if move.config.device.type == "cuda":
            torch.cuda.synchronize(move.config.device)
        durations.append(time.perf_counter() - start)
        fit_shape = tuple(fit_children.shape)
        norm_shape = tuple(fc_normed.shape)
        agg_shape = tuple(agg.shape)

    assert fit_shape and norm_shape and agg_shape
    return BenchmarkResult(
        name=label,
        device=str(move.config.device),
        durations=durations,
        fit_shape=fit_shape,
        norm_shape=norm_shape,
        agg_shape=agg_shape,
    )


def benchmark_sgd_population(
    move: MOVE,
    *,
    batch_size: Optional[int] = None,
    repeats: int = 5,
    seed: int = 0,
    label: str = "sgd_population",
) -> BenchmarkResult:
    """Benchmark the SGD fine-tuning loop used by MOVE."""

    if not move.config.with_grad or move.config.sgd_steps <= 0:
        raise ValueError("SGD benchmarking requires config.with_grad=True and sgd_steps>0")

    batch_size = batch_size or move.config.batch_size
    population = _make_population(move, batch_size, seed=seed)
    genomes = [g for _, _, g in population]
    batch_ids = torch.arange(batch_size, device=move.config.device)

    if move.config.device.type == "cuda":
        torch.cuda.synchronize(move.config.device)

    warm_imgs = move.activate_population(genomes)
    fit_children, fc_normed, agg = move.measure_fitness(population, warm_imgs)
    _, _, baseline = move.sgd_population(
        population,
        batch_ids,
        baseline=(fit_children, fc_normed, agg),
        return_fitness=True,
    )
    if baseline is None:
        baseline = (fit_children, fc_normed, agg)

    if move.config.device.type == "cuda":
        torch.cuda.synchronize(move.config.device)

    durations: List[float] = []
    step_counts: List[int] = []
    forward_passes: List[int] = []
    backward_passes: List[int] = []

    for _ in range(repeats):
        start = time.perf_counter()
        steps_taken, passes, baseline = move.sgd_population(
            population,
            batch_ids,
            baseline=baseline,
            return_fitness=True,
        )
        if move.config.device.type == "cuda":
            torch.cuda.synchronize(move.config.device)
        durations.append(time.perf_counter() - start)
        step_counts.append(int(steps_taken))
        if isinstance(passes, (list, tuple)) and len(passes) >= 2:
            forward_passes.append(int(passes[0]))
            backward_passes.append(int(passes[1]))
        else:
            forward_passes.append(0)
            backward_passes.append(0)

    extras: Dict[str, object] = {
        "steps": step_counts,
        "forward_passes": forward_passes,
        "backward_passes": backward_passes,
    }
    if step_counts:
        extras["mean_sgd_steps"] = f"{statistics.mean(step_counts):.2f}"
    if forward_passes:
        extras["mean_forward_passes"] = f"{statistics.mean(forward_passes):.2f}"
    if backward_passes:
        extras["mean_backward_passes"] = f"{statistics.mean(backward_passes):.2f}"

    return BenchmarkResult(
        name=label,
        device=str(move.config.device),
        durations=durations,
        fit_shape=tuple(fit_children.shape),
        norm_shape=tuple(fc_normed.shape),
        agg_shape=tuple(agg.shape),
        extras=extras,
    )


def run_smoke_benchmark(
    *,
    device: Optional[torch.device | str] = None,
    num_cells: int = 16,
    repeats: int = 5,
    seed: int = 0,
    with_sgd: bool = False,
    sgd_steps: int = 10,
) -> BenchmarkResult:
    """Convenience helper that bundles configuration + benchmarking."""

    cfg = make_benchmark_config(
        device=device,
        num_cells=num_cells,
        with_sgd=with_sgd,
        sgd_steps=sgd_steps,
    )
    move = _build_move(cfg)
    if with_sgd:
        return benchmark_sgd_population(move, batch_size=num_cells, repeats=repeats, seed=seed)
    return benchmark_measure_fitness(move, batch_size=num_cells, repeats=repeats, seed=seed)


def _format_table(result: BenchmarkResult) -> str:
    headers = ["metric", "value"]
    rows = [
        ("device", result.device),
        ("runs", len(result.durations)),
        ("mean (ms)", f"{result.mean * 1e3:.2f}"),
        ("stdev (ms)", f"{result.stdev * 1e3:.2f}"),
        ("median (ms)", f"{result.median * 1e3:.2f}"),
        ("fit shape", result.fit_shape),
        ("norm shape", result.norm_shape),
        ("agg shape", result.agg_shape),
    ]
    if result.extras:
        for key, value in result.extras.items():
            if isinstance(value, (list, tuple, dict)):
                continue
            rows.append((key.replace("_", " "), value))
    lines = [" | ".join(headers), " | ".join(["---", "---"])]
    lines.extend(" | ".join(map(str, row)) for row in rows)
    return "\n".join(lines)


def print_benchmark(result: BenchmarkResult) -> None:
    """Pretty-print a benchmark result as a Markdown table."""

    print(_format_table(result))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark MOVE hot paths.")
    parser.add_argument("--device", type=str, default=None, help="Torch device to run on")
    parser.add_argument("--num-cells", type=int, default=16, help="Population / batch size")
    parser.add_argument("--repeats", type=int, default=5, help="Measurement repetitions")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for genome sampling")
    parser.add_argument("--with-sgd", action="store_true", help="Benchmark SGD fine-tuning instead of pure fitness evaluation")
    parser.add_argument("--sgd-steps", type=int, default=10, help="Number of SGD steps to run when --with-sgd is set")
    args = parser.parse_args()

    result = run_smoke_benchmark(
        device=args.device,
        num_cells=args.num_cells,
        repeats=args.repeats,
        seed=args.seed,
        with_sgd=args.with_sgd,
        sgd_steps=args.sgd_steps,
    )
    print_benchmark(result)
