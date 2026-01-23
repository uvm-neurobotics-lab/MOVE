import copy
import json
import logging
import os
import queue
import threading
from typing import Any, Callable, Optional

import torch
from torchvision.transforms import Resize
import matplotlib.pyplot as plt

from .evolution import CPPNEvolutionaryAlgorithm


class AsyncCpuWorker:
    """Run CPU-side tasks asynchronously to avoid blocking the GPU loop."""

    def __init__(self, name: str = "move-cpu") -> None:
        self._queue: "queue.Queue[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        if self._stop.is_set():
            return
        self._queue.put((fn, args, kwargs))

    def _run(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                fn, args, kwargs = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                fn(*args, **kwargs)
            except Exception:
                logging.exception("CPU worker task failed")
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        self._queue.join()

    def close(self, wait: bool = True) -> None:
        self._stop.set()
        if wait:
            self._queue.join()
            self._thread.join(timeout=5)


class GpuEvolutionMixin:
    """Mixin that offloads CPU-heavy I/O and serialization to a worker thread."""

    _cpu_worker: Optional[AsyncCpuWorker] = None

    def submit_cpu_task(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        worker = getattr(self, "_cpu_worker", None)
        if worker is None:
            fn(*args, **kwargs)
        else:
            worker.submit(fn, *args, **kwargs)

    def flush_cpu_tasks(self) -> None:
        worker = getattr(self, "_cpu_worker", None)
        if worker is not None:
            worker.flush()

    def close_cpu_worker(self) -> None:
        worker = getattr(self, "_cpu_worker", None)
        if worker is not None:
            worker.close()

    def init_dirs(self) -> None:
        print(self.config.run_id)
        self.cond_dir = os.path.join(self.config.output_dir, "conditions", self.config.experiment_condition)
        os.makedirs(self.cond_dir, exist_ok=True)
        self.run_dir = os.path.join(self.cond_dir, f"run_{self.config.run_id:04d}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.image_dir = os.path.join(self.run_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.genomes_dir = os.path.join(self.run_dir, "genomes")
        os.makedirs(self.genomes_dir, exist_ok=True)
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        def _save_config(run_dir: str, config: Any) -> None:
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(copy.deepcopy(config).to_json(), f, indent=4)

        self.submit_cpu_task(_save_config, self.run_dir, self.config)

    def init_target(self) -> None:
        if self.config.target is None:
            return
        if self.target.shape[0] != self.config.initial_batch_size:
            self.target = torch.stack([self.target.squeeze() for _ in range(self.config.initial_batch_size)])

            if len(self.target.shape) > 3:
                self.target = self.target.permute(0, 3, 1, 2)
            else:
                self.target = self.target.unsqueeze(1).repeat(1, 3, 1, 1)

        if self.target.shape[-2] < 33 or self.target.shape[-1] < 33:
            self.target = Resize((33, 33), antialias=True)(self.target)

        self.target = torch.clamp(self.target, 0, 1)

        target_path = os.path.join(self.run_dir, "target.png")

        def _save_target(path: str, target: torch.Tensor) -> None:
            plt.imsave(path, target[0].permute(1, 2, 0).detach().cpu().numpy())

        self.submit_cpu_task(_save_target, target_path, self.target)


class CPPNEvolutionaryAlgorithmGPU(GpuEvolutionMixin, CPPNEvolutionaryAlgorithm):
    """GPU-first evolutionary loop with async CPU offload."""

    def __init__(self, config, debug_output: bool = False) -> None:
        self._cpu_worker = AsyncCpuWorker()
        super().__init__(config, debug_output)
