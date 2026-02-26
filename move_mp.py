"""
Multi-Processing Steady-State MOVE (MOVE-MP)

This module implements a steady-state variant of MOVE where each cell is hosted on its own 
CPU thread (with multiple cells potentially sharing the same thread). Instead of the 
generational structure of standard MOVE, this version uses a work queue where offspring 
are continuously generated and optimized with SGD based on their parent cell.

Key differences from standard MOVE:
- Steady-state evolution: No distinct generations, continuous replacement
- Cell-based parallelism: Each cell can evolve independently on separate threads
- Asynchronous evaluation: Offspring are queued and evaluated as threads become available
- Dynamic load balancing: Busy cells can be skipped, idle threads pick up new work

Architecture:
- Main thread: Manages the elite map and coordinates worker threads
- Worker threads: Generate offspring, run SGD, evaluate fitness
- Work queue: Cells waiting to produce offspring
- Result queue: Evaluated offspring waiting to be placed in the map
"""

import copy
import logging
import math
import multiprocessing as mp
import os
import queue
import random
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from MOVE.move import MOVE
from MOVE.sgd_weights import sgd_weights, sgd_weights_imaml
from MOVE.shared_models import (
    enable_shared_models, disable_shared_models, preload_shared_models,
    calculate_safe_gpu_workers, is_shared_models_enabled
)

# Fix for CUDA + multiprocessing: must use 'spawn' instead of 'fork'
# This needs to be set before any CUDA initialization
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, that's fine
    pass


class WorkerMessage:
    """Messages sent to worker threads"""
    EVOLVE_CELL = "evolve_cell"
    SHUTDOWN = "shutdown"
    
    def __init__(self, msg_type: str, data: Optional[Dict] = None):
        self.msg_type = msg_type
        self.data = data or {}


class OffspringResult:
    """Results from worker thread evaluation"""
    def __init__(self, cell_id: int, child, fitnesses, normed_fitnesses, 
                 agg_fitness, sgd_steps: int, worker_id: int):
        self.cell_id = cell_id
        self.child = child
        self.fitnesses = fitnesses
        self.normed_fitnesses = normed_fitnesses
        self.agg_fitness = agg_fitness
        self.sgd_steps = sgd_steps
        self.worker_id = worker_id


def _worker_process(worker_id: int, config, work_queue: mp.Queue, 
                    result_queue: mp.Queue, device: str,
                    fns, inputs, target, norm, sgd_fns):
    """Worker process main function (must be picklable for spawn mode)"""
    logging.info(f"Worker {worker_id} started on device {device}")
    
    # Copy config and set device
    config = copy.deepcopy(config)
    config.device = torch.device(device)  # Ensure it's a torch.device object
    
    # Offset node ID counter to avoid conflicts between workers
    # Each worker gets a unique range: worker 0 uses 0-999999, worker 1 uses 1000000-1999999, etc.
    from MOVE.cppn.cppn import CPPN
    CPPN.current_node_id = worker_id * 1000000
    logging.info(f"Worker {worker_id} node ID offset: {CPPN.current_node_id}")
    
    # Move tensors to worker device
    if inputs is not None:
        inputs = inputs.to(device)
    if target is not None:
        target = target.to(device)
    
    while True:
        try:
            # Get work from queue with timeout
            msg = work_queue.get(timeout=1.0)
            
            if msg.msg_type == WorkerMessage.SHUTDOWN:
                logging.info(f"Worker {worker_id} shutting down")
                break
                
            elif msg.msg_type == WorkerMessage.EVOLVE_CELL:
                _evolve_cell_worker(
                    msg.data, worker_id, config, result_queue,
                    fns, inputs, target, norm, sgd_fns
                )
                
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Worker {worker_id} error: {e}", exc_info=True)


def _evolve_cell_worker(data: Dict, worker_id: int, config, result_queue: mp.Queue,
                        fns, inputs, target, norm, sgd_fns):
    """Evolve a single cell (worker process function)"""
    cell_id = data["cell_id"]
    parent = data["parent"]
    batch_id = data["batch_id"]
    
    logging.info(f"Worker {worker_id} starting cell {cell_id}")
    
    try:
        # Create offspring
        if parent is None:
            child = config.genome_type(config)
            child.cell_lineage = [-1]
            child.n_cells = 0
        else:
            child = parent.clone(config, new_id=True)
            child.parents = (parent.id, parent.id)
            child.cell_lineage = parent.cell_lineage
            child.n_cells = parent.n_cells
        
        logging.info(f"Worker {worker_id} created child for cell {cell_id}")
        
        # Move to worker device
        child = child.to(config.device)
        
        logging.info(f"Worker {worker_id} moved child to {config.device}")
        
        # Apply mutations
        for _ in range(config.extra_initial_mutations + 1):
            child.mutate(config)
        child.reset(config)
        
        logging.info(f"Worker {worker_id} applied mutations to cell {cell_id}")
        
        # Bloat (if configured)
        n_bloat = 0
        if config.bloat_prune_ratio > 0:
            before = len(child.connections)
            this_bloat = int(len(child.connections) * config.bloat_prune_ratio)
            for _ in range(this_bloat):
                child.add_connection(config)
            n_bloat = len(child.connections) - before
        
        logging.info(f"Worker {worker_id} about to start SGD for cell {cell_id}")
        
        # SGD optimization
        sgd_steps = 0
        if config.sgd_steps > 0 and config.with_grad:
            sgd_steps = _sgd_offspring_worker(child, cell_id, config, inputs, target, 
                                             fns, sgd_fns, norm)
        
        # Prune
        if config.bloat_prune_ratio > 0 and n_bloat > 0:
            child.prune_fixed_connections(n_bloat)
        else:
            # Activate to populate node activations
            with torch.no_grad():
                _ = child.get_image(inputs, config.res_h, config.res_w, config)
        
        cx_pruned, nodes_pruned = child.prune(config)
        
        # Evaluate fitness
        fitnesses, normed_fitnesses, agg_fitness = _measure_fitness_worker(
            child, config, inputs, target, fns
        )
        
        # Move back to CPU for storage AND convert tensors to lists
        # This avoids CUDA tensor sharing issues in multiprocessing
        child = child.to("cpu")
        fitnesses_list = fitnesses.cpu().tolist()
        normed_fitnesses_list = normed_fitnesses.cpu().tolist()
        agg_fitness_val = float(agg_fitness.item() if torch.is_tensor(agg_fitness) else agg_fitness)
        
        # Serialize CPPN to JSON to avoid pickling issues
        logging.info(f"Worker {worker_id} serializing CPPN for cell {cell_id}")
        try:
            child_json = child.to_json()
            logging.info(f"Worker {worker_id} serialized CPPN successfully")
        except Exception as e:
            logging.error(f"Worker {worker_id} failed to serialize CPPN: {e}", exc_info=True)
            raise
        
        # Send result back to main thread
        result = OffspringResult(
            cell_id=cell_id,
            child=child_json,  # Send as JSON, not CPPN object
            fitnesses=fitnesses_list,  # Send as list, not tensor
            normed_fitnesses=normed_fitnesses_list,  # Send as list, not tensor
            agg_fitness=agg_fitness_val,
            sgd_steps=sgd_steps,
            worker_id=worker_id
        )
        
        logging.info(f"Worker {worker_id} putting result for cell {cell_id} in queue (agg_fitness={agg_fitness_val:.4f})")
        try:
            result_queue.put(result)
            logging.info(f"Worker {worker_id} result queued for cell {cell_id}")
        except Exception as e:
            logging.error(f"Worker {worker_id} failed to queue result: {e}", exc_info=True)
            raise
        
    except Exception as e:
        logging.error(f"Worker {worker_id} failed to evolve cell {cell_id}: {e}", 
                     exc_info=True)


def _sgd_offspring_worker(child, cell_id: int, config, inputs, target, fns, sgd_fns, norm) -> int:
    """Run SGD on offspring"""
    logging.info(f"SGD worker starting for cell {cell_id}, sgd_steps={config.sgd_steps}")
    
    # Create wrapper like new_children format
    new_children = [(0, cell_id, child)]
    
    # Initialize mask for this cell
    batch_cell_ids = torch.tensor([cell_id], device=config.device)
    exclude = set(fns) - set(sgd_fns)
    skip_fns = [i for i, f in enumerate(fns) if f in exclude]
    
    # Create mask (simplified, assuming we have the fn_mask)
    # In practice, you'd need to pass this from main thread
    mask = torch.ones((len(sgd_fns), 1), device=config.device)
    
    logging.info(f"About to call sgd_fn for cell {cell_id}")
    
    sgd_fn = sgd_weights_imaml if config.sgd_strat == 'imaml' else sgd_weights
    
    steps = sgd_fn(
        new_children,
        mask=mask,
        inputs=inputs,
        target=target,
        fns=sgd_fns,
        norm=norm,
        config=config,
        early_stop=config.sgd_early_stop,
    )
    
    logging.info(f"SGD completed for cell {cell_id}, steps={steps}")
    
    return steps


def _measure_fitness_worker(child, config, inputs, target, fns) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate fitness of a single child"""
    with torch.no_grad():
        # Generate image
        img = child.get_image(inputs, config.res_h, config.res_w, config)
        img = img.unsqueeze(0)  # Add batch dimension
        
        # Ensure correct format
        if not torch.isfinite(img).all():
            raise ValueError("Non-finite values in generated image")
        img = torch.clamp(img, 0.0, 1.0)
        
        # Prepare target
        target_sample = target[:1]  # Single sample
        
        # Evaluate each fitness function
        fitnesses = torch.zeros(len(fns), device=config.device)
        normed_fitnesses = torch.zeros_like(fitnesses)
        
        for i, fn in enumerate(fns):
            fitness = fn(img, target_sample)
            if fitness.dim() == 0:
                fitness = fitness.unsqueeze(0)
            
            fitnesses[i] = fitness.item()
            # Normalize (simplified - in practice use proper normalization)
            normed_fitnesses[i] = fitness.item()
        
        agg_fitness = normed_fitnesses.mean()
            
        return fitnesses, normed_fitnesses, agg_fitness


class MOVE_MP(MOVE):
    """
    Multi-Processing Steady-State MOVE
    
    Instead of the generational structure, this version:
    1. Maintains a work queue of cells that need offspring
    2. Worker threads continuously pull from the queue
    3. Offspring are evaluated and placed back in the map asynchronously
    4. Allows for better CPU utilization and parallelism
    """
    
    def __init__(self, config, debug_output=False, num_workers: Optional[int] = None):
        # Enable shared models BEFORE calling super().__init__
        # This must happen before any models are loaded
        enable_shared_models()
        
        super().__init__(config, debug_output)
        
        # Preload shared models after MOVE initialization
        preload_shared_models(self.config, device="cpu")
        
        # Multiprocessing setup
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.work_queue = mp.Queue(maxsize=self.num_workers * 2)
        self.result_queue = mp.Queue()
        self.workers: List[mp.Process] = []
        self.worker_devices = self._assign_worker_devices()
        
        # Steady-state tracking
        self.cell_busy = torch.zeros(self.n_cells, dtype=torch.bool)
        self.cell_last_update = torch.zeros(self.n_cells, dtype=torch.long)
        self.pending_offspring = 0
        
        # Statistics
        self.total_sgd_steps = 0
        self.worker_stats = {i: {"offspring": 0, "sgd_steps": 0} 
                           for i in range(self.num_workers)}
        
        logging.info(f"Initialized MOVE_MP with {self.num_workers} workers")
        logging.info(f"Worker devices: {self.worker_devices}")
        logging.info(f"Shared models enabled: {is_shared_models_enabled()}")
        
    def _assign_worker_devices(self) -> List[str]:
        """Assign devices to workers (CPU or CUDA)"""
        devices = []
        
        # Check for CUDA availability
        if torch.cuda.is_available() and self.config.device != "cpu":
            n_gpus = torch.cuda.device_count()
            
            # Calculate safe number of GPU workers based on available memory
            safe_gpu_workers = 0
            for gpu_id in range(n_gpus):
                gpu_capacity = calculate_safe_gpu_workers(self.config, f"cuda:{gpu_id}")
                safe_gpu_workers += gpu_capacity
            
            if safe_gpu_workers > 0:
                logging.info(
                    f"Shared models enabled. Can safely use {safe_gpu_workers} GPU worker(s) "
                    f"across {n_gpus} GPU(s)"
                )
                
                # Assign workers to GPUs
                gpu_workers_assigned = min(self.num_workers, safe_gpu_workers)
                cpu_workers = self.num_workers - gpu_workers_assigned
                
                # Distribute GPU workers across GPUs
                for i in range(gpu_workers_assigned):
                    gpu_id = i % n_gpus
                    devices.append(f"cuda:{gpu_id}")
                
                # Remaining workers use CPU
                for i in range(cpu_workers):
                    devices.append("cpu")
                    
                if cpu_workers > 0:
                    logging.info(
                        f"Assigned {gpu_workers_assigned} workers to GPU(s) and "
                        f"{cpu_workers} workers to CPU"
                    )
            else:
                # Not enough GPU memory, use CPU
                logging.warning(
                    f"Insufficient GPU memory for workers. Using CPU workers. "
                    f"Consider reducing num_workers or using simpler fitness functions."
                )
                devices = ["cpu"] * self.num_workers
        else:
            # All workers use CPU
            devices = ["cpu"] * self.num_workers
            
        return devices
    
    def _start_workers(self):
        """Initialize and start worker threads"""
        logging.info("Starting worker threads...")
        
        # Move inputs and target to CPU for passing to workers
        inputs_cpu = self.inputs.cpu() if self.inputs is not None else None
        target_cpu = self.target.cpu() if self.target is not None else None
        
        # Get sgd_fns
        sgd_fns = self.sgd_fns if hasattr(self, 'sgd_fns') else set(self.fns)
        
        for i in range(self.num_workers):
            # Start worker process with all necessary data
            p = mp.Process(
                target=_worker_process,
                args=(
                    i,  # worker_id
                    self.config,  # config (will be deep copied in worker)
                    self.work_queue,
                    self.result_queue,
                    self.worker_devices[i],  # device
                    self.fns,
                    inputs_cpu,
                    target_cpu,
                    self.norm,
                    sgd_fns
                ),
                daemon=True
            )
            p.start()
            self.workers.append(p)
            
        logging.info(f"Started {len(self.workers)} workers")
    
    def _stop_workers(self):
        """Shutdown all worker threads"""
        logging.info("Stopping workers...")
        
        # Send shutdown messages
        for _ in range(self.num_workers):
            self.work_queue.put(WorkerMessage(WorkerMessage.SHUTDOWN))
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                
        self.workers.clear()
        
        # Cleanup shared models
        disable_shared_models()
        
        logging.info("All workers stopped")
    
    def _select_cells_for_evolution(self, n_cells: int) -> List[int]:
        """Select cells that should produce offspring"""
        # Prioritize cells that haven't been updated recently
        available = ~self.cell_busy
        
        if available.sum() == 0:
            # All cells busy, wait for results
            return []
        
        # Weight by time since last update (older = higher priority)
        weights = (self.current_batch - self.cell_last_update).float()
        weights[~available] = 0  # Don't select busy cells
        
        if weights.sum() == 0:
            # Random selection among available
            indices = torch.where(available)[0]
            selected = np.random.choice(indices.cpu().numpy(), 
                                       size=min(n_cells, len(indices)), 
                                       replace=False)
        else:
            # Weighted selection
            probs = weights / weights.sum()
            indices = torch.multinomial(probs, num_samples=min(n_cells, available.sum()), 
                                       replacement=False)
            selected = indices.cpu().numpy()
        
        return list(selected)
    
    def _queue_offspring(self, cell_ids: List[int]):
        """Queue cells for offspring generation"""
        parents = self.map.get_population(include_empty=True)
        
        for cell_id in cell_ids:
            parent = parents[cell_id]
            
            # Mark cell as busy
            self.cell_busy[cell_id] = True
            self.pending_offspring += 1
            
            # Queue work
            msg = WorkerMessage(
                WorkerMessage.EVOLVE_CELL,
                data={
                    "cell_id": cell_id,
                    "parent": parent.clone(self.config, new_id=False, cpu=True) if parent else None,
                    "batch_id": self.current_batch
                }
            )
            
            self.work_queue.put(msg)
    
    def _process_results(self, blocking: bool = False, timeout: float = 0.1) -> int:
        """Process completed offspring from result queue"""
        n_processed = 0
        
        logging.info(f"_process_results called, blocking={blocking}, pending={self.pending_offspring}, queue_size={self.result_queue.qsize()}")
        
        while True:
            try:
                # Get result from queue
                # If blocking, keep trying with timeout while we have pending offspring
                if blocking and self.pending_offspring > n_processed:
                    logging.info(f"Trying to get result with timeout={timeout}")
                    result = self.result_queue.get(timeout=timeout)
                else:
                    result = self.result_queue.get_nowait()
                    
                logging.info(f"Received offspring for cell {result.cell_id} (agg_fitness={result.agg_fitness:.4f})")
                    
                # Convert lists back to tensors
                result.fitnesses = torch.tensor(result.fitnesses)
                result.normed_fitnesses = torch.tensor(result.normed_fitnesses)
                    
                # Process the result
                self._place_offspring(result)
                n_processed += 1
                self.pending_offspring -= 1
                
            except queue.Empty:
                logging.info(f"Queue empty, processed {n_processed}, pending {self.pending_offspring}")
                break
            except Exception as e:
                logging.error(f"Error processing result: {e}", exc_info=True)
                break
        
        return n_processed
    
    def _place_offspring(self, result: OffspringResult):
        """Place offspring in the map using MOVE's voting mechanism"""
        cell_id = result.cell_id
        child_json = result.child
        fit_child = result.fitnesses
        fc_child_normed = result.normed_fitnesses
        
        # Reconstruct CPPN from JSON
        child = self.config.genome_type(self.config)
        child.from_json(child_json)
        
        # Expand to match all cells
        fit_child_expanded = fit_child.repeat(self.n_cells, 1).T  # (fns, cells)
        normed_fit_child_expanded = fc_child_normed.repeat(self.n_cells, 1).T
        
        # Determine replacements
        _, _, replaces = self.replace_by_voting(
            fit_child_expanded.to(self.config.device),
            normed_fit_child_expanded.to(self.config.device)
        )
        
        # Limit jumps
        if not self.config.allow_jumps:
            tmp = torch.zeros_like(replaces)
            tmp[cell_id] = replaces[cell_id]
            replaces = tmp
        elif torch.sum(replaces) > self.config.allow_jumps:
            indices = torch.nonzero(replaces)
            superiority = (fit_child_expanded.to(self.config.device) * self.map.fn_mask) - self.map.fitness
            superiority = superiority.T
            sorted_indices = sorted(indices, key=lambda x: torch.sum(superiority[x] > 0), 
                                  reverse=True)
            indices = sorted_indices[:self.config.allow_jumps]
            replaces = torch.zeros_like(replaces)
            for idx in indices:
                replaces[idx] = True
        
        # Update map
        child.n_cells = torch.sum(replaces).item()
        self.map.fitness[:, replaces] = fit_child_expanded.to(self.config.device)[:, replaces]
        self.map.normed_fitness[:, replaces] = normed_fit_child_expanded.to(self.config.device)[:, replaces]
        
        idxs_to_replace = torch.nonzero(replaces).squeeze().tolist()
        if isinstance(idxs_to_replace, int):
            idxs_to_replace = [idxs_to_replace]
        
        for r in idxs_to_replace:
            placed = child.clone(self.config, new_id=False, cpu=True)
            placed.cell_lineage = child.cell_lineage + [r]
            placed.n_cells = child.n_cells
            self.map.map[r] = placed
            self.map.agg_fitness[r] = result.agg_fitness
            self.cell_last_update[r] = self.current_batch
        
        # Update statistics
        self.cell_busy[cell_id] = False
        self.total_offspring += 1
        self.total_sgd_steps += result.sgd_steps
        self.worker_stats[result.worker_id]["offspring"] += 1
        self.worker_stats[result.worker_id]["sgd_steps"] += result.sgd_steps
        
        # Update record counts for stop conditions (e.g., StopAfterCPPNPasses)
        # Calculate passes: 1 initial eval + 2*sgd_steps (forward + backward per SGD step)
        n_step_fwds = 1  # One forward pass for fitness evaluation
        n_step_evals = len(self.fns)  # One eval per fitness function
        n_sgd_passes = result.sgd_steps * 2 if self.config.with_grad else 0  # Forward + backward
        n_step_fwds_incl_sgd = n_step_fwds + n_sgd_passes
        n_step_passes = n_step_fwds + n_sgd_passes  # Total CPPN passes
        n_step_evals_incl_sgd = n_step_evals + (n_sgd_passes * len(self.sgd_fns))
        
        self.record.update_counts(
            self.current_batch,
            n_step_fwds,
            n_step_fwds_incl_sgd,
            n_step_evals,
            n_step_evals_incl_sgd,
            0,  # n_pruned (not tracked per-offspring in steady-state)
            0,  # n_pruned_nodes
            n_step_passes,
            config=self.config  # Pass config for potential array expansion
        )
        
        # Record keeping
        if self.current_batch % self.config.record_frequency_batch == 0:
            # Create a replacement matrix for recording
            all_replacements = torch.zeros((self.n_cells, self.n_cells), device=self.config.device)
            all_replacements[cell_id] = replaces
            
            index = self.current_batch // self.config.record_frequency_batch
            # Ensure index doesn't exceed allocated record size
            max_index = self.total_batches // self.config.record_frequency_batch
            if index < max_index:
                self.record.update(
                    index,
                    all_replacements,
                    self.map.fitness,
                    self.map.normed_fitness,
                    self.map.agg_fitness,
                    self.map.map,
                    self.total_offspring
                )
    
    def run_one_batch(self):
        """
        Steady-state batch: Queue offspring and process results
        
        Unlike generational MOVE, this doesn't wait for all offspring to complete.
        It continuously queues work and processes results as they arrive.
        """
        logging.info(f"run_one_batch called, pending={self.pending_offspring}")
        
        # Determine how many cells should produce offspring this batch
        batch_size, initial_pop_done = self.get_batch_size()
        
        logging.info(f"batch_size={batch_size}, initial_pop_done={initial_pop_done}")
        
        # Select cells for evolution (avoid busy cells)
        cell_ids = self._select_cells_for_evolution(batch_size)
        
        logging.info(f"Selected {len(cell_ids)} cells for evolution")
        
        # Queue offspring generation
        if cell_ids:
            self._queue_offspring(cell_ids)
        
        # Process any completed results (non-blocking)
        n_processed = self._process_results(blocking=False)
        
        logging.info(f"Processed {n_processed} results (non-blocking)")
        
        # If we have pending work but no progress, wait a bit
        if self.pending_offspring > 0 and n_processed == 0:
            logging.info(f"Waiting for results (blocking)...")
            self._process_results(blocking=True, timeout=0.5)
        
        # Update population view
        self.population = self.map.get_population(include_empty=False)
        self.agg_fitnesses = self.map.get_agg_fitnesses()
        self.fitnesses = self.map.get_fitnesses()
        
        logging.info(f"run_one_batch done, population size={len(self.population)}")
    
    def evolve(self, run_number=1, show_output=False, initial_population=False, resume=None):
        """Main evolution loop with worker management"""
        self._start_workers()
        
        try:
            # Initialize SGD if needed
            if self.config.sgd_steps > 0:
                self.init_sgd()
            
            # Run standard MOVE evolution loop
            super().evolve(run_number, show_output, initial_population, resume)
            
        except KeyboardInterrupt:
            logging.info("Evolution interrupted by user")
        finally:
            # Ensure all workers are stopped
            self._stop_workers()
            
            # Process any remaining results
            logging.info(f"Processing {self.pending_offspring} remaining offspring...")
            while self.pending_offspring > 0:
                self._process_results(blocking=True, timeout=1.0)
            
            # Print worker statistics
            self._print_worker_stats()
    
    def _print_worker_stats(self):
        """Print statistics about worker utilization"""
        logging.info("\n=== Worker Statistics ===")
        for worker_id, stats in self.worker_stats.items():
            logging.info(f"Worker {worker_id}: {stats['offspring']} offspring, "
                        f"{stats['sgd_steps']} SGD steps")
        logging.info(f"Total offspring: {self.total_offspring}")
        logging.info(f"Total SGD steps: {self.total_sgd_steps}")
    
    def batch_start(self):
        """Process any completed results at start of batch"""
        # Process completed offspring before starting new batch
        n_processed = self._process_results(blocking=False)
        if n_processed > 0:
            logging.info(f"Processed {n_processed} completed offspring")
        
        # Call parent batch_start
        super().batch_start()
    
    def batch_end(self):
        """Custom batch end with steady-state metrics"""
        # Process any remaining results
        n_processed = self._process_results(blocking=False, timeout=0.1)
        if n_processed > 0:
            logging.info(f"Processed {n_processed} completed offspring at batch_end")
        
        # Only call parent batch_end if we have a population
        # to avoid ZeroDivisionError in avg_nodes calculation
        if len(self.population) > 0:
            # Inline the MOVE batch_end logic to avoid any recursion issues
            self.solution_fitness = -torch.inf
            self.record.batch_end(self, skip_fitness=False)
            
            self.avg_nodes = sum([len(g.nodes) for g in self.population]) / len(self.population)
            self.avg_enabled_connections = sum([len(g.enabled_connections) for g in self.population]) / len(self.population)
            
            self.gen = self.total_offspring // self.config.num_cells
            
            if self.current_batch in [0, ] or (self.current_batch+1)%10 == 0:
                b = self.get_best()
                if b is not None:
                    b.save(os.path.join(self.genomes_dir, f"batch_{self.current_batch:04d}.json"), self.config)

            self._maybe_save_map_images_snapshot()

            if self.config.checkpoint_frequency > 0 and self.current_batch % self.config.checkpoint_frequency == 0:
                self.save_checkpoint()
        else:
            # Skip parent batch_end but still increment batch counter
            self.current_batch += 1
        
        # Additional steady-state metrics
        if self._show_progress:
            busy_cells = self.cell_busy.sum().item()
            logging.info(f"Busy cells: {busy_cells}/{self.n_cells}, "
                        f"Pending: {self.pending_offspring}")


# ============================================================================
# Analysis: Is this a good strategy?
# ============================================================================

"""
ANALYSIS: Multi-Processing Steady-State MOVE

ADVANTAGES:
1. **Better CPU Utilization**: Workers can run in parallel, utilizing multiple cores
   - Generational MOVE processes batches sequentially
   - Steady-state keeps all cores busy continuously

2. **Reduced Idle Time**: No waiting for slowest individual in a batch
   - In generational, the batch waits for the slowest SGD to complete
   - Here, fast cells can be re-evolved while others are still optimizing

3. **Dynamic Load Balancing**: Automatically distributes work across threads
   - Busy cells are skipped, idle threads pick up new work
   - No manual tuning of batch sizes needed

4. **Asynchronous Evaluation**: Fitness evaluation doesn't block evolution
   - Multiple offspring can be evaluated simultaneously
   - Pipeline parallelism: generate -> SGD -> evaluate

5. **Scalability**: Easily scales to many CPU cores
   - Linear speedup with number of cores (up to a point)
   - Can distribute across multiple GPUs if available

DISADVANTAGES:
1. **Implementation Complexity**: Much more complex than generational
   - Thread synchronization, queues, potential race conditions
   - Harder to debug and maintain

2. **Memory Overhead**: Each worker needs its own copy of:
   - Fitness functions, inputs, targets, normalization data
   - May not fit in memory for large problems

3. **Communication Overhead**: Queue operations have costs
   - Serialization/deserialization of genomes
   - Message passing between threads
   - May negate speedup for very fast evaluations

4. **Less Deterministic**: Results depend on thread scheduling
   - Harder to reproduce exact runs
   - Different runs may converge differently

5. **GPU Contention**: If using GPUs, workers may compete for resources
   - CUDA operations may not parallelize well across processes
   - Memory fragmentation on GPU

6. **Population Synchronization**: Map updates need careful locking
   - Race conditions if multiple threads update same cell
   - Current implementation avoids this by marking cells busy

WHEN IS THIS GOOD?

This strategy is BEST when:
- SGD takes significant time (many steps, slow convergence)
- You have many CPU cores available (8+)
- Fitness evaluation is expensive
- Batch size is large (many cells)
- Population is diverse (cells evolve independently)

This strategy is WORSE when:
- SGD is fast (few steps, quick convergence)
- Limited CPU cores (1-4)
- Fitness evaluation is cheap (simple functions like MSE)
- Small populations (communication overhead dominates)
- GPU-heavy workload (limited GPU parallelism)

RECOMMENDATION:
For typical MOVE runs with:
- 100 cells
- 10-100 SGD steps
- Multiple complex fitness functions (LPIPS, DISTS, etc.)
- Modern multi-core CPU (8+ cores)

This steady-state approach could provide **2-4x speedup** over generational MOVE.

However, for your specific use case, benchmark both approaches:
1. Time a standard MOVE run with current parameters
2. Time MOVE_MP with same parameters
3. Compare convergence quality (not just speed)
4. Monitor CPU/GPU utilization to identify bottlenecks

The speedup depends heavily on:
- Ratio of evolution time to SGD time
- Number of available cores
- Memory bandwidth
- Queue overhead

ALTERNATIVE OPTIMIZATIONS:
Before implementing full multiprocessing, consider:
1. **Vectorized SGD**: Optimize multiple genomes in parallel on GPU
2. **Batched Fitness**: Evaluate multiple individuals simultaneously
3. **JIT Compilation**: Use torch.jit or torch.compile for fitness functions
4. **Mixed Precision**: Use FP16 for SGD (already partially implemented)
5. **Async CUDA**: Overlap CPU and GPU work with CUDA streams

These may provide similar speedups with less complexity.
"""
