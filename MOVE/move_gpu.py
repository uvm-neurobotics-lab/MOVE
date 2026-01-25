import copy
import json
import logging
import os
import time
from typing import Dict
import torch

from .move import MOVE
from .evolution_gpu import AsyncCpuWorker, GpuEvolutionMixin
from .sgd_weights import sgd_weights, sgd_weights_imaml


class MOVEGPU(GpuEvolutionMixin, MOVE):
    """GPU-first MOVE with async CPU-side I/O and serialization."""

    def __init__(self, config, debug_output: bool = False) -> None:
        self._cpu_worker = AsyncCpuWorker()
        super().__init__(config, debug_output)

    @torch.no_grad()
    def replacement(self, new_children, fit_children, fc_normed, batch_cell_ids, agg_fc_normed, initial_pop_done):
        all_replacements = torch.zeros((self.n_cells, self.n_cells), device=self.config.device)
        latest_jump_for_target: Dict[int, int] = {}

        for child_i, cell_i, child in new_children:
            fit_child = fit_children[child_i]
            normed_fit_child = fc_normed[child_i]

            if len(fit_child.shape) == 1:
                fit_child = fit_child.repeat(self.n_cells, 1).T
                normed_fit_child = normed_fit_child.repeat(self.n_cells, 1).T

            if self.use_avg_fit:
                _, _, replaces = self.replace_by_avg_fit(fit_child)
            else:
                _, _, replaces = self.replace_by_voting(fit_child, normed_fit_child)

            if not initial_pop_done and self.config.enforce_initial_fill:
                tmp = torch.zeros_like(replaces)
                use = replaces[torch.isin(torch.arange(self.n_cells, device=self.config.device), batch_cell_ids)]
                tmp[torch.isin(torch.arange(self.n_cells, device=self.config.device), batch_cell_ids)] = use
                replaces = tmp

            if not self.config.allow_jumps:
                tmp = torch.zeros_like(replaces)
                tmp[child_i] = replaces[child_i]
                replaces = tmp
                assert torch.sum(replaces) <= 1
            elif torch.sum(replaces) > self.config.allow_jumps:
                indices = torch.nonzero(replaces)
                superiority = (fit_child * self.map.fn_mask) - self.map.fitness
                superiority = superiority.T
                sorted_indices = sorted(indices, key=lambda x: torch.sum(superiority[x] > 0), reverse=True)
                indices = sorted_indices[: self.config.allow_jumps]
                replaces = torch.zeros_like(replaces)
                for idx in indices:
                    replaces[idx] = True

            child.n_cells = torch.sum(replaces).item()
            all_replacements[cell_i] = replaces
            self.map.fitness[:, replaces] = fit_child[:, replaces].detach()
            self.map.normed_fitness[:, replaces] = normed_fit_child[:, replaces].detach()

            idxs_to_replace = torch.nonzero(replaces).squeeze().tolist()
            if isinstance(idxs_to_replace, int):
                idxs_to_replace = [idxs_to_replace]

            if self.debug_output:
                logging.debug("Replacing cells: %s with %s", idxs_to_replace, child.id)

            source_idx = int(cell_i.item()) if torch.is_tensor(cell_i) else int(cell_i)
            for r in idxs_to_replace:
                target_idx = int(r)
                if target_idx != source_idx:
                    prev_jump = latest_jump_for_target.get(target_idx)
                    if prev_jump is not None:
                        self.record.mark_jump_overridden(prev_jump)
                    jump_index = self.record.log_jump(source_idx, target_idx, self.current_batch)
                    latest_jump_for_target[target_idx] = jump_index
                placed = child.clone(self.config, new_id=False, cpu=False)
                placed.cell_lineage = child.cell_lineage + [r]
                placed.n_cells = child.n_cells
                self.map.map[r] = placed
                self.map.agg_fitness[r] = agg_fc_normed[child_i]
            del child

            if not self.allow_multiple_placements:
                assert torch.sum(replaces) <= 1

        total_replacements = int(torch.sum(all_replacements).item())
        self._update_progress_stat("repl", str(total_replacements))
        return all_replacements

    def sgd_population(self, new_children, batch_cell_ids, *, baseline=None, return_fitness: bool = False):
        steps = 0
        n_passes = [len(new_children), 0]
        after_result = baseline
        before_fit = None

        if self.config.sgd_steps > 0 and self.config.with_grad and (self.current_batch + 1) % self.config.grad_every == 0:
            self.init_sgd(batch_cell_ids)
            if self.config.thread_count > 1:
                raise NotImplementedError("Multiprocessing no longer implemented for MOVE")
            else:
                if baseline is not None:
                    before_fit, _, _ = baseline
                else:
                    before_fit, _, _ = self.measure_fitness(new_children, None)

                sgd_fn = sgd_weights
                if self.config.sgd_strat == "imaml":
                    sgd_fn = sgd_weights_imaml

                steps = sgd_fn(
                    new_children,
                    mask=self.mask,
                    inputs=self.inputs,
                    target=self.target,
                    fns=self.sgd_fns,
                    norm=self.norm,
                    config=self.config,
                    early_stop=self.config.sgd_early_stop,
                    record_passes=n_passes,
                    normalizer=lambda fn, fitness: self._normalize_with_cache(fn, fitness, clamp=True),
                )

                if steps > 0:
                    after_result = self.measure_fitness(new_children, None)
                else:
                    after_result = baseline if baseline is not None else (before_fit, None, None)
        self._update_progress_stat("sgdΔ", None)

        if return_fitness:
            return steps, n_passes, after_result
        return steps, n_passes

    @torch.no_grad()
    def save_checkpoint(self):
        def _save_checkpoint(record, run_dir, checkpoints_dir, map_obj, config, current_batch):
            record.save_checkpoint(run_dir, checkpoints_dir, map_obj, config, current_batch, save_data=True)

        self.submit_cpu_task(
            _save_checkpoint,
            self.record,
            self.run_dir,
            self.checkpoints_dir,
            self.map,
            self.config,
            self.current_batch,
        )
        self.save_move_info()

    @torch.no_grad()
    def save_move_info(self):
        def _save_info(run_dir: str, fn_mask: torch.Tensor, cell_names, fns):
            with open(os.path.join(run_dir, "cell_names.csv"), "w") as f:
                f.write(",".join(cell_names))
            with open(os.path.join(run_dir, "function_names.csv"), "w") as f:
                f.write(",".join([fn.__name__ for fn in fns]))
            torch.save(fn_mask, os.path.join(run_dir, "fn_mask.pt"))

        self.submit_cpu_task(_save_info, self.run_dir, self.map.fn_mask, self.map.cell_names, self.fns)

    @torch.no_grad()
    def batch_end(self):
        self.solution_fitness = -torch.inf
        if hasattr(self, "agg_fitnesses") and len(self.agg_fitnesses) > 0:
            if len(self.population) > 0:
                self.population = sorted(self.population, key=lambda x: self.agg_fitnesses[x.id], reverse=True)
                self.this_gen_best = self.population[0].clone(self.config, cpu=False)
        self.diversity = torch.zeros(1, device=self.config.device)[0]
        self.n_unique = len(set([g.id for g in self.population])) if len(self.population) > 0 else 0

        if len(self.population) > 0:
            if self.agg_fitnesses[self.population[0].id] > self.solution_fitness:
                self.solution = self.population[0]
                self.solution_fitness = self.agg_fitnesses[self.population[0].id]
                self.solution_generation = self.gen
                self.best_genome = self.solution

        if len(self.population) > 0:
            self.avg_nodes = sum([len(g.nodes) for g in self.population]) / len(self.population)
            self.avg_enabled_connections = sum([len(g.enabled_connections) for g in self.population]) / len(self.population)
        else:
            self.avg_nodes = 0
            self.avg_enabled_connections = 0

        self.gen = self.total_offspring // self.config.num_cells

        if self.current_batch in [0] or (self.current_batch + 1) % 10 == 0:
            b = self.get_best()
            if b is not None:
                fname = os.path.join(self.genomes_dir, f"batch_{self.current_batch:04d}.json")
                self.submit_cpu_task(b.save, fname, self.config)

        if self.config.checkpoint_frequency > 0 and self.current_batch % self.config.checkpoint_frequency == 0:
            self.save_checkpoint()

    def on_end(self):
        self.end_time = time.time()
        self.time_elapsed = self.end_time - self.start_time
        print(
            "\n\nEvolution completed with",
            self.gen,
            "generations,",
            self.current_batch + 1,
            "batches, and",
            self.total_offspring,
            "offspring",
            "in",
            self.time_elapsed,
            "seconds",
        )
        print("Wrapping up, please wait...")

        self.run_number = self.config.run_id

        def _save_run_artifacts(run_dir, image_dir, inputs, config, total_offspring, best_img_path, target_path):
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(copy.deepcopy(config).to_json(), f, indent=4)
            with open(os.path.join(run_dir, "total_offspring.txt"), "w") as f:
                f.write(str(total_offspring))
            torch.save(inputs, os.path.join(run_dir, "inputs.pt"))
            with open(os.path.join(run_dir, "target.txt"), "w") as f:
                print("\ntarget_path:", target_path)
                print()
                
                if target_path != None and target_path != "None" and os.path.exists(target_path):
                    print("\nUsing target path:", target_path)
                    f.write(str(target_path))
                elif getattr(config, "clip_text_target", None) is not None:
                    print("\saving clip_text_target: ", str(config.clip_text_target))
                    f.write(str(config.clip_text_target))
                else:
                    print("\nsaving target tensor")
                    f.write(str(config.target))
            self.save_best_img(best_img_path, do_graph=True)

        best_img = os.path.join(self.image_dir, f"best_{self.config.run_id:04d}.png")
        self.submit_cpu_task(
            _save_run_artifacts,
            self.run_dir,
            self.image_dir,
            self.inputs,
            self.config,
            self.total_offspring,
            best_img,
            getattr(self.config, "target_path", None),
        )

        self.submit_cpu_task(self.record.save, self.run_dir, True)
        self.save_move_info()

        if not self.config.dry_run:
            self.submit_cpu_task(self.record.save_map, self.image_dir, self.map, self.config, self.inputs)

        def _save_lineages(run_dir, alg):
            lineages = {i: v for i, v in enumerate(alg.get_lineages())}
            with open(os.path.join(run_dir, "lineages.json"), "w") as f:
                json.dump(lineages, f, indent=4)

        self.submit_cpu_task(_save_lineages, self.run_dir, self)
        self.save_checkpoint()

        self.flush_cpu_tasks()
        self.close_cpu_worker()
