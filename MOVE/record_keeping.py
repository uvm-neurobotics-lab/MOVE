import gzip
import logging
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import Callable, List
import matplotlib.pyplot as plt
import json
import numpy as np
import time
from .cppn.util import *
from .cppn import CPPN

import threading
import queue

class Record():
    def __init__(self, config, n_fns, n_cells, total_batches, low_mem=False) -> None:
        self.low_mem = low_mem
        self.fit_df = pd.DataFrame(columns=['condition', 'target', 'cell', 'run', 'function', 'gen', 'fitness'])
        self.replace_df = pd.DataFrame(columns=['condition','target',  'cell', 'run', 'gen', 'replacements'])
        
        num_data_points = total_batches // config.record_frequency_batch
        self.agg_fitness_by_batch = torch.full((n_cells, num_data_points), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))
        print("Initializing record with size", self.agg_fitness_by_batch.shape)
        self.n_evals = 0
        self.n_fwds = 0
        self.n_fwds_incl_sgd = 0
        self.n_evals_incl_sgd = 0
        self.n_cppn_passes = 0
        self.evals_by_batch = torch.full((num_data_points, 5), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))
        self.normed_fitness_by_batch = torch.full((n_fns, n_cells, num_data_points), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))
        
        if not self.low_mem:
            self.total_pruned = torch.full((num_data_points,2), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))
            self.fitness_by_batch = torch.full((n_fns, n_cells, num_data_points), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))
            # self.votes_by_batch = torch.zeros((n_cells, n_fns, n_cells, num_data_points), device='cpu')
            self.replacements_by_batch = torch.zeros((n_cells, n_cells, num_data_points), device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))
            self.ids_by_batch = torch.full((n_cells, num_data_points), -1, device='cpu', dtype=torch.int64)
            self.parents_by_batch = torch.full((2, n_cells, num_data_points), -1, device='cpu', dtype=torch.int64)
            self.lr_by_batch = torch.full((n_cells, num_data_points), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))
            self.offspring_by_batch = torch.full((num_data_points,), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))
            self.cx_by_batch = torch.full((num_data_points,3), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))
            self.nodes_by_batch = torch.full((num_data_points,3), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))
            self.time_elapsed = torch.full((num_data_points,), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))
            
            self.start_time = time.time()

        self.jump_history: List[List[object]] = []
        
        self.update_queue = queue.Queue()
        self._stop_event = threading.Event()
        self.update_thread = threading.Thread(target=self._update_worker, daemon=True)
        self.update_thread.start()
    
    def expand_arrays(self, additional_batches, config):
        additional_points = additional_batches // config.record_frequency_batch
        print(f"Expanding record arrays by {additional_points} points")
        self.agg_fitness_by_batch = torch.cat(
            (self.agg_fitness_by_batch,
             torch.full((self.agg_fitness_by_batch.shape[0], additional_points), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))),
            dim=1,
        ).pin_memory()
        self.evals_by_batch = torch.cat(
            (self.evals_by_batch,
             torch.full((additional_points, self.evals_by_batch.shape[1]), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))),
            dim=0,
        ).pin_memory()
        self.normed_fitness_by_batch = torch.cat(
            (self.normed_fitness_by_batch,
             torch.full((self.normed_fitness_by_batch.shape[0], self.normed_fitness_by_batch.shape[1], additional_points), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))),
            dim=2,
        ).pin_memory()
        if not self.low_mem:
            self.total_pruned = torch.cat(
                (self.total_pruned,
                 torch.full((self.total_pruned.shape[0]+additional_points, self.total_pruned.shape[1]), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))),
                dim=0,
            ).pin_memory()
            self.fitness_by_batch = torch.cat(
                (self.fitness_by_batch,
                 torch.full((self.fitness_by_batch.shape[0], self.fitness_by_batch.shape[1], additional_points), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))),
                dim=2,
            ).pin_memory()
            # self.votes_by_batch = torch.cat((self.votes_by_batch, torch.zeros((self.votes_by_batch.shape[0], self.votes_by_batch.shape[1], self.votes_by_batch.shape[2], additional_points), device='cpu')), dim=3)
            self.replacements_by_batch = torch.cat(
                (self.replacements_by_batch,
                 torch.zeros((self.replacements_by_batch.shape[0], self.replacements_by_batch.shape[1], additional_points), device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))),
                dim=2,
            ).pin_memory()
            self.ids_by_batch = torch.cat(
                (self.ids_by_batch,
                 torch.full((self.ids_by_batch.shape[0], additional_points), -1, device='cpu', dtype=torch.int64)),
                dim=1,
            )
            self.parents_by_batch = torch.cat(
                (self.parents_by_batch,
                 torch.full((self.parents_by_batch.shape[0], self.parents_by_batch.shape[1], additional_points), -1, device='cpu', dtype=torch.int64)),
                dim=2,
            )
            self.lr_by_batch = torch.cat(
                (self.lr_by_batch,
                 torch.full((self.lr_by_batch.shape[0], additional_points), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))),
                dim=1,
            ).pin_memory()
            self.offspring_by_batch = torch.cat(
                (self.offspring_by_batch,
                 torch.full((self.offspring_by_batch.shape[0]+additional_points,), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))),
                dim=0,
            ).pin_memory()
            self.cx_by_batch = torch.cat(
                (self.cx_by_batch,
                 torch.full((self.cx_by_batch.shape[0]+additional_points, self.cx_by_batch.shape[1]), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))),
                dim=0,
            ).pin_memory()
            self.nodes_by_batch = torch.cat(
                (self.nodes_by_batch,
                 torch.full((self.nodes_by_batch.shape[0]+additional_points, self.nodes_by_batch.shape[1]), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))),
                dim=0,
            ).pin_memory()
            self.time_elapsed = torch.cat(
                (self.time_elapsed,
                 torch.full((self.time_elapsed.shape[0]+additional_points,), -torch.inf, device='cpu', pin_memory=torch.device(config.device)!=torch.device('cpu'))),
                dim=0,
            ).pin_memory()



    def update_counts(self, index, n_step_fwds, n_step_fwds_incl_sgd, n_step_evals, n_step_evals_incl_sgd, n_pruned, n_pruned_nodes, n_step_passes, config=None):
        # check to ensure we don't overflow
        if index >= self.evals_by_batch.shape[0]:
            # expand - use simple object if config not provided
            if config is None:
                class SimpleConfig:
                    record_frequency_batch = 1
                    device = 'cpu'
                config = SimpleConfig()
            self.expand_arrays((index - self.evals_by_batch.shape[0] + 1)*10, config=config)


        self.n_fwds += n_step_fwds
        self.n_fwds_incl_sgd += n_step_fwds_incl_sgd
        self.n_evals += n_step_evals
        self.n_evals_incl_sgd += n_step_evals_incl_sgd
        self.n_cppn_passes += n_step_passes
        if not self.low_mem:
            self.total_pruned[index,0] = n_pruned
            self.total_pruned[index,1] = n_pruned_nodes
        self.evals_by_batch[index,0] = n_step_fwds
        self.evals_by_batch[index,1] = n_step_fwds_incl_sgd
        self.evals_by_batch[index,2] = n_step_evals
        self.evals_by_batch[index,3] = n_step_evals_incl_sgd
        self.evals_by_batch[index,4] = n_step_passes
    

    def log_jump(self, from_cell, to_cell, batch) -> int:
        entry = [int(from_cell), int(to_cell), True, int(batch)]
        self.jump_history.append(entry)
        return len(self.jump_history) - 1


    def mark_jump_overridden(self, jump_index: int) -> None:
        if 0 <= jump_index < len(self.jump_history):
            self.jump_history[jump_index][2] = False


    def update(self, index, all_replacements, fitnesses, normed_fitnesses, agg_fitnesses, population, total_offspring):
        self.update_queue.put((index, all_replacements, fitnesses, normed_fitnesses, agg_fitnesses, population, total_offspring))
        return
        self.agg_fitness_by_batch[:,index] = agg_fitnesses.cpu()
        self.normed_fitness_by_batch[:,:,index] = normed_fitnesses.cpu()
        if not self.low_mem:
            self.fitness_by_batch[:,:,index] = fitnesses.cpu()
            # self.votes_by_batch[:,:,:,index] = all_votes
            self.ids_by_batch[:,index] = torch.tensor([-1 if g is None else g.id for g in population]).cpu()
            self.parents_by_batch[:,:,index] = torch.tensor([[-1 if g is None else g.parents[0] for g in population], [-1 if g is None else g.parents[1] for g in population]]).cpu()
            self.lr_by_batch[:,index] = torch.tensor([-torch.inf if g is None else g.sgd_lr for g in population], dtype=torch.float32)
            self.offspring_by_batch[index] = total_offspring
            self.cx_by_batch[index,0] = torch.mean(torch.tensor([len(g.enabled_connections) for g in population if g is not None], dtype=torch.float32))
            self.nodes_by_batch[index,0] = torch.mean(torch.tensor([len(g.hidden_nodes) for g in population if g is not None], dtype=torch.float32))
            self.cx_by_batch[index,1] = torch.min(torch.tensor([len(g.enabled_connections) for g in population if g is not None], dtype=torch.float32))
            self.nodes_by_batch[index,1] = torch.min(torch.tensor([len(g.hidden_nodes) for g in population if g is not None], dtype=torch.float32))
            self.cx_by_batch[index,2] = torch.max(torch.tensor([len(g.enabled_connections) for g in population if g is not None], dtype=torch.float32))
            self.nodes_by_batch[index,2] = torch.max(torch.tensor([len(g.hidden_nodes) for g in population if g is not None], dtype=torch.float32))
            self.time_elapsed[index] = time.time() - self.start_time
            if all_replacements is not None:
                self.replacements_by_batch[:,:,index] = all_replacements.cpu()
    
    
    def _update_worker(self):
        while not self._stop_event.is_set():
            try:
                # Wait for work or timeout so we can check the stop event
                args = self.update_queue.get(timeout=0.1)
                self._perform_update(*args)
                self.update_queue.task_done()
            except queue.Empty:
                continue    


    def _perform_update(self, index, all_replacements, fitnesses, normed_fitnesses, agg_fitnesses, population, total_offspring):
        agg_src = agg_fitnesses.detach()
        self.agg_fitness_by_batch[:, index].copy_(agg_src, non_blocking=agg_src.is_cuda)

        normed_src = normed_fitnesses.detach()
        self.normed_fitness_by_batch[:, :, index].copy_(normed_src, non_blocking=normed_src.is_cuda)
        if not self.low_mem:
            fitness_src = fitnesses.detach()
            self.fitness_by_batch[:, :, index].copy_(fitness_src, non_blocking=fitness_src.is_cuda)
            self.ids_by_batch[:, index] = torch.tensor([-1 if g is None else g.id for g in population]).cpu()
            self.parents_by_batch[:, :, index] = torch.tensor([[-1 if g is None else g.parents[0] for g in population],
                                                            [-1 if g is None else g.parents[1] for g in population]]).cpu()
            self.lr_by_batch[:, index] = torch.tensor([-torch.inf if g is None else g.sgd_lr for g in population], dtype=torch.float32)
            self.offspring_by_batch[index] = total_offspring
            cx_counts = torch.tensor([len(g.enabled_connections) for g in population if g is not None], dtype=torch.float32)
            node_counts = torch.tensor([len(g.hidden_nodes) for g in population if g is not None], dtype=torch.float32)
            self.cx_by_batch[index] = torch.tensor([torch.mean(cx_counts), torch.min(cx_counts), torch.max(cx_counts)])
            self.nodes_by_batch[index] = torch.tensor([torch.mean(node_counts), torch.min(node_counts), torch.max(node_counts)])
            self.time_elapsed[index] = time.time() - self.start_time
            if all_replacements is not None:
                repl_src = all_replacements.detach().to(self.replacements_by_batch.dtype)
                self.replacements_by_batch[:, :, index].copy_(repl_src, non_blocking=repl_src.is_cuda)
            
        logging.debug("Updated record index %s", index)
    

    def stop_update_thread(self):
        self._stop_event.set()
        self.update_thread.join()


    def save(self, run_dir, plot=True):
        logging.info("Saving record")
        torch.save(self.agg_fitness_by_batch, os.path.join(run_dir, "agg_fitness_by_batch.pt"))
        
        torch.save(self.evals_by_batch, os.path.join(run_dir, "evals_by_batch.pt"))
        torch.save(self.normed_fitness_by_batch, os.path.join(run_dir, "normed_fitness_by_batch.pt"))
        
        with open(os.path.join(run_dir, "evals.csv"), 'w') as f:
            f.write("total_fwds,total_fwds_incl_sgd,total_evals,total_evals_incl_sgd,total_fwds_backs\n")
            f.write(f"{self.n_fwds},{self.n_fwds_incl_sgd},{self.n_evals},{self.n_evals_incl_sgd},{(self.n_fwds_incl_sgd-self.n_fwds)*2}\n")

        jump_df = pd.DataFrame(self.jump_history, columns=["from_cell", "to_cell", "survived", "batch"])
        jump_df.to_csv(os.path.join(run_dir, "jump_history.csv"), index=False)
        
        # plot normed fitness
        if plot:
            plt.close()
            # cut after first nan
            plt.figure(figsize=(10,6))
            plt.rcParams.update({'font.size': 14})
            normed_no_nan = self.normed_fitness_by_batch.clone()
            plt.plot( torch.amax(normed_no_nan.nanmean(dim=0), dim=0), label='Best')
            plt.plot( torch.nanmean(normed_no_nan.nanmean(dim=0), dim=0), label='Mean')
            plt.legend()
            plt.savefig(os.path.join(run_dir, "normed_fitness.png"))
            
        if not self.low_mem:
            torch.save(self.total_pruned[:,0], os.path.join(run_dir, "pruned_cxs.pt"))
            torch.save(self.total_pruned[:,1], os.path.join(run_dir, "pruned_nodes.pt"))
            torch.save(self.fitness_by_batch, os.path.join(run_dir, "fitness_by_batch.pt"))
            torch.save(self.replacements_by_batch, os.path.join(run_dir, "replacements_by_batch.pt"))
            torch.save(self.ids_by_batch, os.path.join(run_dir, "ids_by_batch.pt"))
            torch.save(self.parents_by_batch, os.path.join(run_dir, "parents_by_batch.pt"))
            # torch.save(self.votes_by_batch, os.path.join(run_dir, "votes_by_batch.pt"))
            torch.save(self.lr_by_batch, os.path.join(run_dir, "lr_by_batch.pt"))
            torch.save(self.offspring_by_batch, os.path.join(run_dir, "offspring_by_batch.pt"))
            torch.save(self.cx_by_batch, os.path.join(run_dir, "cx_by_batch.pt"))
            torch.save(self.nodes_by_batch, os.path.join(run_dir, "nodes_by_batch.pt"))
            torch.save(self.time_elapsed, os.path.join(run_dir, "time_elapsed_by_batch.pt"))
            

    def load(self, run_dir):
        logging.info("Loading record")
        self.agg_fitness_by_batch = torch.load(os.path.join(run_dir, "agg_fitness_by_batch.pt"))
        
        self.evals_by_batch = torch.load(os.path.join(run_dir, "evals_by_batch.pt"))
        self.normed_fitness_by_batch = torch.load(os.path.join(run_dir, "normed_fitness_by_batch.pt"))
        
        with open(os.path.join(run_dir, "evals.csv"), 'r') as f:
            f.readline()
            line = f.readline()
            parts = line.split(',')
            self.n_fwds = int(parts[0])
            self.n_fwds_incl_sgd = int(parts[1])
            self.n_evals = int(parts[2])
            self.n_evals_incl_sgd = int(parts[3])

        jump_path = os.path.join(run_dir, "jump_history.csv")
        if os.path.exists(jump_path):
            jump_df = pd.read_csv(jump_path)
            if len(jump_df.columns) > 0:
                if "survived" in jump_df.columns:
                    jump_df["survived"] = jump_df["survived"].apply(
                        lambda v: bool(v)
                        if isinstance(v, (bool, np.bool_))
                        else str(v).strip().lower() in {"true", "1", "yes"}
                    )
                self.jump_history = jump_df[["from_cell", "to_cell", "survived", "batch"]].values.tolist()
            else:
                self.jump_history = []
        else:
            self.jump_history = []
        
        if not self.low_mem:
            self.total_pruned[:,0] = torch.load(os.path.join(run_dir, "pruned_cxs.pt"))
            self.total_pruned[:,1] = torch.load(os.path.join(run_dir, "pruned_nodes.pt"))
            self.fitness_by_batch = torch.load(os.path.join(run_dir, "fitness_by_batch.pt"))
            self.replacements_by_batch = torch.load(os.path.join(run_dir, "replacements_by_batch.pt"))
            self.ids_by_batch = torch.load(os.path.join(run_dir, "ids_by_batch.pt"))
            self.parents_by_batch = torch.load(os.path.join(run_dir, "parents_by_batch.pt"))
            # self.votes_by_batch = torch.load(os.path.join(run_dir, "votes_by_batch.pt"))
            self.lr_by_batch = torch.load(os.path.join(run_dir, "lr_by_batch.pt"))
            self.offspring_by_batch = torch.load(os.path.join(run_dir, "offspring_by_batch.pt"))
            self.cx_by_batch = torch.load(os.path.join(run_dir, "cx_by_batch.pt"))
            self.nodes_by_batch = torch.load(os.path.join(run_dir, "nodes_by_batch.pt"))
            self.time_elapsed = torch.load(os.path.join(run_dir, "time_elapsed_by_batch.pt"))         
    
    
    def save_images(self, images_path, map, config, inputs):
        flat_map = map.get_population()
        map_path = os.path.join(images_path, "final_map")
        os.makedirs(map_path, exist_ok=True)        
        pbar = tqdm(total=len(flat_map), desc="Saving final map...")
        
        imgs = []
        genomes = []
        for i in range(len(flat_map)):
            cell_fns_inds = map.cell_fn_inds[i] # flat
            cell_fns = [map.fns[i] for i in cell_fns_inds]
            if(flat_map[i] is not None):
                individual = flat_map[i]
                original_device = getattr(individual, "device", torch.device("cpu"))
                target_device = torch.device(config.device)
                move_back = original_device != target_device
                individual = individual.to(target_device)
                img = individual(inputs, channel_first=True, act_mode="node").detach().cpu()
                if img.shape[0]<3:
                    img = img.repeat(3, 1, 1)

                try:
                    img = img.permute(1,2,0) # (H,W,C)
                    img = torch.clamp(img,0,1).numpy()
                    imgs.append(img)
                    if map.using_soft_mask:
                        cell_name = map.cell_names[i]
                        name = cell_name+f"+{len(list(individual.enabled_connections))}c"
                    else:
                        name = "_".join([(fn.__name__ if isinstance(fn, Callable) else fn) for fn in cell_fns])+f"{len(list(individual.enabled_connections))}c"
                    name = name + ".png"
                    
                    plt.imsave(os.path.join(map_path, name), img, cmap='gray')
                    plt.close()
                except Exception as e:
                    print(e)
                    pass
                if move_back:
                    individual.to(original_device)
                genomes.append(flat_map[i].clone(config, new_id=False).to_json())
            else:
                genomes.append("null")
            pbar.update(1)
        pbar.close()
        if len(imgs)> 0:
            try:
                average_image = np.mean(imgs, axis=0) 
                plt.imsave(os.path.join(images_path, f"avg_{config.run_id:04d}.png"), average_image, cmap='gray')
            except Exception as e:
                print(e)
                pass

        
    def save_map(self, images_path, map, config, inputs, compress=False, save_path=None):
        # save all images
        flat_map = map.get_population()
        print(f"Saving map with {len(flat_map)} genomes")
        if images_path is not None:
            self.save_images(images_path, map, config, inputs)
        
        genomes = []
        for i in range(len(flat_map)):
            cell_fns_inds = map.cell_fn_inds[i] # flat
            cell_fns = [map.fns[i] for i in cell_fns_inds]
            if(flat_map[i] is not None):
                genomes.append(flat_map[i].clone(config, new_id=False).to_json())
            else:
                genomes.append("null")

        if save_path is None:
            save_path = os.path.join(images_path, "final_map", "map.json")
        if compress:
            with gzip.open(save_path, "wt") as f:
                json.dump(genomes, f)
        else:
            with open(save_path, "w") as f:
                json.dump(genomes, f)
    

    def save_checkpoint(self, run_dir, checkpoint_dir, map, config, current_batch, save_data=False):
        if save_data:
            self.save(run_dir)
        checkpoint_name = os.path.join(checkpoint_dir, f"{current_batch:04d}.json.gz")
        self.save_map(None, map, config,None, compress=True, save_path=checkpoint_name)
    

    def load_checkpoint(self, run_dir, checkpoint_dir, map, config, current_batch=None):
        self.load(run_dir)
        current_batch = current_batch if current_batch is not None else max([int(f.split(".")[0]) for f in os.listdir(checkpoint_dir) if f.endswith(".json.gz")])
        checkpoint_name = os.path.join(checkpoint_dir, f"{current_batch:04d}.json.gz")
        with gzip.open(checkpoint_name, "rt") as f:
            genomes = json.load(f)
            for i, g in enumerate(genomes):
                if g == "null":
                    map.map[i] = None
                else:
                    map.map[i] = CPPN.create_from_json(g, config)
                    map.map[i].to(config.device)
        return current_batch
    

    def batch_end(self, alg, skip_fitness=False):
        if hasattr(alg, 'agg_fitnesses') and len(alg.agg_fitnesses) > 0:
            if len(alg.population) > 0:
                # TODO: this shouldn't reorder alg.population
                alg.population = sorted(alg.population, key=lambda x: alg.agg_fitnesses[x.id], reverse=True) # sort by fitness
                # if alg.config.with_grad:
                    # alg.population[0].discard_grads()
                alg.this_gen_best = alg.population[0].clone(alg.config, cpu=True)  # still sorted by fitness
        
        div_mode = alg.config.get('diversity_mode', None)
        if div_mode == 'full':
            std_distance, avg_distance, max_diff = calculate_diversity_full(alg.population)
        elif div_mode == 'stochastic':
            std_distance, avg_distance, max_diff = calculate_diversity_stochastic(alg.population)
        else:
            std_distance, avg_distance, max_diff = torch.zeros(1)[0], torch.zeros(1)[0], torch.zeros(1)[0]
        alg.diversity = avg_distance
        # n_nodes = get_avg_number_of_hidden_nodes(alg.population)
        # n_connections = get_avg_number_of_connections(alg.population)
        # max_connections = get_max_number_of_connections(alg.population)
        # max_nodes = get_max_number_of_hidden_nodes(alg.population)

        alg.n_unique = len(set([g.id for g in alg.population]))

        if not skip_fitness and len(alg.population) > 0:
            # fitness
            if alg.agg_fitnesses[alg.population[0].id] > alg.solution_fitness: # if the new parent is the best found so far
                alg.solution = alg.population[0]                 # update best solution records
                alg.solution_fitness = alg.agg_fitnesses[alg.population[0].id]
                alg.solution_generation = alg.gen
                alg.best_genome = alg.solution
            
            # alg.save_best_img(os.path.join(alg.run_dir, "images", f"current_best_output.png"))
        # if alg.solution is not None:
        #     plt.close()
        #     plt.plot(self.normed_fitness_by_batch.mean(dim=0).max(dim=0)[0], label='Best')
        #     plt.plot(self.normed_fitness_by_batch.mean(dim=(0,1)), label='Mean')
        #     plt.legend()
        #     plt.xlabel("Batch")
        #     plt.ylabel("Aggregated fitness")
        #     plt.savefig(os.path.join(alg.run_dir, "current_fitness.png"))
        #     plt.close()
