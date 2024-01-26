import json
import os
from typing import Callable
import random
import copy
import torch
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

# from cppn_torch import ImageCPPN

# from cppn_torch.graph_util import activate_population
# from cppn.util import visualize_network, initialize_inputs
from cppn.util import *
# from evolution_torch import CPPNEvolutionaryAlgorithm
from evolution import CPPNEvolutionaryAlgorithm
import torch.multiprocessing as mp

from util import *

from move_map import MOVEMap
from run_setup import run_setup
from sgd_weights import sgd_weights 
from record_keeping import Record
# from move_multiprocessing import MOVEWorker, activate_population_async, sgd_population_async

from norm import norm_tensor, read_norm_data
import fitness.fitness_functions as ff


class MOVE(CPPNEvolutionaryAlgorithm):
    def __init__(self, config, debug_output=False) -> None:
        self.config = copy.deepcopy(config)
        
        
        if self.config.objective_functions is None:
            # default: use all from paper
            # self.fns = ff.all_available_fns
            # self.fns = [
            #     # "clipiqa",
            #     # "clipiqa",
            #     # "clipiqa",
            #     "topiq_fr",
            #     "topiq_fr",
            #     "mse",
            # ]
            # self.fns = ff.initialize_fns(self.fns)
            # print([f.__name__ for f in self.fns])
            self.fns = [
                        ff.psnr,
                        ff.mse,
                        ff.lpips,
                        ff.dists,
                        ff.ssim,
                        #ff.style,
                        ff.haarpsi,
                        ff.vif,
                        ff.msssim,
                        ff.dss,
                        ff.gmsd,
                        ff.fsim,
                        ff.mdsi,
                        ff.vsi,
                        ]
            self.config.objective_functions = self.fns
        else:
            self.fns = self.config.objective_functions
            for i, fn in enumerate(self.fns):
                if isinstance(fn, str):
                    self.fns[i] = getattr(ff, fn)
                
        
        self.allow_multiple_placements = self.config.allow_jumps > 0
        
        super().__init__(self.config, debug_output)
        
        self.map = MOVEMap(self.config, self.fns)
        
        self.n_cells = self.map.n_cells
        self.n_fns = self.map.n_fns
        self.fns_per_cell = self.map.fns_per_cell
        self.use_avg_fit = self.config.get("use_avg_fit", False)
        
        self.record = Record(self.config, self.n_fns, self.n_cells, self.total_batches, self.config.low_mem)
        self.norm = read_norm_data(self.config.norm_df_path, self.config.target_name)
        
        self.init_inputs()
        
        self.init_target()        
        
        if self.config.with_grad:
            self.init_sgd()
        
        if self.config.thread_count > 1:
            self.in_queue = mp.Queue()
            self.out_queue = mp.Queue()
            self.workers = [MOVEWorker(self.config, self.in_queue, self.out_queue, i, self.inputs, self.mask, self.sgd_fns, self.target, self.norm) for i in range(self.config.thread_count)]
        
        
        
        print("Initialized MOVE on device:", self.config.device)
    

        
                            
    def init_sgd(self, batch_cell_ids=None):
        if self.config.sgd_steps == 0:
            return
        if batch_cell_ids is None:
            # all of them
            batch_cell_ids = torch.arange(self.n_cells, device=self.config.device)    
        exclude = set(ff.NO_GRADIENT).intersection(self.fns)
        self.sgd_fns = set(self.fns).difference(exclude)
        skip_fns = [self.fns.index(f) for f in exclude]
        self.mask = torch.stack([torch.index_select(self.map.fn_mask[i], 0, batch_cell_ids) 
                            for i in range(len(self.map.fn_mask)) if i not in skip_fns])
    
    def run_one_batch(self):
        # reproduce
        self.selection_and_reproduction()
        self.population = self.map.get_population(include_empty=False)
        self.agg_fitnesses = self.map.get_agg_fitnesses()
        self.fitnesses = self.map.get_fitnesses()
    
    def evolve(self, run_number = 1, show_output=False, initial_population=False):
        # start evolving, defaults to no initial population because the initial pop is generated during gen 0
        try:
            super().evolve(run_number, show_output, initial_population)
        except KeyboardInterrupt:
            pass # allow user to stop early
        
        
    def new_child(self, parent, all_parents):
        if parent is None:
            child = self.genome_type(self.config)
            # child = self.genome_type(self.config.num_inputs, self.config.num_outputs, self.config.hidden_nodes_at_start, self.config.init_connection_probability)
            child.cell_lineage = [-1]
            child.n_cells = 0
            return child
        if self.config.do_crossover:
            # sexual reproduction, choose another parent randomly
            all_parents = list(filter(lambda x: x is not None, all_parents))
            other_parent = np.random.choice(all_parents)
            child = parent.crossover(other_parent, self.config) # crossover
            child.mutate(self.config) # mutate
            child.n_cells = parent.n_cells
            # TODO lineage
            return child
        else:
            # asexual reproduction, child is mutated clone of parent
            # if self.config.with_grad:
                # parent.discard_grads()
            child = parent.clone(self.config, new_id=True)
            child.mutate(self.config)
            child.parents = (parent.id, parent.id)
            child.cell_lineage = parent.cell_lineage
            child.n_cells = parent.n_cells
            return child
  
    
    
    
    def measure_fitness(self, genomes, imgs, skip_genotype=False):
        fit_children = torch.zeros((len(imgs), len(self.fns)), device=self.config.device)
        fc_normed = []

        if self.config.dry_run:
            # random fitness
            fit_children = torch.rand((len(imgs), len(self.fns)), device=self.config.device)
        else:
            for i, fn in enumerate(self.fns):
                if fn in ff.GENOTYPE_FUNCTIONS:
                    if skip_genotype:
                        continue
                    # evaluating genotype directly
                    fitness = fn([g for _,_,g in genomes]) # (children)
                    normed_fitness = torch.tensor([-torch.inf for _ in range(len(imgs))], device=self.config.device)
                else:
                    fitness = fn(imgs, self.target) # (children)
                    if len(fitness.shape) == 0:
                        fitness = fitness.unsqueeze(0)
                        
                    # print(fn.__name__)
                    # print(fitness.min(), fitness.max())
                    # normalize
                    # normed_fitness = fitness # TODO no
                    # if len(normed_fitness.shape) == 0:
                        # normed_fitness = normed_fitness.unsqueeze(0)
                    
                    if not fn in ff.NO_NORM:
                        normed_fitness = norm_tensor(fitness, self.norm, fn.__name__, warn=False)
                        
                    if normed_fitness.min() < -1 or normed_fitness.max() > 1:
                        print("WARNING: normed fitness out of bounds")
                        print(fitness.min().item(), fitness.max().item())
                        print(normed_fitness.min().item(), normed_fitness.max().item())
                        print(fn.__name__)
                        print()
                # print(fn.__name__, fitness.shape)                    
                if not fn in ff.NO_MEAN:
                    fc_normed.append(normed_fitness)
               
                if self.use_avg_fit:
                    # use normed as fitness
                    fit_children[:,i] = normed_fitness.detach().clone()
                else:
                    # use raw as fitness
                    fit_children[:,i] = fitness.detach()
        if len(fc_normed) > 0:
            fc_normed = torch.stack(fc_normed, dim=1)
        else:
            fc_normed = fit_children.clone() # not normalizing anything
                
        # take the mean of fitness values as overall fitness (not used in selection)
        for g, f in zip(genomes, fc_normed):
            f = f[f != -torch.inf] # no need to include -inf (un-normed fitness)
            g[2].fitness = f.mean().detach()
        return fit_children, fc_normed


    def selection_and_reproduction(self):
        parents = self.map.get_population(include_empty=True) # current elite map

        new_children = []
        

        assert len(parents) == self.map.n_cells
        
        initial_pop_done = self.total_offspring >= self.config.num_cells
        batch_size = self.config.batch_size if initial_pop_done else self.config.initial_batch_size
        

        if initial_pop_done or not self.config.enforce_initial_fill:
            # random parents 
            batch_cell_ids = torch.tensor(np.random.choice(self.map.n_cells, size=batch_size, replace=False), device=self.config.device)
        else:
            # insure each cell is used once at first
            batch_cell_ids = torch.arange(start=self.total_offspring, end=min(self.map.n_cells, self.total_offspring+batch_size), device=self.config.device)

        if hasattr(self, "target"):
            if self.target.shape[0]>len(batch_cell_ids) :
                print(f"WARNING: target batch size is larger than population size, truncating target from {self.target.shape[0]} to {batch_size}")
                print("Population size:", self.config.num_cells)
                print("Total offspring:", self.total_offspring)
                self.target = self.target[:len(batch_cell_ids)]
        
        # reproduction
        for child_i, cell_i in enumerate(batch_cell_ids):
            p = parents[cell_i]
            child = self.new_child(p, parents)
            # child.to(self.config.device)
            for _ in range(self.config.initial_mutations):
                child.mutate()
                
            new_children.append((child_i, cell_i, child))
            # new_children[-1][2].to(self.config.device)
            
            self.total_offspring += 1
            
            if len(new_children) >= batch_size:
                break
        steps=0
        if self.config.sgd_steps > 0 and self.config.with_grad and (self.current_batch+1) % self.config.grad_every == 0:
            # do SGD update
            self.init_sgd(batch_cell_ids)
            if self.config.thread_count > 1:
                new_children = sgd_population_async(new_children,
                                                    self.in_queue,
                                                    self.out_queue,
                                                    self.target,
                                                    self.config,
                                                    batch_cell_ids)    
            else:
                steps = sgd_weights(new_children, 
                            mask        = self.mask,
                            # mask        = None,
                            inputs      = self.inputs,
                            target      = self.target,
                            fns         = self.sgd_fns,
                            norm        = self.norm,
                            # norm        = None,
                            config      = self.config,
                            early_stop  = self.config.sgd_early_stop,
                            )
        
        else:
            # we need to populate the node activations or else they will be pruned for inactivity
            imgs = self.activate_population([g for _,_,g in new_children])
               
        n_pruned, n_pruned_nodes = 0,0
        for _,_,child in new_children:
            cx_pruned,nodes_pruned = child.prune(self.config)
            n_pruned += cx_pruned
            n_pruned_nodes += nodes_pruned
                    
        # measure children
        imgs = self.activate_population([g for _,_,g in new_children])
        fit_children, fc_normed = self.measure_fitness(new_children, imgs)
        agg_fc_normed = fc_normed.detach().mean(dim=1)
        # agg_fc_normed = agg_fc_normed.detach().mean(dim=1).to("cpu")
        
        # for record keeping
        n_step_fwds = len(new_children)
        n_step_fwds_incl_sgd = n_step_fwds+(n_step_fwds * steps) if self.config.with_grad else n_step_evals
        n_step_evals = len(new_children) * len(self.fns)
        n_step_evals_incl_sgd = n_step_evals+(n_step_evals * steps) if self.config.with_grad else n_step_evals
        self.record.update_counts(self.current_batch, n_step_fwds, n_step_fwds_incl_sgd, n_step_evals, n_step_evals_incl_sgd, n_pruned, n_pruned_nodes)
        all_replacements = torch.zeros((self.n_cells, self.n_cells), device=self.config.device)
        # all_votes = torch.zeros((self.n_cells, self.n_cells), device=self.config.device)

        random.shuffle(new_children) # random order of children
            
        for _, c_tuple in enumerate(new_children):
            child_i, cell_i, child = c_tuple
            fit_child = fit_children[child_i] # (fns)
            
            # repeat for comparison against current map
            fit_child = fit_child.repeat(self.n_cells, 1).T # (fns, cells)
            normed_fit_child = fc_normed[child_i].repeat(self.n_cells, 1).T # (fns, cells)
            
            # find where this child is better than the current elite
            votes = None
            replaces = None
            D = None
            if self.use_avg_fit:
                votes, D, replaces = self.select_by_avg_fit(fit_child)
            else:
                """ 
                The meat of MOVE
                """
                D = fit_child > self.map.fitness 
            
                # filter by the functions that are in each cell:
                D = D * self.map.fn_mask 
            
                # tabulate votes by taking sum over masked functions
                votes = D.sum(dim=0) 
            
                # find out if we should replace the current elites
                replaces = votes > self.fns_per_cell/2 
                
                """
                End of meat
                """
            
            # all_votes[cell_i] = votes
            
            # update the elites
          
            if not initial_pop_done and self.config.enforce_initial_fill:
                # only replace the selected cells initially, to preserve the initial population diversity
                tmp = torch.zeros_like(replaces) # start all 0s
                use = replaces[torch.isin(torch.arange(self.n_cells, device=self.config.device), batch_cell_ids)]
                tmp[torch.isin(torch.arange(self.n_cells, device=self.config.device), batch_cell_ids)] = use
                replaces = tmp
          
            if not self.config.allow_jumps:
                # can only replace parent
                tmp = torch.zeros_like(replaces) # start all 0s
                tmp[child_i] = replaces[child_i] # keep parent cell the same
                replaces = tmp
                assert torch.sum(replaces) <= 1
            

            elif torch.sum(replaces) > self.config.allow_jumps:
                # only allow_jumps cells can be replaced, sorted by superiority
                indices = torch.nonzero(replaces) # find replaces
                superiority = (fit_child * self.map.fn_mask) - self.map.fitness # (fns, cells)
                superiority = superiority.T # (cells, fns)
                sorted_indices = sorted(indices, key=lambda x: torch.sum(superiority[x]>0), reverse=True) # sort by superiority
                indices = sorted_indices[:self.config.allow_jumps] # take the most dominated allow_jumps cells
                replaces = torch.zeros_like(replaces) # start all 0s
                for idx in indices:
                    replaces[idx] = True # set the chosen cells to 1
            
            child.n_cells = torch.sum(replaces).item()
            
            all_replacements[cell_i] = replaces
            
            self.map.fitness[:,replaces] = fit_child[:,replaces].detach() # update fitness values
            self.map.normed_fitness[:,replaces] = normed_fit_child[:,replaces].detach() # update fitness values
            
            idxs_to_replace = torch.nonzero(replaces).squeeze().tolist()
            if isinstance(idxs_to_replace, int):
                idxs_to_replace = [idxs_to_replace]
                
            if self.debug_output:
                logging.debug(f"Replacing cells: {idxs_to_replace} with {child.id}")
                
            # child.to('cpu')
            # child.clear_data() # save memory
            for r in idxs_to_replace:
                # placed = child.clone(new_id=False, cpu=True, deepcopy=False)
                # placed = child.clone(self.config, new_id=False, cpu=True)
                placed = child.clone(self.config, new_id=False, cpu=False)
                placed.cell_lineage = child.cell_lineage + [r]
                placed.n_cells = child.n_cells
                self.map.map[r] = placed
                self.map.agg_fitness[r] = agg_fc_normed[child_i] 
            del child
            
            if not self.allow_multiple_placements:
                assert torch.sum(replaces) <= 1

        # Record keeping
        
        # gens_per_population = self.config.num_cells // self.config.batch_size
        
        if self.current_batch % self.config.record_frequency_batch != 0:
            pass # don't record
        else:
            index = self.current_batch // self.config.record_frequency_batch
            self.record.update(index,
                                all_replacements,
                                self.map.fitness,
                                self.map.normed_fitness,
                                self.map.agg_fitness,
                                self.map.map,
                                self.total_offspring
                                )
            
       
    def select_by_avg_fit(self, fit_child):
        """For testing without voting, use the average fitness of the child"""
        divisors = self.map.fn_mask.sum(dim=0) # will all be equal to self.fns_per_cell
        # mask out the functions that are not in each cell:
        masked_f = fit_child * self.map.fn_mask 
        # sum over functions that are in cell
        summed_f = masked_f.sum(dim=0) 
        # mean over functions that are in cell
        D = summed_f / divisors 
        # mask elite map (len(self.fns), num_cells) by the functions that are in each cell:
        masked_f = self.map.fitness * self.map.fn_mask 
        summed_f = masked_f.sum(dim=0) 
        E = summed_f / divisors 
        # replace nan with -inf TODO
        E = torch.where(torch.isnan(E), torch.full_like(E, -float('inf')), E)
        # find out if we should replace the current elites
        votes = D > E 
        replaces = votes
        return votes, D, replaces
   
   
    def batch_end(self):
        self.solution_fitness = -torch.inf # force to update
        # self.record_keeping(skip_fitness=False)
        self.record.batch_end(self, skip_fitness=False)
        
        self.avg_nodes = sum([len(g.nodes) for g in self.population]) / len(self.population)
        self.avg_enabled_connections = sum([len(g.enabled_connections) for g in self.population]) / len(self.population)
        
        self.gen = self.total_offspring // self.config.num_cells
        
        if self.current_batch in [0, ] or (self.current_batch+1)%10 == 0:
            b = self.get_best()
            if b is not None:
                b.save(os.path.join(self.genomes_dir, f"batch_{self.current_batch:04d}.json"), self.config)
        
        
    def on_end(self):
        super().on_end()
        
        if self.config.thread_count > 1:
            for w in self.workers:
                w.close()
        # save fitness over time
        self.record.save(self.run_dir) # save statistics
           
        # save other data
        with open(os.path.join(self.run_dir, "cell_names.csv"), "w") as f:
            f.write(",".join(self.map.cell_names))
        with open(os.path.join(self.run_dir, "function_names.csv"), "w") as f:
            f.write(",".join([fn.__name__ for fn in self.fns]))
      
            
        torch.save(self.map.fn_mask, os.path.join(self.run_dir, "Fm_mask.pt"))
        
        if not self.config.dry_run:
            self.record.save_map(self.image_dir, self.map, self.config, self.inputs)
        
        logging.info("Saving lineages")
        lineages = {i:v for i,v in enumerate(self.get_lineages())}
        json.dump(lineages, open(os.path.join(self.run_dir, "lineages.json"), "w"), indent=4)
        
    
    
    def get_lineages(self):
        for g in self.map.map:
            if g is None:
                yield None
            else:
                yield g.cell_lineage       
        
       

if __name__ == '__main__':
    # python -m torch.utils.bottleneck /path/to/source/script.py [args]
    for config, verbose in run_setup():
        alg = MOVE(config, debug_output=verbose)
        if config.do_profile:
            import cProfile
            prof_path = os.path.join(alg.config.output_dir, f"{config.run_id:04d}.prof")
            cProfile.run("alg.evolve()", prof_path, sort="cumtime")
            import pstats

            file = open(os.path.join(alg.config.output_dir, f"{config.run_id:04d}.prof.txt"), 'w')
            profile = pstats.Stats(prof_path, stream=file)
            profile.sort_stats('cumulative') # Sorts the result according to the supplied criteria
            profile.print_stats(1000) # Prints the first 1000 lines of the sorted report
            file.close() 
        else:
            alg.evolve()
