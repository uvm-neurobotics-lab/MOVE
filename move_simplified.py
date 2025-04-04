import json
import os
import random
import copy
import time
import torch
import logging
import math

import numpy as np
import pandas as pd

from move_config import MOVEConfig
from tqdm import tqdm

from move_map import MOVEMap
from record_keeping import Record
from cppn.visualize import visualize_network

import matplotlib.pyplot as plt

from norm import norm_tensor, read_norm_data # TODO REMOVE
import fitness.fitness_functions as ff # TODO REMOVE
from tqdm import trange

from cppn.util import *
from util import *

from move_map import MOVEMap
from run_setup import run_setup
from sgd_weights import sgd_weights, sgd_weights_imaml
from record_keeping import Record
from stopping import name_to_stop_condition_map
from cppn.cppn import CPPN


class MOVE():
    def __init__(self, config=MOVEConfig(), debug_output=False) -> None:
        self.config = config
        config.setup()
        self.debug_output = debug_output
        
        if config.genome_type is None:
            # TODO: ugly
            config.genome_type = CPPN
        self.GenomeType = config.genome_type

        self.stop_condition = name_to_stop_condition_map[self.config.stop_condition](self.config.stop_condition_value)

        self.fns = config.objective_functions
            
        self.init_dirs()
        self.init_inputs()
        
        self.target = self.config.target
        
        self.gen = 0
        self.current_batch = 0
        self.time_elapsed = 0
        self.population = []
        self.solution = None
        self.device = config.device
        self.run_number = 0
        self.diversity = 0
        self.total_offspring = 0
        # self.avg_nodes = 0
        # self.avg_enabled_connections = 0
        
        self.solution_fitness = -math.inf
        self.best_genome = None

        self.fitnesses = {}
        self.total_batches = self.stop_condition.n_batches(self)
        
        print("Expecting up to", self.total_batches, "batches")
        print("Stop condition:", self.stop_condition.__class__.__name__ if self.stop_condition is not None else "None")
        print("\tAt:", self.stop_condition.value)
        
        self.map = MOVEMap(self.config, self.fns)
        
        self.n_cells = self.map.n_cells
        self.n_fns = self.map.n_fns
        self.fns_per_cell = self.map.fns_per_cell
        self.use_avg_fit = self.config.get("use_avg_fit", False)
        
        self.record = Record(self.config, self.n_fns, self.n_cells, self.total_batches, self.config.low_mem)
        self.norm = read_norm_data(self.config.norm_df_path, self.config.target_path)
        
        
        if self.config.with_grad and self.config.sgd_steps > 0:
            self.init_sgd()
        else:
            torch.autograd.set_grad_enabled(False)
            
        
        
        print("Initialized MOVE on device:", self.config.device)

       
    def init_dirs(self):
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
        # save config to run dir
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(copy.deepcopy(self.config).to_json(), f, indent=4)
            
        
        # save target to output directory # TODO MOVE THIS
        target_path = os.path.join(self.run_dir, "target.png")
        plt.imsave(target_path, self.config.target[0].permute(1,2,0).cpu().numpy())
    

    def init_inputs(self):
        """ Initializes the inputs for the genome type.
       
        Returns:
            Tensor: The inputs for the genome type.
        """
        self.inputs = self.GenomeType.init_inputs(self.config)
        return self.inputs
    
    
    def init_sgd(self, batch_cell_ids=None):
        if self.config.sgd_steps == 0:
            return
        torch.autograd.set_grad_enabled(True)

        if batch_cell_ids is None:
            # all of them
            batch_cell_ids = torch.arange(self.n_cells, device=self.config.device)    
        exclude = set(config.NO_GRADIENT).intersection(self.fns)
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
    
    def evolve(self, run_number = 1, show_output=False, initial_population=False, resume=None):
        # start evolving, defaults to no initial population because the initial pop is generated during gen 0
        try:
            if resume is not None:
                self.current_batch = self.record.load_checkpoint(resume, self.checkpoints_dir, self.map, self.config)
                print(self.map.get_population())
            # super().evolve(run_number, show_output, initial_population)
            self.start_time = time.time()
            self.run_number = run_number
            self.show_output = show_output or self.debug_output
            if isinstance(initial_population,list):
                # use the provided population
                self.population = initial_population
            elif initial_population:
                # make a new population
                for i in range(self.config.initial_batch_size): 
                    self.population.append(self.GenomeType(self.config)) # generate new random individuals as parents
            
            if initial_population:
                self.activate_population(self.population)
                
                self.update_fitnesses_and_novelty()
                self.population = sorted(self.population, key=lambda x: x.fitness.item(), reverse=True) # sort by fitness
                self.solution = self.population[0].clone(self.config, cpu=True) 

            try:
                # Run algorithm
                if self.config.stop_condition is None:
                    pbar = trange(self.config.total_offspring, desc=f"Run {self.run_number}") # default progress
                else:
                    pbar = trange(self.config.stop_condition_value, desc=f"Run {self.run_number}, {self.config.stop_condition}")
            
                while True:
                    # self.batch_start()
                    self.run_one_batch()
                    self.batch_end()
                    b = self.get_best()
                    if b is not None:
                        pbar.set_postfix_str(f"bf: {self.agg_fitnesses[b.id]:.4f} (id:{b.id}) af:{np.mean(list(self.agg_fitnesses.values())):.4f} n:{self.avg_nodes:.2f} cx:{self.avg_enabled_connections:.2f} u:{self.n_unique} ")
                    else:
                        pbar.set_postfix_str(f"d:{self.diversity:.4f}")
                    
                    self.current_batch += 1
                    if self.config.stop_condition is None:
                        pbar.n = self.total_offspring # default progress
                    else:
                        pbar.n = self.stop_condition.curr
                    pbar.refresh()
                   
                    if self.stop_condition(self):
                        print(f"Stop condition: {self.stop_condition.__class__.__name__ if self.stop_condition is not None else 'None'} met")
                        break
                    
                
            except KeyboardInterrupt:
                self.on_end()
                return  
        except KeyboardInterrupt:
            self.on_end()
            return 
        
        self.on_end()
              
   
    def get_best(self):
        if len(self.population) == 0:
            print("No individuals in population")
            return None
        max_fitness_individual = max(self.population, key=lambda x: self.agg_fitnesses[x.id])
        return max_fitness_individual
        
    def new_child(self, parent, all_parents):
        if parent is None:
            child = self.GenomeType(self.config)
            # child = self.GenomeType(self.config.num_inputs, self.config.num_outputs, self.config.hidden_nodes_at_start, self.config.init_connection_probability)
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
            child.reset(self.config)
            # child.update_layers()
            # child.disable_invalid_connections(self.config)
            # TODO lineage
            return child
        else:
            # asexual reproduction, child is mutated clone of parent
            child = parent.clone(self.config, new_id=True)
            child.mutate(self.config)
            child.parents = (parent.id, parent.id)
            child.cell_lineage = parent.cell_lineage
            child.n_cells = parent.n_cells
            child.reset(self.config)
            # child.update_layers()
            # child.disable_invalid_connections(self.config)
            return child
  
    def correct_target_count(self, count):
        """If the number of children has changed between iterations, 
        adjust the target to match
        """
        if self.target.shape[0]>count :
            print(f"WARNING: target batch size is larger than population size, truncating target from {self.target.shape[0]} to {count}")
            print("Population size:", self.config.num_cells)
            print("Total offspring:", self.total_offspring)
            self.target = self.target[:count]
        elif self.target.shape[0]<count:
            print(f"WARNING: target batch size is smaller than population size, repeating target from {self.target.shape[0]} to {count}")
            print("Population size:", self.config.num_cells)
            print("Total offspring:", self.total_offspring)
            self.target = self.target.repeat(count//self.target.shape[0], *([1]*len(self.target.shape[1:])))
    
    def activate_population(self, genomes):
        if self.config.activation_mode == 'population':
            outputs = activate_population(genomes, self.config, self.inputs)
        else:
            if self.config.thread_count > 1:
                outputs = activate_population_async(genomes,
                                                 self.in_queue,
                                                 self.out_queue,
                                                 self.target,
                                                 self.config)
            else:
                outputs = torch.stack([g(self.inputs) for g in genomes])
            
            outputs = outputs.clamp_(0,1)
        if hasattr(self, "target"):
            outputs, self.target = ff.correct_dims(outputs, self.target)
        return outputs


    @torch.no_grad()
    def measure_fitness(self, genomes, imgs, skip_genotype=False):
        # Calculate the number of batches required
        num_batches = math.ceil(len(genomes) / self.config.batch_size)
        print("Measuring fitness in", num_batches, "batches")
        genomes_batched = [genomes[i:i+self.config.batch_size] for i in range(0, len(genomes), self.config.batch_size)]

        fit_children = torch.zeros((len(genomes), len(self.fns)), device=self.config.device, requires_grad=False)
        fc_normed = torch.zeros((len(genomes), len(self.fns)), device=self.config.device, requires_grad=False)

        if self.config.dry_run:
            # Random fitness for dry run
            fit_children = torch.rand((len(genomes), len(self.fns)), device=self.config.device, requires_grad=False)
            return fit_children, fit_children.clone()

        pbar = tqdm(total=num_batches, desc="Measuring fitness")
        for batch_index, batch_genomes in enumerate(genomes_batched):
            self.correct_target_count(len(batch_genomes))
            batch_start_idx = batch_index * self.config.batch_size
            if imgs is None:
                batch_imgs = self.activate_population([g for _,_,g in batch_genomes])
            else:
                batch_imgs = imgs[batch_start_idx:batch_start_idx+len(batch_genomes)]
            for i, fn in enumerate(self.fns):
                if fn in ff.GENOTYPE_FUNCTIONS:
                    if skip_genotype:
                        continue
                    # Evaluating genotype directly
                    fitness = fn([g[2] for g in genomes[batch_start_idx:batch_start_idx+len(batch_imgs)]])
                    normed_fitness = torch.tensor([-torch.inf]*len(batch_imgs), device=self.config.device, requires_grad=False)
                else:
                    # Evaluating based on images
                    fitness = fn(batch_imgs, self.target)
                    if fitness.dim() == 0:
                        fitness = fitness.unsqueeze(0)
                    if fn not in ff.NO_NORM:
                        normed_fitness = norm_tensor(fitness, self.norm, fn.__name__, warn=False)

                if not fn in ff.NO_MEAN:
                    fc_normed[batch_start_idx:batch_start_idx+len(batch_imgs), i] = normed_fitness

                fit_children[batch_start_idx:batch_start_idx+len(batch_imgs), i] = normed_fitness if self.use_avg_fit else fitness
            pbar.update(1)
            
        # Normalizing collected fitness components
        if len(fc_normed) > 0:
            fc_normed = fc_normed
        else:
            fc_normed = fit_children.clone()  # Not normalizing anything

        # Assigning final fitness to genomes
        for idx, (g, f) in enumerate(zip(genomes, fc_normed)):
            f = f[f != -torch.inf]
            g[2].fitness = f.mean().detach()
            
        agg_fc_normed   = fc_normed.detach().mean(dim=1)

        return fit_children, fc_normed, agg_fc_normed


    def selection(self):
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
           self.correct_target_count(len(batch_cell_ids))
        
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
        
                
        return new_children, batch_cell_ids, initial_pop_done

    def bloat_population(self, new_children):
        n_bloat = torch.zeros(len(new_children), device=self.config.device, dtype=torch.int32)
        if config.bloat_prune_ratio>0:
            for child_i,_,c in new_children:
                before = len(c.connections)
                this_bloat = int(len(c.connections) * config.bloat_prune_ratio)
                for _ in range(this_bloat):
                    c.add_connection(config)
                after = len(c.connections)
                n_bloat[child_i] = after - before
                # print(f"{c_i} Bloat {this_bloat} connections")
        return n_bloat
    

    def sgd_population(self, new_children, batch_cell_ids):
        steps=0
        if self.config.sgd_steps > 0 and self.config.with_grad and (self.current_batch+1) % self.config.grad_every == 0:
            # do SGD update
            self.init_sgd(batch_cell_ids)
            if self.config.thread_count > 1:
                raise NotImplementedError("Multiprocessing no longer implemented for MOVE")
            else:
                before, _, _ = self.measure_fitness(new_children, None)

                sgd_fn = sgd_weights 
                if self.config.sgd_strat == 'imaml':
                    sgd_fn = sgd_weights_imaml
                steps = sgd_fn(new_children, 
                            mask        = self.mask,
                            # mask        = None, # SGD on all functions
                            inputs      = self.inputs,
                            target      = self.target,
                            fns         = self.sgd_fns,
                            norm        = self.norm,
                            # norm        = None,
                            config      = self.config,
                            early_stop  = self.config.sgd_early_stop,
                            )
                after, _, _ = self.measure_fitness(new_children, None)
                print("SGD improvement:", (after.mean()-before.mean()).item())
        return steps
        
    def prune_population(self, new_children, n_bloat):
        
        if config.bloat_prune_ratio>0:
            for child_i, _, c in new_children:
                num = c.prune_fixed_connections(n_bloat[child_i])
        else:
            # populate the node activations or else they will be pruned for inactivity
            imgs = self.activate_population([g for _,_,g in new_children])
               
        n_pruned, n_pruned_nodes = 0,0
        for _,_,child in new_children:
            cx_pruned,nodes_pruned = child.prune(self.config)
            n_pruned += cx_pruned
            n_pruned_nodes += nodes_pruned
        return n_pruned, n_pruned_nodes


    def replace_by_voting(self, fit_child, normed_fit_child):
        """ 
        The meat of MOVE
        """
        # find out if we should replace the current elites (must be better than the elite on average)
        if self.map.using_soft_mask:
            self.map.fitness = torch.where(torch.isinf(self.map.fitness), torch.ones_like(self.map.fitness)*-1000, self.map.fitness) # TODO no
            
            if self.config.soft_replace:
                # use soft replacement
                replaces = self.soft_replacement(normed_fit_child)
            else:
                # simple weighted average, replace if better or same
                replaces = (fit_child * self.map.fn_mask).sum(dim=0) >= (self.map.fitness * self.map.fn_mask).sum(dim=0)
            votes = None
            improvement = None
        else:
            # voting
            # find the cell/function combinations where the child is better than the current elite
            improvement = fit_child > self.map.fitness 
        
            # filter by the functions that are in each cell:
            improvement = improvement * self.map.fn_mask 
        
            # tabulate votes by taking sum over masked functions
            votes = improvement.sum(dim=0) 
            replaces = votes > self.fns_per_cell/2 
        
        return votes, improvement, replaces
    

    def replacement(self, new_children, fit_children, fc_normed, batch_cell_ids, agg_fc_normed, initial_pop_done):
        all_replacements = torch.zeros((self.n_cells, self.n_cells), device=self.config.device)
        
        random.shuffle(new_children) # random order of children
        for _, c_tuple in enumerate(new_children):
            # repeat for comparison against current map
            child_i, cell_i, child = c_tuple
            fit_child = fit_children[child_i] # (fns)
            fc_child_normed = fc_normed[child_i] # (fns)
            
            fit_child = fit_child.repeat(self.n_cells, 1).T # (fns, cells)
            normed_fit_child = fc_child_normed.repeat(self.n_cells, 1).T # (fns, cells)
            
            # find where this child is better than the current elite
            replaces = None
            if self.use_avg_fit:
                _, _, replaces = self.replace_by_avg_fit(fit_child)
            else:
               _, _, replaces = self.replace_by_voting(fit_child, normed_fit_child)
            
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
            
            all_replacements[cell_i] = replaces # all_replacements is (cells, cells) where each row is the replacement for that cell
            
            self.map.fitness[:,replaces] = fit_child[:,replaces].detach() # update fitness values
            self.map.normed_fitness[:,replaces] = normed_fit_child[:,replaces].detach() # update fitness values
            
            idxs_to_replace = torch.nonzero(replaces).squeeze().tolist()
            if isinstance(idxs_to_replace, int):
                idxs_to_replace = [idxs_to_replace]
                
            if self.debug_output:
                logging.debug(f"Replacing cells: {idxs_to_replace} with {child.id}")
                
            for r in idxs_to_replace:
                placed = child.clone(self.config, new_id=False, cpu=False)
                placed.cell_lineage = child.cell_lineage + [r]
                placed.n_cells = child.n_cells
                self.map.map[r] = placed
                self.map.agg_fitness[r] = agg_fc_normed[child_i] 
            del child
            
            if not self.config.allow_jumps > 0:
                assert torch.sum(replaces) <= 1
                
        print("\n", torch.sum(all_replacements).item(), "replacements")
        
        return all_replacements
    
    
    def save_checkpoint(self):
        print("Saving checkpoint")
        self.record.save_checkpoint(self.run_dir, self.checkpoints_dir, self.map, self.config, self.current_batch,
                                    save_data=False)
        self.save_move_info()
        
    def selection_and_reproduction(self):
        
        # Choose parents
        new_children, batch_cell_ids, initial_pop_done = self.selection()
        
        # Bloat, SGD, prune        
        n_bloat                  = self.bloat_population(new_children)
        steps                    = self.sgd_population(new_children, batch_cell_ids)
        n_pruned, n_pruned_nodes = self.prune_population(new_children, n_bloat)
                    
        # Measure children
        fit_children, fc_normed, agg_fc_normed = self.measure_fitness(new_children, None)
        
        # Replace elites        
        all_replacements = self.replacement(new_children,
                                            fit_children,
                                            fc_normed,
                                            batch_cell_ids,
                                            agg_fc_normed,
                                            initial_pop_done)

        # Record keeping
        self.record_keep(new_children, steps, n_pruned, n_pruned_nodes, all_replacements)
            
       
    def replace_by_avg_fit(self, fit_child):
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
        if self.config.checkpoint_frequency > 0 and self.current_batch % self.config.checkpoint_frequency == 0:
            self.save_checkpoint()
    
    def save_move_info(self):
        with open(os.path.join(self.run_dir, "cell_names.csv"), "w") as f:
            f.write(",".join(self.map.cell_names))
        with open(os.path.join(self.run_dir, "function_names.csv"), "w") as f:
            f.write(",".join([fn.__name__ for fn in self.fns]))
      
            
        torch.save(self.map.fn_mask, os.path.join(self.run_dir, "fn_mask.pt"))
        
        
    
    def on_end(self):
        # super().on_end()
        self.end_time = time.time()     
        self.time_elapsed = self.end_time - self.start_time  
        print("\n\nEvolution completed with", self.gen, "generations,", self.current_batch, "batches, and", self.total_offspring, "offspring", "in", self.time_elapsed, "seconds")
        print("Wrapping up, please wait...")

        # save results
        print("Saving data...")
        self.run_number = self.config.run_id

             
        # save config file
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(copy.deepcopy(self.config).to_json(), f, indent=4)
        
        with open(os.path.join(self.run_dir, "total_offspring.txt"), "w") as f:
            f.write(str(self.total_offspring))
            
        torch.save(self.inputs, os.path.join(self.run_dir, "inputs.pt")) # save inputs
        
     
        with open(os.path.join(self.run_dir, f"target.txt"), 'w') as f:
            f.write(str(self.config.target_path))
        
        self.save_best_img(os.path.join(self.image_dir, f"best_{self.config.run_id:04d}.png"), do_graph=True)
        print("Saved run to: ", self.run_dir)
        
        if self.config.thread_count > 1:
            for w in self.workers:
                w.close()
                
        # save data
        self.record.save(self.run_dir, plot=True) # save statistics
           
        # save other data
        self.save_move_info()
        
        if not self.config.dry_run:
            self.record.save_map(self.image_dir, self.map, self.config, self.inputs)
        
        # save lineages
        logging.info("Saving lineages")
        lineages = {i:v for i,v in enumerate(self.get_lineages())}
        json.dump(lineages, open(os.path.join(self.run_dir, "lineages.json"), "w"), indent=4)
        
        self.save_checkpoint()
        
    
    def record_keep(self, new_children, steps, n_pruned, n_pruned_nodes, all_replacements):
        n_step_fwds = len(new_children)
        n_step_fwds_incl_sgd = n_step_fwds+(n_step_fwds * steps) if self.config.with_grad else n_step_evals
        n_step_evals = len(new_children) * len(self.fns)
        n_step_evals_incl_sgd = n_step_evals+(n_step_evals * steps) if self.config.with_grad else n_step_evals
        self.record.update_counts(self.current_batch, n_step_fwds, n_step_fwds_incl_sgd, n_step_evals, n_step_evals_incl_sgd, n_pruned, n_pruned_nodes)
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
    
    def save_best_img(self, fname, do_graph=False, show_target=False):
        # if not do_graph and not self.gen % 10 == 0:
        #     return
        b = self.get_best()
        if b is None:
            return
        # b.to(self.config.device)
        img = b(self.inputs, channel_first=False, act_mode="node")
        if len(self.config.color_mode)<3:
            img = img.repeat(1, 1, 3)
        
        img = torch.clamp(img,0,1).detach().cpu().numpy()
        
        # show as subplots
        if show_target:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(img, cmap='gray')
            ax2.imshow(self.target.squeeze(), cmap='gray')
            ax1.set_title("Champion")
            ax2.set_title("Target")
            plt.savefig(fname)

        else:
            try:
                plt.imsave(fname, img, cmap='gray')
            except:
                print("Failed to save image")
                return
        
        plt.close()
        
        # if self.gen % 10 == 0:
        #     do_graph = True # always do graph 
        
        if do_graph:
            c_b = b.clone(self.config, new_id=False)
            # c_b.forward(self.inputs)
            # c_b.vis(fname.replace(".png", "_torch_graph"))
            c_b.vis(self.inputs, fname.replace(".png", "_torch_graph"))
            
            visualize_network(b, self.config, save_name=fname.replace(".png", "_graph.png"))
            plt.close()
    
    
    def get_lineages(self):
        for g in self.map.map:
            if g is None:
                yield None
            else:
                yield g.cell_lineage       
        


    def soft_replacement(self, normed_fit_child):
        # random probability based on difference in fitness
        self.map.normed_fitness = torch.where(torch.isinf(self.map.fitness), torch.ones_like(self.map.normed_fitness)*-1000, self.map.fitness) # TODO FIXME
        diff  = normed_fit_child - self.map.normed_fitness
        diff  = diff * self.map.fn_mask
        diff  = diff.mean(dim=0)
        diff *= config.soft_replace_mod

        # random replacement
        return torch.rand(self.n_cells, device=self.config.device) < diff

if __name__ == "__main__":
    # Example usage
    config = MOVEConfig()
    config.target_path = "data/skull.png"
    config.sgd_steps = 20
    config.stop_condition_value = 100*config.sgd_steps*3*100
    alg = MOVE(config)
    if config.do_profile:
        import cProfile
        prof_path = os.path.join(alg.config.output_dir, f"{config.run_id:04d}.prof")
        cProfile.run("alg.evolve(resume=args.resume)", prof_path, sort="cumtime")
        import pstats

        file = open(os.path.join(alg.config.output_dir, f"{config.run_id:04d}.prof.txt"), 'w')
        profile = pstats.Stats(prof_path, stream=file)
        profile.sort_stats('cumulative') # Sorts the result according to the supplied criteria
        profile.print_stats(1000) # Prints the first 1000 lines of the sorted report
        file.close() 
    else:
        alg.evolve()    
        