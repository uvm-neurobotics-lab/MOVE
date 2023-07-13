import json
import os
from typing import Callable
import random
import copy
import torch
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision.transforms import Resize
from tqdm import tqdm

from cppn_torch import ImageCPPN
from cppn_torch.graph_util import activate_population
from cppn_torch.fourier_features import add_fourier_features
from evolution_torch import CPPNEvolutionaryAlgorithm


from move_map import MOVEMap
from run_setup import run_setup
from sgd_weights import sgd_weights 
from record_keeping import Record

from norm import norm_tensor, read_norm_data
import fitness.fitness_functions as ff


class MOVE(CPPNEvolutionaryAlgorithm):
    def __init__(self, config, debug_output=False) -> None:
        self.config = copy.deepcopy(config)
        if self.config.objective_functions is None:
            # default: use all
            self.fns = [
                        ff.psnr,
                        ff.mse,
                        ff.lpips,
                        ff.dists,
                        ff.ssim,
                        ff.style,
                        ff.haarpsi,
                        ff.vif,
                        ff.msssim,
                        ff.dss,
                        ff.gmsd,
                        ff.fsim,
                        ff.mdsi,
                        ff.vsi,
                        ]
        else:
            self.fns = self.config.objective_functions
        
        self.allow_multiple_placements = self.config.allow_jumps > 0
        self.config.fns = [fn.__name__ for fn in self.fns] # for serialization
        
        super().__init__(self.config, debug_output)
        
        self.map = MOVEMap(self.config)
        
        self.n_cells = self.map.n_cells
        self.n_fns = self.map.n_fns
        self.fns_per_cell = self.map.fns_per_cell
        res_h, res_w = self.config.res_h, self.config.res_w
        self.use_avg_fit = hasattr(self.config, "use_avg_fit") and self.config.use_avg_fit
        
        self.record = Record(self.config, self.n_fns, self.n_cells)
        self.norm = read_norm_data(self.config.norm_df_path, self.config.target_name)
        
        # initialize the inputs
        
        self.inputs = ImageCPPN.initialize_inputs(
                res_h//2**config.num_upsamples,
                res_w//2**config.num_upsamples,
                self.config.use_radial_distance,
                self.config.use_input_bias,
                self.config.num_inputs,
                self.device,
                coord_range=self.config.coord_range
                )
        if self.config.use_fourier_features:
            self.inputs = add_fourier_features(
                self.inputs,
                self.config.n_fourier_features,
                self.config.fourier_feature_scale,
                dims=2,
                include_original=True,
                mult_percent=self.config.get("fourier_mult_percent", 0.0),
                )
        self.config.num_inputs = self.inputs.shape[-1]
        
        # initialize the target
        
        # repeat the target for easy comparison
        self.target = torch.stack([self.target.squeeze() for _ in range(self.config.num_cells)])

        if len(self.config.color_mode) > 1:
            self.target = self.target.permute(0, 3, 1, 2) # move color channel to front
        else:
            self.target = self.target.unsqueeze(1).repeat(1,3,1,1) # add color channel
            
        if self.target.shape[-2] < 32 or self.target.shape[-1] < 32:
                self.target = Resize((32,32),antialias=True)(self.target)
                
        print("Initialized MOVE on device:", self.config.device)
    
    
    def run_one_generation(self):
        # reproduce
        self.selection_and_reproduction()
        self.population = self.map.get_population(include_empty=False)
    
    
    def update_fitnesses_and_novelty(self):
        return # handled in selection_and_reproduction
    
    
    def evolve(self, run_number = 1, show_output=False, initial_population=False):
        # start evolving, defaults to no initial population because the initial pop is generated during gen 0
        try:
            super().evolve(run_number, show_output, initial_population)
        except KeyboardInterrupt:
            pass 
        
        
    def new_child(self, parent, all_parents):
        if self.config.do_crossover:
            # sexual reproduction, choose another parent randomly
            all_parents = list(filter(lambda x: x is not None, all_parents))
            other_parent = np.random.choice(all_parents)
            child = parent.crossover(other_parent) # crossover
            child.mutate() # mutate
            # TODO lineage
            return child
        else:
            # asexual reproduction, child is mutated clone of parent
            if self.config.with_grad:
                parent.discard_grads()
            child = parent.clone(new_id=True)
            child.mutate()
            child.parents = (parent.id, parent.id)
            child.lineage = parent.lineage
            return child
    
      
    def measure_fitness(self, genomes):
        if self.config.activation_mode == 'population':
            imgs = activate_population(genomes, self.config, self.inputs)
        else:
            imgs = torch.stack([g.get_image(self.inputs) for _, g in genomes])
            imgs = imgs.clamp_(0,1)
        imgs, self.target = ff.correct_dims(imgs, self.target)
        
        fit_children = torch.zeros((len(genomes), len(self.fns)), device=self.config.device)
        fc_normed = []

        if self.config.dry_run:
            # random fitness
            fit_children = torch.rand((len(genomes), len(self.fns)), device=self.config.device)
        else:
            for i, fn in enumerate(self.fns):
                if fn in ff.GENOTYPE_FUNCTIONS:
                    # evaluating genotype directly
                    fitness = fn([g[1] for g in genomes]) # (children)
                    normed_fitness = torch.tensor([-torch.inf for _ in range(len(genomes))], device=self.config.device)
                else:
                    fitness = fn(imgs, self.target) # (children)
                    
                    # normalize
                    if not fn in ff.NO_NORM:
                        normed_fitness = norm_tensor(fitness, self.norm, fn.__name__, warn=False)
                    
                if not fn in ff.NO_MEAN:
                    fc_normed.append(normed_fitness)
               
                if self.use_avg_fit:
                    # use normed as fitness
                    fit_children[:,i] = normed_fitness.detach().clone()
                else:
                    # use raw as fitness
                    fit_children[:,i] = fitness.detach()
                    
      
        fc_normed = torch.stack(fc_normed, dim=1)
                
        # take the mean of fitness values as overall fitness
        for g, f in zip(genomes, fc_normed):
            f = f[f != -torch.inf] # no need to include -inf (un-normed fitness)
            g[1].fitness = f.mean().detach()
            
        return fit_children, fc_normed

        
    def selection_and_reproduction(self):
        parents = self.map.get_population(include_empty=True) # current elite map

        new_children = []

        assert len(parents) == self.map.n_cells

        # random order of parents 
        if self.config.use_steady_state and self.gen > 0:
            np.random.shuffle(parents)
            self.target = self.target[0:1] # only need one target for steady state
                        
        # reproduction
        for i, p in enumerate(parents):
            if p is None:
                # empty cell, create random
                child = self.genome_type(self.config)
                child.lineage = [-1]
                new_children.append((i, child))
            else:
                # create child
                child = self.new_child(p, parents)
                for _ in range(self.config.initial_mutations):
                    child.mutate()
                new_children.append((i, child))
            new_children[-1][1].to(self.config.device)
            if self.config.use_steady_state and self.gen > 0:
                # max one child per iteration after first
                break
                
        
        
        if not self.config.use_steady_state:
            assert len(new_children) == len(parents) 
        
        if self.config.with_grad and (self.gen+1) % self.config.grad_every == 0:
            # do SGD update
            exclude = set(ff.NO_GRADIENT).intersection(self.fns)
            sgd_fns = set(self.fns).difference(exclude)
            skip_fns = [self.fns.index(f) for f in exclude]
            mask = torch.stack([self.map.fn_mask[i] for i in range(len(self.map.fn_mask)) if i not in skip_fns])
            
            sgd_weights(new_children, 
                        mask        = mask,
                        # mask      = None, # can mask loss by Fm_mask, results in more exploitation within cells
                        inputs      = self.inputs,
                        target      = self.target,
                        fns         = sgd_fns,
                        norm        = self.norm,
                        config      = self.config,
                        early_stop  = self.config.sgd_early_stop,
                        )
            for child in new_children:
                child[1].prune()
                    
                    
        # measure children
        fit_children, fc_normed = self.measure_fitness(new_children)
        fc_normed = fc_normed.detach().mean(dim=1).to("cpu")
        
        # for record keeping
        all_replacements = []
        all_votes = []

        random.shuffle(new_children) # random order of children
            
        for _, c_tuple in enumerate(new_children):
            i, c = c_tuple
            fit_child = fit_children[i] # (len(self.fns))
            
            # repeat for comparison against current map
            fit_child = fit_child.repeat(self.n_cells, 1).T # (len(self.fns), num_cells)
            
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
            
            all_votes.append(D)
            
            # update the elites
            if not self.config.allow_jumps:
                # can only replace parent
                tmp = torch.zeros_like(replaces) # start all 0s
                tmp[i] = replaces[i] # keep parent cell the same
                replaces = tmp
                assert torch.sum(replaces) <= 1

            elif torch.sum(replaces) > self.config.allow_jumps:
                # only allow one jump
                indices = torch.nonzero(replaces) # find replaces
                superiority = (fit_child * self.map.fn_mask) - self.map.fitness # (fns, cells)
                superiority = superiority.T # (cells, fns)
                sorted_indices = sorted(indices, key=lambda x: torch.sum(superiority[x]>0), reverse=True) # sort by superiority
                indices = sorted_indices[:self.config.allow_jumps] # take the most dominated allow_jumps cells
                replaces = torch.zeros_like(replaces) # start all 0s
                for i in indices:
                    replaces[i] = True # set the chosen cells to 1
                
            all_replacements.append(replaces.cpu())
            
            self.map.fitness[:,replaces] = fit_child[:,replaces] # update fitness values
            
            idxs_to_replace = torch.nonzero(replaces).squeeze().tolist()
            if isinstance(idxs_to_replace, int):
                idxs_to_replace = [idxs_to_replace]
                
            if self.debug_output:
                logging.debug(f"Replacing cells: {idxs_to_replace} with {c.id}")
                
            c.cpu()
            c.outputs = None
            for r in idxs_to_replace:
                placed = c.clone(new_id=False, cpu=True)
                placed.lineage = c.lineage + [r]
                self.map.map[r] = placed
                self.map.agg_fitness[r] = fc_normed[i] 
            del c
            
            if not self.allow_multiple_placements:
                assert torch.sum(replaces) <= 1

        # Record keeping
        if self.config.use_steady_state and not self.gen % self.config.population_size == 0:
            pass # don't record steady state generations unless we've done whole population
        else:
            gen_index = self.gen
            if self.config.use_steady_state:
                gen_index = self.gen // self.config.population_size
            all_replacements = torch.vstack(all_replacements)   # (new children, replacements of old elites)
            if not self.use_avg_fit:
                all_votes = torch.stack(all_votes)                  # (new children, len(self.fns), better than old)
            
            self.record.update(gen_index, all_replacements, all_votes, self.map)
            
            if not self.allow_multiple_placements:
                    assert self.record.replacements_over_time[:,:,gen_index].sum() <= self.n_cells
       
       
    def select_by_avg_fit(self, fit_child):
        """For testing without voting, use the average fitness of the child"""
        divisors = self.map.fn_mask.sum(dim=0) # will all be equal to self.fns_per_cell
        # mask out the functions that are not in each cell:
        masked_f = fit_child * self.map.fn_mask # (len(self.fns), num_cells)
        # sum over functions that are in cell
        summed_f = masked_f.sum(dim=0) # (num_cells)
        # mean over functions that are in cell
        D = summed_f / divisors # (num_cells)
        # mask elite map (len(self.fns), num_cells) by the functions that are in each cell:
        masked_f = self.map.fitness * self.map.fn_mask # (len(self.fns), num_cells)
        summed_f = masked_f.sum(dim=0) # (num_cells)
        E = summed_f / divisors# (num_cells)
        # replace nan with -inf
        E = torch.where(torch.isnan(E), torch.full_like(E, -float('inf')), E)
        # find out if we should replace the current elites
        votes = D > E # (num_cells)
        replaces = votes
        return votes, D, replaces
   
   
    def generation_end(self):
        self.solution_fitness = -torch.inf # force to update
        self.record_keeping(skip_fitness=False)
        
        
    def on_end(self):
        super().on_end()
        
        # save fitness over time
        cond_dir = os.path.join(self.config.output_dir, "conditions", self.config.experiment_condition)
        os.makedirs(cond_dir, exist_ok=True)
        run_dir = os.path.join(cond_dir, f"run_{self.config.run_id:04d}")
        
        self.record.save(run_dir) # save statistics
        
        torch.save(self.inputs, os.path.join(run_dir, "inputs.pt")) # save inputs
        
        # save config file
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(copy.deepcopy(self.config).to_json(), f, indent=4)
        
        # save other data
        with open(os.path.join(run_dir, "cell_names.csv"), "w") as f:
            f.write(",".join(self.map.cell_names))
        with open(os.path.join(run_dir, "function_names.csv"), "w") as f:
            f.write(",".join([fn.__name__ for fn in self.fns]))
            
        torch.save(self.map.fn_mask, os.path.join(run_dir, "Fm_mask.pt"))
       
        self.save_map()
        
        lineages = {i:v for i,v in enumerate(self.get_lineages())}
        json.dump(lineages, open(os.path.join(run_dir, "lineages.json"), "w"), indent=4)
        
        print("Saved run to: ", run_dir)
    
    
    def get_lineages(self):
        for g in self.map.map:
            if g is None:
                yield None
            else:
                yield g.lineage       
        
        
    def save_map(self):
        if self.config.dry_run:
            print("dry run, not saving map")
            return
        # save all images
        dirpath = os.path.join(self.config.output_dir, "images", "final_map", self.config.experiment_condition, f"run_{self.config.run_id:04d}")
        os.makedirs(dirpath, exist_ok=True)
        genomes_path = os.path.join(self.config.output_dir, "genomes", self.config.experiment_condition, f"run_{self.config.run_id:04d}")
        os.makedirs(genomes_path, exist_ok=True)
        genomes = []
        
        flat_map = self.map.get_population()
        print(f"Saving map with {len(flat_map)} genomes")
        pbar = tqdm(total=len(flat_map), desc="Saving final map...")
        
        for i in range(len(flat_map)):
            cell_fns_inds = self.map.cell_fn_inds[i] # flat
            cell_fns = [self.fns[i] for i in cell_fns_inds]
            if(flat_map[i] is not None):
                individual = flat_map[i]
                individual.to(self.config.device)
                img = individual.get_image(self.inputs, channel_first=False).detach().cpu().numpy()
                name = "_".join([(fn.__name__ if isinstance(fn, Callable) else fn) for fn in cell_fns])+f"_{individual.fitness.item():.3f}+{len(list(individual.enabled_connections()))}c"
                name = name + ".png"
                plt.imsave(os.path.join(dirpath, name), img, cmap='gray')
                plt.close()
                if self.config.with_grad:
                    flat_map[i].discard_grads()
                genomes.append(flat_map[i].clone(new_id=False).to_json())
            else:
                genomes.append("null")
            pbar.update(1)
            
        pbar.close()
        with open(os.path.join(genomes_path, "map.json"), "w") as f:
            json.dump(genomes, f)
            
        lineages = self.get_lineages()
        with open(os.path.join(genomes_path, "lineages.json"), "w") as f:
            lineages_dict = {i: l for i, l in enumerate(lineages)}
            json.dump(lineages_dict, f)


    def save_best_img(self, fname):
        b = self.get_best()
        if b is None:
            return
        b.to(self.config.device)
        img = b.get_image(self.inputs, channel_first=True)
        if len(self.config.color_mode)<3:
            img = img.unsqueeze(-1).repeat(1,1,3)
        else:
            img = img.permute(1,2,0) # (H,W,C)
        img = img.detach().cpu().numpy()
        plt.imsave(fname, img, cmap='gray')
        plt.close()

if __name__ == '__main__':
    for config, verbose in run_setup():
        alg = MOVE(config, debug_output=verbose)
        alg.evolve()