import copy
import os
import pandas as pd
import torch
from torchvision.transforms import Resize

from cppn import CPPN as ImageCPPN
from cppn.cppn import random_choice
from evolution import CPPNEvolutionaryAlgorithm
from move_config import MoveConfig
import logging

import fitness.fitness_functions as ff
from run_setup import run_setup
from sgd_weights import sgd_weights
from cppn.fourier_features import add_fourier_features
from norm import norm_tensor, read_norm_data
from util import *
from cppn.util import *
from record_keeping import Record

class BaselineConfig(MoveConfig):
    def __init__(self):
        super().__init__()
        self.fitness_function = "aggregate"
        self.name = 'default_baseline'
        self.alg = "baseline"
        self.fitness_function = "aggregate"
        # self.fitness_function = ff.fsim

fns = [
        ff.psnr,
        ff.mse,
        ff.lpips,
        ff.dists,
        ff.ssim,
        # ff.style,
        ff.haarpsi,
        ff.vif,
        ff.msssim,
        ff.dss,
        ff.gmsd,
        ff.fsim,
        ff.mdsi,
        ff.vsi,
        ]


fit_df = pd.DataFrame(columns=["fn", "max_fit", "max_fit_normed", "gen"])

CURRENT_GEN = 0

def aggregate_fitness(genomes, target, norm_df, inputs, skip_genotype=False):
    global fit_df, fns
    use_fns = fns
    if isinstance(genomes, torch.Tensor):
        imgs = genomes
    else:
        imgs = torch.stack([g(inputs) for g in genomes], dim=0)
        imgs, target = ff.correct_dims(imgs, target)
    
    if skip_genotype:
        use_fns = [fn for fn in fns if fn not in ff.GENOTYPE_FUNCTIONS]
    normed_fits = torch.zeros((len(imgs), len(use_fns)))
    fits = torch.zeros((len(imgs), len(use_fns)))
    for i, fn in enumerate(use_fns):
        if fn in ff.GENOTYPE_FUNCTIONS:
            fitness = fn(genomes)
        else:
            fitness = fn(imgs, target) # (children)
        fits[:, i] = fitness
        if fn not in ff.NO_NORM:
            normed = norm_tensor(fitness, norm_df, fn.__name__)
            normed_fits[:, i] = normed
        else:
            normed_fits[:, i] = fitness

    if len(fit_df)==0 or CURRENT_GEN > fit_df["gen"].max():
        elite = normed_fits.mean(dim=1).argmax().item()
        this_elite_mean = normed_fits.mean(dim=1)[elite].item()
        do_replace = True
        if len(fit_df)>0:
            prev_elite_mean = fit_df[fit_df["gen"]==fit_df["gen"].max()]["max_fit_normed"].mean()
            do_replace = this_elite_mean > prev_elite_mean
            
        for i, fn in enumerate(use_fns):
            if do_replace:
                max_fit = fits[elite, i].item()
                max_fit_normed = normed_fits[elite, i].item()
            
            else:
                # same as last gen
                last_gen = fit_df[(fit_df["gen"]==CURRENT_GEN-1) & (fit_df["fn"]==fn.__name__)]
                max_fit = last_gen["max_fit"].values[0]
                max_fit_normed = last_gen["max_fit_normed"].values[0]

            fit_df = pd.concat(
                [fit_df,
                pd.DataFrame({
                    "fn": fn.__name__,
                    "max_fit": max_fit,
                    "max_fit_normed": max_fit_normed,
                    "gen": CURRENT_GEN,
                    "cell": 'None'
                    }, index = [len(fit_df)]),
                    
                ],
            ignore_index=True).reset_index(drop=True)
            
            
    return normed_fits.mean(dim=1)


class Baseline(CPPNEvolutionaryAlgorithm):
    def __init__(self, config, debug_output=False) -> None:
        self.config = copy.deepcopy(config)
        
        if self.config.with_grad or self.config.fitness_function == "aggregate":
            # needs normalization data
            self.norm_df = read_norm_data(self.config.norm_df_path, self.config.target_name)
        
        self.name_function_map = ff.__dict__ # override default (cppn_torch.fitness_functions)
        if not self.config.fitness_function == "aggregate":
            self.sgd_fitness_function = self.name_function_map[
                self.config.fitness_function.__name__ if callable(self.config.fitness_function) else self.config.fitness_function
                ]
        
        
        res_h, res_w = self.config.res_h, self.config.res_w
        self.init_inputs()
        
        self.is_aggregate = False
        self.skip_norm = False
        if self.config.fitness_function == "aggregate":
            self.is_aggregate = True
            self.config.fitness_function = lambda x,y: aggregate_fitness(x, y, self.norm_df, self.inputs, False)
            self.config.fitness_function.__name__ = "aggregate"
            ff.register_fitness_function("aggregate", self.config.fitness_function)
            self.skip_norm = True
            self.sgd_fitness_function = lambda x,y: aggregate_fitness(x, y, self.norm_df, self.inputs, True)
            self.n_fns = len(fns)
        
        super().__init__(self.config, debug_output)
        
        self.fit_measures_genomes = False
        if self.config.fitness_function == "aggregate":
            if len(set(ff.GENOTYPE_FUNCTIONS).union(fns)) > 0:
                # at least one genotype function in fns
                self.fit_measures_genomes = True

        if self.config.autoencoder_frequency >0:
            self.config.autoencoder_frequency = 0 # does not use autoencoder
            logging.warning("algorithm does not use autoencoder, setting autoencoder_frequency to 0")
            
        self.lowest_fitness = torch.inf
        
         # repeat the target for easy comparison
        self.init_target()
        
        self.fitness_function = self.config.fitness_function
        
        assert config.batch_size == config.initial_batch_size, "Baseline is generational, batch_size must equal initial_batch_size"
        
        self.record = Record(self.config, 1, 1, self.total_batches, self.config.low_mem)
        
        self.total_offspring = self.config.initial_batch_size # started with initial batch size
        
        self.fitness = torch.ones((1, 1), device=self.config.device)*-torch.inf # fns in each cell
        self.normed_fitness = torch.ones((1, 1), device=self.config.device)*-torch.inf # fns in each cell
        self.agg_fitness = torch.zeros((1), device="cpu") # fitness of each cell (normed)
        
    def batch_end(self):
        """Called at the end of each generation"""
        self.solution_fitness = -torch.inf # force to update
        global CURRENT_GEN
        CURRENT_GEN += 1 
        
        self.record.batch_end(self, skip_fitness=False)
        
        self.avg_nodes = sum([len(g.nodes) for g in self.population]) / len(self.population)
        self.avg_enabled_connections = sum([len(g.enabled_connections) for g in self.population]) / len(self.population)
        
        self.gen = self.total_offspring // self.config.batch_size
        
        if self.current_batch in [0, ] or (self.current_batch+1)%10 == 0:
            b = self.get_best()
            if b is not None:
                b.save(os.path.join(self.genomes_dir, f"batch_{self.current_batch:04d}.json"), self.config)
        
        
    def run_one_batch(self):
        # always generational:
        self.run_one_generation()
    
    
    def run_one_generation(self):
        assert self.fitness_function is not None
            
        min_fit = self.population[-1].fitness.item()
        if min_fit < self.lowest_fitness:
            self.lowest_fitness = min_fit
        
        # reproduce
        self.population = self.selection_and_reproduction()
        
        # sort
        self.population = sorted(self.population, key=lambda x: x.fitness.item(), reverse=True) # sort by fitness
        self.agg_fitnesses = {self.population[i].id:self.population[i].fitness.item() for i in range(len(self.population))}
        
        
        if alg.current_batch % self.config.record_frequency_batch != 0:
            pass # don't record
        else:
            index = self.current_batch // self.config.record_frequency_batch
            self.record.update(index,
                                None,
                                self.fitness,
                                self.normed_fitness,
                                self.agg_fitness,
                                self.population[:1], # elite
                                self.total_offspring
                                )
                                
                                
                                
    def update_fitnesses_and_novelty(self, population=None):
        if population is None:
            population = self.population # assume whole population
            
        if self.config.dry_run:
            fits = torch.rand(len(population))
        else:
            if self.fit_measures_genomes:
                fits = self.fitness_function(population, self.target)
            else:
                if self.config.activation_mode == "population":
                    imgs = self.activate_population(population, self.config, self.inputs)
                else:
                    imgs = torch.stack([g(self.inputs) for g in population])
                    
                imgs, target = ff.correct_dims(imgs, self.target)
                fits = self.fitness_function(imgs, target)
                
        for i, g in enumerate(population):
            g.fitness = fits[i]


    def selection_and_reproduction(self):
        assert self.fitness_function is not None
        
        # reproduce
        new_children:list[ImageCPPN] = []
        for i in range(self.config.batch_size):
            if self.config.do_crossover:
                parent1 = self.population[0]
                parent2 = random_choice(self.population, 1, True)[0]
                child = parent1.crossover(parent2, self.config)
            else:
                parent = self.population[0] # simple hill climber
                child = parent.clone(self.config, new_id=True)

            child.mutate(self.config)
            new_children.append(child)
        
        self.total_offspring += len(new_children)
        
        if self.config.with_grad and (self.gen+1) % self.config.grad_every == 0:
            # do SGD update
            norm = None if self.skip_norm else self.norm_df
            # sgd_weights([(i,g) for g in new_children], 
            steps = sgd_weights(new_children, 
                    # mask   = self.map.Fm_mask.T,
                    mask    = None, # can mask loss by Fm_mask, results in more exploitation within cells
                    inputs  = self.inputs,
                    target  = self.target,
                    fns     = [self.sgd_fitness_function],
                    norm    = norm,
                    config  = self.config)
              
        # Pruning  
        n_pruned, n_pruned_nodes = 0,0
        for child in new_children:
            cx_pruned,nodes_pruned = child.prune(self.config)
            n_pruned += cx_pruned
            n_pruned_nodes += nodes_pruned
                    

        self.update_fitnesses_and_novelty(new_children)
        
        # replace parent
        for i, g in enumerate(new_children):
            if g.fitness > self.population[0].fitness:
                self.population[0] = g

        self.fitness = torch.tensor([self.population[0].fitness.item()], device=self.config.device).unsqueeze(0)
        self.normed_fitness = torch.tensor([self.population[0].fitness.item()], device=self.config.device).unsqueeze(0)
        self.agg_fitness = torch.tensor([self.population[0].fitness.item()], device=self.config.device).unsqueeze(0)
        
        n_step_fwds = len(new_children)
        n_step_fwds_incl_sgd = n_step_fwds+(n_step_fwds * steps) if self.config.with_grad else n_step_evals
        n_step_evals = len(new_children) * (1 if not self.is_aggregate else self.n_fns)
        n_step_evals_incl_sgd = n_step_evals+(n_step_evals * steps) if self.config.with_grad else n_step_evals
        self.record.update_counts(self.current_batch, n_step_fwds, n_step_fwds_incl_sgd, n_step_evals, n_step_evals_incl_sgd, n_pruned, n_pruned_nodes)


        return self.population
    
    def on_end(self):
        super().on_end()
                
        # save fitness over time
        self.record.save(self.run_dir) # save statistics
        
        sorted_pop = sorted(self.population, key=lambda x: x.fitness.item(), reverse=True) # sort by fitness
        highest_fitness = sorted_pop[0].fitness.item()
        
        if self.is_aggregate:
            global fit_df
            fit_df["target"] = self.config.target_name
            fit_df.to_pickle(os.path.join(self.run_dir, f"fits.pkl"))
        
        # record ranges
        filename_max = os.path.join(self.config.output_dir, "fits", self.config.experiment_condition, f"max_fit.txt")
        filename_min = os.path.join(self.config.output_dir, "fits", self.config.experiment_condition, f"min_fit.txt")
        
        if not os.path.exists(os.path.dirname(filename_max)):
            os.makedirs(os.path.dirname(filename_max))
        old_max, old_min = -torch.inf, torch.inf
        if os.path.exists(filename_max):
            with open(filename_max, "r") as f:
                old_max = float(f.read())
        if os.path.exists(filename_min):
            with open(filename_min, "r") as f:
                old_min = float(f.read())
        with open(filename_max, "w") as f:
            if highest_fitness > old_max:
                old_max = highest_fitness
            f.write(str(old_max))
        with open(filename_min, "w") as f:
            if self.lowest_fitness < old_min:
                old_min = self.lowest_fitness
            f.write(str(old_min))
    
    


if __name__ == '__main__':
    for config, verbose in run_setup(BaselineConfig):
        alg = Baseline(config, debug_output=verbose)
        alg.evolve(initial_population=True)