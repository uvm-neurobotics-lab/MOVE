import copy
import os
import pandas as pd
import torch
from torchvision.transforms import Resize

from cppn_torch import ImageCPPN
from cppn_torch.cppn import random_choice
from cppn_torch.graph_util import activate_population
from evolution_torch import CPPNEvolutionaryAlgorithm
import logging

import fitness.fitness_functions as ff
from run_setup import run_setup
from sgd_weights import sgd_weights
from cppn_torch.fourier_features import add_fourier_features
from norm import norm_tensor, read_norm_data

fns = [
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
        
        self.inputs = ImageCPPN.initialize_inputs(
                res_h//2**config.num_upsamples,
                res_w//2**config.num_upsamples,
                self.config.use_radial_distance,
                self.config.use_input_bias,
                self.config.num_inputs,
                self.config.device,
                coord_range=self.config.coord_range
                )
        if self.config.use_fourier_features:
            self.inputs = add_fourier_features(
                self.inputs,
                self.config.n_fourier_features,
                self.config.fourier_feature_scale,
                dims=self.inputs.shape[-1],
                include_original=True,
                mult_percent=self.config.get("fourier_mult_percent", 0.0),
                sin_and_cos=self.config.fourier_sin_and_cos
                )
            self.config.num_inputs = self.inputs.shape[-1]
        
        self.is_aggregate = False
        self.skip_norm = False
        if self.config.fitness_function == "aggregate":
            self.is_aggregate = True
            self.config.fitness_function = lambda x,y: aggregate_fitness(x, y, self.norm_df, self.inputs, False)
            self.skip_norm = True
            self.sgd_fitness_function = lambda x,y: aggregate_fitness(x, y, self.norm_df, self.inputs, True)
        
        super().__init__(self.config, debug_output)
        
        self.fit_measures_genomes = False
        if self.config.fitness_function == "aggregate":
            if len(set(ff.GENOTYPE_FUNCTIONS).union(fns)) > 0:
                # at least one genotype function in fns
                self.fit_measures_genomes = True

        if self.config.autoencoder_frequency >0:
            self.config.autoencoder_frequency = 0 # does not use autoencoder
            logging.warning("algorithm does not use autoencoder, setting autoencoder_frequency to 0")
            
        os.makedirs(os.path.join(self.config.output_dir,"images"), exist_ok=True)
        self.lowest_fitness = torch.inf
        
         # repeat the target for easy comparison
        self.target = torch.stack([self.target.squeeze() for _ in range(self.config.num_cells)])

        # if len(self.config.color_mode) > 1:
        self.target = self.target.permute(0, 3, 1, 2) # move color channel to front
        
        self.target = self.target[:self.config.num_children]
        print(self.target.shape)
        
        # else:
            # print(self.target.shape)
            # self.target = self.target.permute()
            
        if self.target.shape[-2] < 32 or self.target.shape[-1] < 32:
                self.target = Resize((32,32),antialias=True)(self.target)
        
    def generation_end(self):
        """Called at the end of each generation"""
        self.solution_fitness = -torch.inf # force to update
        global CURRENT_GEN
        CURRENT_GEN += 1 
        super().record_keeping()
        
    def run_one_generation(self):
        assert self.fitness_function is not None
            
        # reproduce
        min_fit = self.population[-1].fitness.item()
        if min_fit < self.lowest_fitness:
            self.lowest_fitness = min_fit
        
        self.population = self.selection_and_reproduction()
        self.population = sorted(self.population, key=lambda x: x.fitness.item(), reverse=True) # sort by fitness
        
    def update_fitnesses_and_novelty(self):
        return # handled in selection_and_reproduction

    def selection_and_reproduction(self):
        assert self.fitness_function is not None
        
        # reproduce
        new_children = []
        for i in range(self.config.num_children):
            if self.config.do_crossover:
                parent1 = self.population[0] 
                parent2 = random_choice(self.population, 1, True)[0]
                child = parent1.crossover(parent2)
            else:
                parent = self.population[0]
                child = parent.clone(new_id=True)

            child.mutate()
            new_children.append(child)
        
        if self.config.with_grad and (self.gen+1) % self.config.grad_every == 0:
            # do SGD update
            norm = None if self.skip_norm else self.norm_df
            # sgd_weights([(i,g) for g in new_children], 
            sgd_weights(new_children, 
                    # mask   = self.map.Fm_mask.T,
                    mask    = None, # can mask loss by Fm_mask, results in more exploitation within cells
                    inputs  = self.inputs,
                    target  = self.target,
                    fns     = [self.sgd_fitness_function],
                    norm    = norm,
                    config  = self.config)
                

        if self.config.dry_run:
            fits = torch.rand(len(new_children))
        else:
            if self.fit_measures_genomes:
                fits = self.fitness_function(new_children, self.target)
            else:
                if self.config.activation_mode == "population":
                    imgs = activate_population(new_children, self.config, self.inputs)
                else:
                    imgs = torch.stack([g.get_image(inputs=self.inputs) for g in new_children])
                    
                imgs, target = ff.correct_dims(imgs, self.target)
                fits = self.fitness_function(imgs, target)
                
        for i, g in enumerate(new_children):
            g.fitness = fits[i]
            if g.fitness > self.population[0].fitness:
                self.population[0] = g

        return self.population
    
    def on_end(self):
        sorted_pop = sorted(self.population, key=lambda x: x.fitness.item(), reverse=True) # sort by fitness
        highest_fitness = sorted_pop[0].fitness.item()
        img_path = os.path.join(self.config.output_dir, "images", "top", self.config.experiment_condition)
        os.makedirs(img_path, exist_ok=True)
        self.save_best_img(os.path.join(img_path, f"run_{self.config.run_id}_{self.config.target_name.replace('/','-')}_{highest_fitness}.png"))
        
        if self.is_aggregate:
            cond_dir = os.path.join(self.config.output_dir, "conditions", self.config.experiment_condition)
            run_dir = os.path.join(cond_dir, f"run_{self.config.run_id}")
            os.makedirs(run_dir, exist_ok=True)
            # fit_df.to_csv(os.path.join(cond_dir, f"run_{self.config.run_id}_max_fits.csv"))
            global fit_df
            # fit_df = fit_df.reset_index()
            fit_df["target"] = self.config.target_name
            fit_df.to_pickle(os.path.join(run_dir, f"fits.pkl"))
        
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
        return super().on_end()
    

if __name__ == '__main__':
    for config, verbose in run_setup():
        alg = Baseline(config, debug_output=verbose)
        alg.evolve()