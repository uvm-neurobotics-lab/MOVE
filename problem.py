import numpy as np
import imageio as iio
import sys
import torch
import random
import copy
from  pymoo.core.sampling import Sampling
from pymoo.core.problem import Problem
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair, NoRepair
from pymoo.core.crossover import Crossover
from pymoo.core.mating import Mating
from pymoo.core.operator import Operator
from pymoo.core.callback import Callback



from pymoo.core.variable import Real, get
from pymoo.core.population import Population

import fitness.fitness_functions as ff
import cppn.activation_functions as af
from   sgd_weights import sgd_weights
from   cppn.cppn import CPPN
from   cppn.util import *
from   cppn.util import center_crop, resize
from   cppn.fourier_features import add_fourier_features
from   norm import norm_tensor, read_norm_data
from   move_config import MOVEConfig

skip_genotype=True

#%%


class CPPNProblem(Problem):

    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        self.fns = [
                ff.mse,
                ff.psnr,
                ff.lpips,
                ff.dists,
                ff.style,
                ff.vif,
                ff.dss,
                ff.ssim,
                ff.msssim,
                ff.haarpsi,
            ] # TODO: from config
        
        
        super().__init__(
                         n_obj=len(self.fns),
                       )
        
        res_h, res_w = self.config.res_h, self.config.res_w
        self.inputs = initialize_inputs(
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
                sin_and_cos=self.config.fourier_sin_and_cos
                )
        self.config.num_inputs = self.inputs.shape[-1]
        
        # self.target = torch.tensor(iio.imread(self.target), dtype=torch.float32, device=device)
        self.target = config.target
        tar = self.target.cpu().numpy()

        target_resize = config.target_resize
        res_fact = tar.shape[0] / target_resize[0], tar.shape[1] / target_resize[1]
        tar = resize(tar, (tar.shape[0] // int(res_fact[0]), tar.shape[1] // int(res_fact[1])))
        tar = center_crop(tar, target_resize[0], target_resize[1])
        self.target = torch.tensor(tar, dtype=torch.float32, device=self.device)
        self.target = self.target.permute(2,0,1) / 255.0
        self.norm = read_norm_data(self.config.norm_df_path, self.config.target_path)




    def _evaluate(self, x, out, return_imgs=False, *args, **kwargs):
        imgs = torch.stack([g.get_image(self.inputs) for g in x])
            
        imgs = imgs.clamp_(0,1)
        imgs, self.target = ff.correct_dims(imgs, self.target)
        fit_children = torch.zeros((len(imgs), len(self.fns)), device=self.config.device)
        fc_normed = []
        
        if self.target.shape[0]!=imgs.shape[0]:
            # resize target to match imgs
            self.target = self.target[0].unsqueeze(0).repeat(imgs.shape[0], 1, 1, 1)
            
        
        if self.config.dry_run:
            # random fitness
            fit_children = torch.rand((len(imgs), len(self.fns)), device=self.config.device)
        else:
            for i, fn in enumerate(self.fns):
                if fn in ff.GENOTYPE_FUNCTIONS:
                    if skip_genotype:
                        continue
                    # evaluating genotype directly
                    fitness = fn([g for g in x]) # (children)
                    normed_fitness = torch.tensor([-torch.inf for _ in range(len(imgs))], device=self.config.device)
                else:
                    fitness = fn(imgs, self.target) # (children)
                    
                    # normalize
                    if not fn in ff.NO_NORM:
                        normed_fitness = norm_tensor(fitness, self.norm, fn.__name__, warn=False)
                    
                if not fn in ff.NO_MEAN:
                    fc_normed.append(normed_fitness)

                fit_children[:,i] = fitness.detach()
        
        
        # for g in x:
            # serialize
            # g.clear_data()
            # g.generator = None # TODO: reproducibility
        
        out["F"] = 1.0-fit_children.cpu().numpy()
        out["n_F"] = 1.0-fit_children.cpu().numpy()
        out["imgs"] = imgs.detach().permute(0, 2, 3, 1).cpu().numpy()
        if not return_imgs:
            del imgs


class DoNothing(Operator):
    def do(self, problem, pop, **kwargs):
        X = np.array([ind.X for ind in pop])
        if self.vtype is not None:
            X = X.astype(self.vtype)

        Xp = self._do(problem, X, **kwargs)

        pop.set("X", Xp)
        return pop

    def _do(self, problem, X, **kwargs):
        return X

nothing = DoNothing()
    
class CPPNSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        val = []
        print("sampling", n_samples)
        for i in range(n_samples):
            x = CPPN(problem.config)
            val.append(x)
            # x.discard_grads()
        return val


class CPPNMutation(Mutation):
    def do(self, problem, pop, inplace=True, **kwargs):
        # if not inplace copy the population first
        if not inplace:
            pop = copy.deepcopy(pop)

        n_mut = len(pop)

        # get the variables to be mutated
        X = pop.get("X")

        # retrieve the mutation variables
        Xp = self._do(problem, X, **kwargs)
        pop.set("X", Xp)
        return pop
        # the likelihood for a mutation on the individuals
        prob = get(self.prob, size=n_mut)
        mut = np.random.random(size=n_mut) <= prob

        # store the mutated individual back to the population
        pop[mut].set("X", Xp[mut])

        return pop
    
    def _do(self, problem, X, **kwargs):
        # for x in X:
            # x.discard_grads()
        # Xp = copy.deepcopy(X)
        Xp = [x.clone(problem.config) for x in X]
        for i in range(len(Xp)):
            Xp[i].mutate(problem.config)
            # Xp[i].generator = None # TODO: reproducibility
            # Xp[i].id = CPPN.get_id()
            
        steps = sgd_weights([(None,None,Xp[i]) for i in range(len(Xp))], 
            mask        = None,
            inputs      = problem.inputs,
            target      = problem.target,
            fns         = problem.fns,
            norm        = problem.norm,
            config      = problem.config,
            early_stop  = problem.config.sgd_early_stop,
            )
        
        return Xp
    
    
class CPPNCrossover(Crossover):
    def do(self, problem, pop, **kwargs):
        X = []
        parents = pop.get("X")
        for pair in parents:
            child = pair[0].crossover(pair[1], problem.config)
            child.generator = None # TODO: reproducibility
            child.id = CPPN.get_id()
            X.append(child)
        X = np.array(X).reshape(-1)
        pop = Population.new("X", X)
        return pop

class NoCrossover(Crossover):
    def do(self, problem, pop, **kwargs):
        
        # flatten the population
        if len(pop.shape) != 1:
            X = []
            for pair in pop:
                for ind in pair:
                    X.append(ind.X)
            X = np.array(X).reshape(-1)
            pop = Population.new("X", X)
        return nothing.do(problem, pop, **kwargs)

    def _do(self, problem, X, **kwargs):
        return nothing._do(problem, X, **kwargs)

from pymoo.core.selection import Selection
        
class NoSelection(Selection):
    def _do(self, problem, pop, n_select, n_parents=None, **kwargs):
        return np.arange(2)
    
class CPPNRepair(Repair):
    def do(self, problem, pop, **kwargs):
        return nothing.do(problem, pop, **kwargs)

    def _do(self, problem, X, **kwargs):
        return nothing._do(problem, X, **kwargs)
       
