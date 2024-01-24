# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari


from pymoo.algorithms.moo.nsga3 import NSGA3


from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

#Importing required modules
import math
import random
import matplotlib.pyplot as plt

from mnk_landscapes import *

import numpy as np
from pymoo.core.problem import Problem
from mnk_landscapes import Rmnk 

class CustomRmnkProblem(Problem):
    def __init__(self, rmnk):
        super().__init__(n_var=rmnk.n,  # Number of decision variables
                         n_obj=rmnk.m,  # Number of objectives
                         n_constr=0,    # Number of constraints
                         xl=0,          # Lower bound of decision variables
                         xu=1)          # Upper bound of decision variables
        self.rmnk = rmnk

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], self.rmnk.m))

        for i in range(X.shape[0]):
            decision_vector = np.where(X[i] < 0.5, 0, 1)  # Convert to binary representation
            for j in range(self.rmnk.m):
                F[i, j] = self.rmnk.f(j, decision_vector)
        out["F"] = F





if __name__ == "__main__":
    import os
    from move import MOVE
    from run_setup import run_setup
    from tqdm import tqdm

    for config, verbose in run_setup(config_class=MNKConfig):
        
        # Initialize RMNK problem
        # rho = Objective correlation coefficient
        # m = Number of objective functions
        # n = Length of solutions
        # k = Number of epistatic links (non-linearity)


        print("Loading RMNK instance from", config.rmnk_instance_file)
        rmnk = Rmnk(instance_file=config.rmnk_instance_file)
        config.rmnk_r = rmnk.rho
        config.rmnk_m = rmnk.m
        config.rmnk_n = rmnk.n
        config.rmnk_k = rmnk.k
        
        print("RMNK:", config.rmnk_r, config.rmnk_m, config.rmnk_n, config.rmnk_k)


        config.mutation_sigma = 0.4
        config.mutation_percent = 0.05


        fitness_functions = []

        for i in range(config.rmnk_m):
            fn = lambda genomes, i=i, config=config, rmnk=rmnk: -1.0 * evaluate_genomes(genomes, rmnk, i, config) # invert sign to maximize
            fn.__name__ = f"Objective {i}"
            fitness_functions.append(fn)
            ff.register_fitness_function(fn.__name__, fn, True)
            ff.NO_NORM.append(fn)
            ff.NO_MEAN.append(fn)
            
        
        if not hasattr(config, 'output_dir'):
            config.output_dir = "../results/nsga-mnk"
        config.mutate_sgd_lr_sigma = False
        config.objective_functions = fitness_functions
        config.sgd_steps = 0
        config.genome_type = MNKGenome
        config.target=None

        config.experiment_condition = config.experiment_condition + " (NSGA-III)"
       
        cond_dir = os.path.join(config.output_dir, "conditions", config.experiment_condition)
        os.makedirs(cond_dir, exist_ok=True)
        run_dir = os.path.join(cond_dir, f"run_{config.run_id:04d}")
        os.makedirs(run_dir, exist_ok=True)
        
        
        # 50,000 total evaluations


        # create the reference directions to be used for the optimization
        ref_dirs = get_reference_directions("das-dennis", rmnk.m, n_partitions=12)

        # create the algorithm object
        algorithm = NSGA3(pop_size=100,
                        ref_dirs=ref_dirs)

        # execute the optimization
        res = minimize(CustomRmnkProblem(rmnk),
                    algorithm,
                    seed=0,
                    termination=('n_eval', 50_000), 
                    save_history=True,
                    verbose=True
                    )

        f = Scatter().add(res.F)
        plt.savefig(os.path.join(config.output_dir, "nsga3.png"))

        evals_by_batch = torch.tensor([s.evaluator.n_eval for s in res.history], dtype=torch.float32)
        normed_fitness_by_batch = torch.ones(len(fitness_functions), 1000, 10_000)*-torch.inf
        F = [s.opt.get("F") for s in res.history]

        for gen in range(len(F)):
            front = F[gen].T
            for obj_no, obj in enumerate(front):
                for i, val in enumerate(obj):
                    normed_fitness_by_batch[obj_no, i, gen] = -1.0*val
                    
        # reshape to:  torch.ones((max_gens, 4), device='cpu')*-torch.inf
        evals_by_batch = evals_by_batch.unsqueeze(1).repeat(1, 4) # line up with MOVE data (gross TODO)
        # save as evals_by_batch.pt and normed_fitness_by_batch.pt
        import torch
        print(evals_by_batch)
        print(normed_fitness_by_batch)
        torch.save(evals_by_batch, os.path.join(run_dir, "evals_by_batch.pt"))
        torch.save(normed_fitness_by_batch, os.path.join(run_dir, "normed_fitness_by_batch.pt"))

    