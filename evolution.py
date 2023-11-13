import math
import random
import time
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
import __main__ as main
if not hasattr(main, '__file__'):
    try:
        from tqdm.notebook import trange
    except ImportError:
        from tqdm import trange
else:
    from tqdm import trange
import pandas as pd
import torch
import os
from cppn.cppn import CPPN as ImageCPPN
import logging
from stopping import *
from util import get_dynamic_mut_rate


class CPPNEvolutionaryAlgorithm(object):
    def __init__(self, config, debug_output=False) -> None:
        self.config = config
        self.stop_condition = name_to_stop_condition_map[self.config.stop_condition](self.config.stop_condition_value)
        
        if self.config.with_grad:
            torch.autograd.set_grad_enabled(True)
        else:
            torch.autograd.set_grad_enabled(False)
        if self.config.autoencoder_frequency > 0:
            from evolution_torch.autoencoder import initialize_encoders, AutoEncoder
        
        if not hasattr(self, "inputs"):
            self.inputs = None # default to coord inputs in CPPN class
        
        self.init_dirs()
        
        
        self.gen = 0
        self.current_batch = 0
        self.debug_output = debug_output
        self.show_output = False
        
        # self.results = pd.DataFrame(columns=['condition', 'target', 'run', 'gen', 'batch', 'fitness', 'mean_fitness', 'diversity', 'population', 'avg_num_connections', 'avg_num_hidden_nodes', 'max_num_connections', 'max_num_hidden_nodes', 'time', 'total_offspring'])
                
        self.solutions_over_time = []
        self.time_elapsed = 0
        self.solution_generation = -1
        self.population = []
        self.solution = None
        self.this_gen_best = None
        self.novelty_archive = []
        self.device = config.device
        self.run_number = 0
        self.diversity = 0
        self.total_offspring = 0
        
        self.solution_fitness = -math.inf
        self.best_genome = None
        if self.config.genome_type is None:
            self.config.genome_type = ImageCPPN
        self.genome_type = config.genome_type

        self.target = self.config.target.to(self.device)
        
        if len(self.target.shape) < 3:
            # grayscale image
            if self.config.color_mode != "L":
                logging.warning("Target image is grayscale, but color_mode is not set to 'L'. Setting color_mode to 'L'")
                self.config.color_mode = "L"
                
        if self.config.res_w != self.target.shape[0]:
            self.config.res_w = self.target.shape[0]
            logging.warning("Target image width does not match config.res_w. Setting config.res_w to target image width")
        if self.config.res_h != self.target.shape[1]:
            self.config.res_h = self.target.shape[1]
            logging.warning("Target image height does not match config.res_h. Setting config.res_h to target image height")

        self.fitnesses = {}
        
        assert self.config.total_offspring >= self.config.num_cells, "total_offspring must be >= num_cells for now"
    
        initial_batches = math.ceil(self.config.num_cells / self.config.initial_batch_size)
        other_batches = math.ceil((self.config.total_offspring-self.config.num_cells) / self.config.batch_size)

    
        self.total_batches = initial_batches + other_batches
        print("Expecting up to", self.total_batches, "batches")
        print("Stop condition:", self.stop_condition.__class__.__name__ if self.stop_condition is not None else "None")
        
        
        
    def init_dirs(self):
        self.cond_dir = os.path.join(self.config.output_dir, "conditions", self.config.experiment_condition)
        os.makedirs(self.cond_dir, exist_ok=True)
        self.run_dir = os.path.join(self.cond_dir, f"run_{self.config.run_id:04d}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.image_dir = os.path.join(self.run_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.genomes_dir = os.path.join(self.run_dir, "genomes")
        os.makedirs(self.genomes_dir, exist_ok=True)

    
    def get_mutation_rates(self):
        """Get the mutate rates for the current generation 
        if using a mutation rate schedule, else use config values
            abandon all hope all ye who enter here

        Returns:
            float: prob_mutate_activation,
            float: prob_mutate_weight,
            float: prob_add_connection,
            float: prob_add_node,
            float: prob_remove_node,
            float: prob_disable_connection,
            float: weight_mutation_max, 
            float: prob_reenable_connection
        """
        
        if(self.config.use_dynamic_mutation_rates):
            # Behold:
            run_progress = self.total_offspring / self.config.total_offspring
            end_mod = self.config.dynamic_mutation_rate_end_modifier
            prob_mutate_activation   = get_dynamic_mut_rate(self.config.prob_mutate_activation,     run_progress, end_mod)
            prob_mutate_weight       = get_dynamic_mut_rate(self.config.prob_mutate_weight,         run_progress, end_mod)
            prob_add_connection      = get_dynamic_mut_rate(self.config.prob_add_connection,        run_progress, end_mod)            
            prob_add_node            = get_dynamic_mut_rate(self.config.prob_add_node,              run_progress, end_mod)
            prob_remove_node         = get_dynamic_mut_rate(self.config.prob_remove_node,           run_progress, end_mod)
            prob_disable_connection  = get_dynamic_mut_rate(self.config.prob_disable_connection,    run_progress, end_mod)
            weight_mutation_max      = get_dynamic_mut_rate(self.config.weight_mutation_max,        run_progress, end_mod)
            prob_reenable_connection = get_dynamic_mut_rate(self.config.prob_reenable_connection,   run_progress, end_mod)
            return  prob_mutate_activation, prob_mutate_weight, prob_add_connection, prob_add_node, prob_remove_node, prob_disable_connection, weight_mutation_max, prob_reenable_connection
        else:
            # just return the config values directly
            return  self.config.prob_mutate_activation, self.config.prob_mutate_weight, self.config.prob_add_connection, self.config.prob_add_node, self.config.prob_remove_node, self.config.prob_disable_connection, self.config.weight_mutation_max, self.config.prob_reenable_connection

    def evolve(self, run_number = 1, show_output=False, initial_population=True):
        self.start_time = time.time()
        self.run_number = run_number
        self.show_output = show_output or self.debug_output
        if initial_population:
            for i in range(self.config.num_cells): 
                self.population.append(self.genome_type(self.config)) # generate new random individuals as parents
            
            # update novelty encoder 
            if self.config.novelty_mode == "encoder":  
                initialize_encoders(self.config, self.target)  
            if self.config.activation_mode == "population":
                activate_population(self.population, self.config, self.inputs)
            else:
                for g in self.population: g(inputs=self.inputs)
            self.update_fitnesses_and_novelty()
            self.population = sorted(self.population, key=lambda x: x.fitness.item(), reverse=True) # sort by fitness
            self.solution = self.population[0].clone(cpu=True) 

        try:
            # Run algorithm
            pbar = trange(self.config.total_offspring, desc=f"Run {self.run_number}")
        
            while self.total_offspring < self.config.total_offspring:
                self.batch_start()
                self.run_one_batch()
                self.batch_end()
                b = self.get_best()
                if b is not None:
                    pbar.set_postfix_str(f"bf: {self.agg_fitnesses[b.id]:.4f} (id:{b.id}) d:{self.diversity:.4f} af:{np.mean(list(self.agg_fitnesses.values())):.4f} u:{self.n_unique}")
                else:
                    pbar.set_postfix_str(f"d:{self.diversity:.4f}")
                
                if self.stop_condition(self):
                    print(f"Stop condition: {self.stop_condition.__class__.__name__ if self.stop_condition is not None else 'None'} met")
                    break
                
                self.current_batch += 1
                
                # set pbar to self.total_offspring
                pbar.n = self.total_offspring
                pbar.refresh()

            
        except KeyboardInterrupt:
            self.on_end()
            raise KeyboardInterrupt()  
        
        self.on_end()

    def on_end(self):
        self.end_time = time.time()     
        self.time_elapsed = self.end_time - self.start_time  
        print("\n\nEvolution completed with", self.gen, "generations,", self.current_batch+1, "batches, and", self.total_offspring, "offspring", "in", self.time_elapsed, "seconds")
        print("Wrapping up, please wait...")

        # save results
        print("Saving data...")
        self.run_number = self.config.run_id
        
        
     
        with open(os.path.join(self.run_dir, f"target.txt"), 'w') as f:
            f.write(self.config.target_name)
        
        # save to run dir
        # save to output dir
        if False:
            self.results.loc[self.run_number, "run_id"] = self.config.run_id
            filename = os.path.join(self.run_dir, f"results.pkl")
            self.results.to_pickle(filename)
            filename = os.path.join(self.config.output_dir, f"results.pkl")
            if os.path.exists(filename):
                tries = 0
                while tries < 5:
                    try:
                        with open(filename, 'rb') as f:
                            save_results = pd.read_pickle(f)
                            save_results = pd.concat([save_results, self.results]).reset_index(drop=True)
                            break
                    except:
                        tries += 1
                        time.sleep(1)
                if tries == 5:
                    logging.warning("Failed to read output_dir results file, overwriting")
                    save_results = self.results
            else:
                save_results = self.results
            save_results.to_pickle(filename)
        
            
    def batch_start(self):
        """Called at the start of each generation"""

        if self.show_output:
            self.print_fitnesses()
            
        # update the autoencoder used for novelty
        if self.config.autoencoder_frequency > 0 and self.current_batch % self.config.autoencoder_frequency == 0:
            AutoEncoder.instance.update_novelty_network(self.population) 
            
    def run_one_batch(self):
        """Run one generation of the algorithm"""
        raise NotImplementedError("run_one_batch() not implemented for base class")

    def batch_end(self):
        """Called at the end of each generation"""
        self.record_keeping()

    def get_best(self)->ImageCPPN:
        if len(self.population) == 0:
            print("No individuals in population")
            return None
        max_fitness_individual = max(self.population, key=lambda x: self.agg_fitnesses[x.id])
        return max_fitness_individual
    