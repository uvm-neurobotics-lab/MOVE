import os
import pandas as pd
import torch


class Record():
    def __init__(self, config, n_fns, n_cells, low_mem=False) -> None:
        self.low_mem = low_mem
        self.fit_df = pd.DataFrame(columns=['condition', 'target', 'cell', 'run', 'function', 'gen', 'fitness'])
        self.replace_df = pd.DataFrame(columns=['condition','target',  'cell', 'run', 'gen', 'replacements'])
        
        num_gen_data_points = config.num_generations
        if config.use_steady_state:
            num_gen_data_points = round(config.num_generations / config.population_size)
        
        self.agg_fitness_over_time = torch.ones((n_cells, num_gen_data_points), device='cpu')*-torch.inf
        
        if not self.low_mem:
            self.fitness_over_time = torch.ones((n_fns, n_cells, num_gen_data_points), device='cpu')*-torch.inf
            self.votes_over_time = torch.zeros((n_cells, n_fns, n_cells, num_gen_data_points), device='cpu')
            self.replacements_over_time = torch.zeros((n_cells, n_cells, num_gen_data_points), device='cpu')
            self.ids_over_time = torch.ones((n_cells, num_gen_data_points), device='cpu', dtype=torch.int64)*-1
            self.parents_over_time = torch.ones((2, n_cells, num_gen_data_points), device='cpu', dtype=torch.int64)*-1
        
    def update(self, gen_index, all_replacements, all_votes, map):
        self.agg_fitness_over_time[:,gen_index] = map.agg_fitness.cpu()
        
        if not self.low_mem:
            self.fitness_over_time[:,:,gen_index] = map.fitness.cpu()
            self.replacements_over_time[:,:,gen_index] = all_replacements.cpu()
            self.votes_over_time[:,:,:,gen_index] = all_votes
            self.ids_over_time[:,gen_index] = torch.tensor([-1 if g is None else g.id for g in map.map]).cpu()
            self.parents_over_time[:,:,gen_index] = torch.tensor([[-1 if g is None else g.parents[0] for g in map.map], [-1 if g is None else g.parents[1] for g in map.map]]).cpu()


    def save(self, run_dir):
        torch.save(self.agg_fitness_over_time, os.path.join(run_dir, "agg_fitness_over_time.pt"))
        if not self.low_mem:
            torch.save(self.fitness_over_time, os.path.join(run_dir, "fitness_over_time.pt"))
            torch.save(self.replacements_over_time, os.path.join(run_dir, "replacements_over_time.pt"))
            torch.save(self.ids_over_time, os.path.join(run_dir, "ids_over_time.pt"))
            torch.save(self.parents_over_time, os.path.join(run_dir, "parents_over_time.pt"))
            torch.save(self.votes_over_time, os.path.join(run_dir, "votes_over_time.pt"))