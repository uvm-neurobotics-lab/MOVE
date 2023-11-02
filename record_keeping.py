import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import Callable
import matplotlib.pyplot as plt
import json
import numpy as np

class Record():
    def __init__(self, config, n_fns, n_cells, low_mem=False) -> None:
        self.low_mem = low_mem
        self.fit_df = pd.DataFrame(columns=['condition', 'target', 'cell', 'run', 'function', 'gen', 'fitness'])
        self.replace_df = pd.DataFrame(columns=['condition','target',  'cell', 'run', 'gen', 'replacements'])
        
        num_gen_data_points = (config.num_generations*config.population_size)//config.record_frequency
        # if config.batch_size != config.population_size:
        #     num_gen_data_points = config.num_generations*config.batch_size//config.population_size
        self.agg_fitness_over_time = torch.ones((n_cells, num_gen_data_points), device='cpu')*-torch.inf
        
        self.n_evals = 0
        self.n_evals_incl_sgd = 0
        self.evals_over_time = torch.ones((num_gen_data_points), device='cpu')*-torch.inf
        self.evals_over_time_incl_sgd = torch.ones((num_gen_data_points), device='cpu')*-torch.inf
        
        if not self.low_mem:
            self.fitness_over_time = torch.ones((n_fns, n_cells, num_gen_data_points), device='cpu')*-torch.inf
            self.votes_over_time = torch.zeros((n_cells, n_fns, n_cells, num_gen_data_points), device='cpu')
            self.replacements_over_time = torch.zeros((n_cells, n_cells, num_gen_data_points), device='cpu')
            self.ids_over_time = torch.ones((n_cells, num_gen_data_points), device='cpu', dtype=torch.int64)*-1
            self.parents_over_time = torch.ones((2, n_cells, num_gen_data_points), device='cpu', dtype=torch.int64)*-1
            self.lr_over_time = torch.ones((num_gen_data_points), device='cpu')*-torch.inf
            self.offspring_over_time = torch.ones((num_gen_data_points), device='cpu')*-torch.inf
    
    def update_counts(self, gen, n_evals, n_evals_incl_sgd):
        self.n_evals += n_evals
        self.n_evals_incl_sgd += n_evals_incl_sgd
        self.evals_over_time[gen] = self.n_evals
        self.evals_over_time_incl_sgd[gen] = self.n_evals_incl_sgd
    
    def update(self, gen_index, all_replacements, map, total_offspring):
        self.agg_fitness_over_time[:,gen_index] = map.agg_fitness.cpu()
        if not self.low_mem:
            self.fitness_over_time[:,:,gen_index] = map.fitness.cpu()
            self.replacements_over_time[:,:,gen_index] = all_replacements.cpu()
            # self.votes_over_time[:,:,:,gen_index] = all_votes
            self.ids_over_time[:,gen_index] = torch.tensor([-1 if g is None else g.id for g in map.map]).cpu()
            self.parents_over_time[:,:,gen_index] = torch.tensor([[-1 if g is None else g.parents[0] for g in map.map], [-1 if g is None else g.parents[1] for g in map.map]]).cpu()
            self.lr_over_time[gen_index] = torch.mean(torch.tensor([g.sgd_lr for g in map.map if g is not None]))
            self.offspring_over_time[gen_index] = total_offspring

    def save(self, run_dir):
        torch.save(self.agg_fitness_over_time, os.path.join(run_dir, "agg_fitness_over_time.pt"))
        torch.save(self.evals_over_time, os.path.join(run_dir, "evals_over_time.pt"))
        torch.save(self.evals_over_time_incl_sgd, os.path.join(run_dir, "evals_over_time_incl_sgd.pt"))
        
        with open(os.path.join(run_dir, "evals.csv"), 'w') as f:
            f.write("evals,evals_incl_sgd\n")
            f.write(f"{self.n_evals},{self.n_evals_incl_sgd}\n")
            
        if not self.low_mem:
            torch.save(self.fitness_over_time, os.path.join(run_dir, "fitness_over_time.pt"))
            torch.save(self.replacements_over_time, os.path.join(run_dir, "replacements_over_time.pt"))
            torch.save(self.ids_over_time, os.path.join(run_dir, "ids_over_time.pt"))
            torch.save(self.parents_over_time, os.path.join(run_dir, "parents_over_time.pt"))
            torch.save(self.votes_over_time, os.path.join(run_dir, "votes_over_time.pt"))
            torch.save(self.lr_over_time, os.path.join(run_dir, "lr_over_time.pt"))
            torch.save(self.offspring_over_time, os.path.join(run_dir, "offspring_over_time.pt"))
            
     
    def save_map(self, dirpath, map, config, inputs):
        # save all images
        os.makedirs(dirpath, exist_ok=True)
        genomes_path = os.path.join(config.output_dir, "genomes", config.experiment_condition, f"run_{config.run_id:04d}")
        os.makedirs(genomes_path, exist_ok=True)
        genomes = []
        
        flat_map = map.get_population()
        print(f"Saving map with {len(flat_map)} genomes")
        pbar = tqdm(total=len(flat_map), desc="Saving final map...")
        imgs = []
        for i in range(len(flat_map)):
            cell_fns_inds = map.cell_fn_inds[i] # flat
            cell_fns = [map.fns[i] for i in cell_fns_inds]
            if(flat_map[i] is not None):
                individual = flat_map[i]
                individual.to(config.device)
                img = individual.get_image(inputs, channel_first=True, act_mode="node").detach().cpu()
                if len(config.color_mode)<3:
                    img = img.repeat(3, 1, 1)
                img = img.permute(1,2,0) # (H,W,C)
                img = img.numpy()
                imgs.append(img)
                name = "_".join([(fn.__name__ if isinstance(fn, Callable) else fn) for fn in cell_fns])+f"{len(list(individual.enabled_connections))}c"
                name = name + ".png"
                plt.imsave(os.path.join(dirpath, name), img, cmap='gray')
                plt.close()
                # if config.with_grad:
                    # flat_map[i].discard_grads()
                genomes.append(flat_map[i].clone(config, new_id=False).to_json())
            else:
                genomes.append("null")
            pbar.update(1)

            
        pbar.close()
        with open(os.path.join(genomes_path, "map.json"), "w") as f:
            json.dump(genomes, f)
      
        
        average_image = np.mean(imgs, axis=0) 
        plt.imsave(os.path.join(config.output_dir, "images", f"{config.run_id:04d}/avg_{config.run_id:04d}.png"), average_image, cmap='gray')
        
