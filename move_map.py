import itertools
import random
import numpy as np

import torch

class MOVEMap(object):
    def __init__(self, config) -> None:
        self.fns = config.fns
        self.config = config
        
        if self.fns is None:
            return

        self.n_fns = len(self.fns)
        self.fns_per_cell = self.config.move_fns_per_cell
        
        assert self.fns_per_cell <= self.n_fns, "Cannot have more functions per cell than total functions"
        assert self.fns_per_cell % 2 == 1, "Must have an odd number of functions per cell"
        
        if hasattr(self.config, "load_cell_fns"):
            fn_names = self.fns
            
            with open(self.config.load_cell_fns, "r") as f:
                cell_names = f.readlines()
            for i, cell_name in enumerate(cell_names):
                cell_names[i] = cell_name.strip()
            cell_names = [cell_name.split("_") for cell_name in cell_names]
            combs = [[fn_names.index(cell_name) for cell_name in cell_names[i]] for i in range(len(cell_names))]
            self.n_cells = len(cell_names)
            print("Using loaded cells: ", combs)
            
        else:
            # assign combinations to cells
            combs = list(itertools.combinations(torch.arange(len(self.fns)),  self.fns_per_cell))
            max_count = len(combs)
            if not hasattr(self.config, "cell_fns_seed"):
                combs_random = random
            else:
                combs_random  = random.Random(self.config.cell_fns_seed)
                
            if hasattr(self.config, "num_cells"):
                if len(combs) > self.config.num_cells:
                    combs = combs_random.sample(combs, self.config.num_cells)
                elif len(combs) < self.config.num_cells:
                    print("WARN: less combinations than cells, filling with random repeats")
                    while len(combs) < self.config.num_cells:
                        if not hasattr(self.config, "cell_fns_seed"):
                            combs.append(combs_random.choice(combs))
                        else:
                            combs.append(combs_random.choice(combs))
            else:
                raise ValueError("Must specify num_cells")
            
            if hasattr(self.config, "cell_fns_seed"):
                print("Using fixed combs: ", combs)
        
        self.n_cells = len(combs)
        print(f"Using {self.n_cells} out of {max_count} possible combinations")
        self.map = [None] * np.prod(self.n_cells) # fill with None
        self.cell_fn_inds = torch.tensor(combs, device=self.config.device)
        self.cell_names = []
        for c in combs:
            self.cell_names.append("_".join([self.fns[i] for i in c]))

        assert self.n_cells == self.config.population_size, "Must have same number of cells as population size for now"
        print("Number of cells: ", self.n_cells)
        
        # use novelty as a tiebreaker?
        self.tiebreak_novel = bool(config.tiebreak_novel)
        if self.tiebreak_novel:
            if config.autoencoder_frequency <= 0:
                raise ValueError("Cannot use novelty tiebreaker without an autoencoder.")
            raise NotImplementedError("Novelty tiebreaker not implemented yet")
                
        # Set up fn mask matrix
        self.fn_mask = torch.zeros((self.n_fns, self.n_cells), device=self.config.device, dtype=torch.bool) # fns in each cell
        for i, comb in enumerate(combs):
            for j in comb:
                self.fn_mask[j, i] = 1
                
        # sum of each col should be self.fns_per_cell
        assert torch.all(torch.sum(self.fn_mask, dim=0) == self.fns_per_cell)
        
        # Set up fitness matrix
        self.fitness = torch.ones((self.n_fns, self.n_cells), device=self.config.device)*-torch.inf # fns in each cell
        
        self.agg_fitness = torch.zeros((self.n_cells), device="cpu") # fitness of each cell (normed)
        

    def get_population(self, include_empty=False):
        output = []
        output = np.array(self.map).flatten()
        if include_empty:
            return output.tolist()
        output = output[output != None]
        output = output.tolist()
        return output


    def get_agg_fitnesses(self):
        agg_fitnesses = {}
        for i, c in enumerate(self.map):
            if c is not None:
                agg_fitnesses[c.id] = self.agg_fitness[i].item()
        return agg_fitnesses

    def get_fitnesses(self):
        return self.fitness.cpu()

        
    def random_non_empty_cell(self):
        pop = self.get_population()
        if len(pop) == 0 :
            return None
        
        return random.choice(pop)


    def count_full_cells(self):
        pop = self.get_population()
        return len(pop) 
    