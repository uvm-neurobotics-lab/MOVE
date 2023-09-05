"""Stores configuration parameters for the MOVE algorithm."""
from cppn_torch.activation_functions import *
from evolution_torch import AlgorithmConfig
import torch


class MoveConfig(AlgorithmConfig):
    """Stores configuration parameters for the MOVE algorithm."""
    def __init__(self) -> None:
        # Initialize to default values:
        super().__init__()

        # Overrides:
        self.num_generations = 3_000
        self.activations=  [sin, gauss, softplus, sigmoid, linear] # MOVE
        self.do_crossover = False
        self.population_elitism = 0
        self.tiebreak_novel = False
        self.num_upsamples = 0
        self.num_conv = 0
        self.max_weight = torch.inf
        self.weight_init_std = 2.0
        self.clamp_weights = False
        self.use_input_bias = True
        self.use_radial_distance = True
        self.num_inputs = 4 # x,y,b,d
        self.target_resize = (100,100)
        self.color_mode = "HSL"

        self.activation_mode = "node"
        self.output_activation = identity
        self.normalize_outputs = "min_max"
        self.initial_mutations = 0
        
        self.diversity_mode = None # don't record diversity (it's slow)
        self.autoencoder_frequency = 0 # used for novelty, disabled for MOVE
        
        self.coord_range = (-0.5, 0.5)
        
        self.grad_every = 1
        
        self.batch_size = 1
        self.initial_batch_size = 64 # just for the initial population
        

        self.hidden_nodes_at_start = 8
        self.init_connection_probability = 0.85
        self.prune_threshold = 0 # don't prune
        self.min_pruned = 0
        self.dense_init_connections = False
        self.enforce_initial_fill = False
        self.fourier_sin_and_cos = False
        
        # MOVE specific:
        self.move_fns_per_cell = 3
        self.allow_jumps = torch.inf
        self.num_cells = 50
        self.population_size = self.num_cells
        self.objective_functions =  [
            "psnr",
            "mse",
            "lpips",
            "dists",
            "ssim",
            "style",
            "haarpsi",
            "vif",
            "msssim",
            "dss",
            "gmsd",
            "fsim",
            "mdsi",
            "vsi"
            ]
        
        self.low_mem = False # don't record as much data to save memory
        self.thread_count = 1 # don't use multiple threads
        
        self.norm_df_path = 'data/target_fitness_fn_ranges.csv'
        
        self.record_frequency = 1 # record every offspring
        
        # Used by baseline:
        self.num_children = 5
        
        self._make_dirty() # force cppns to be regenerated if config type is changed
        
            