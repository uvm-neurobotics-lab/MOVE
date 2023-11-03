"""Stores configuration parameters for the MOVE algorithm."""
from cppn.activation_functions import *
from evolution_torch import AlgorithmConfig
import torch


class MoveConfig(AlgorithmConfig):
    """Stores configuration parameters for the MOVE algorithm."""
    def __init__(self) -> None:
        # Initialize to default values:
        super().__init__()

        # Overrides:
        self.num_generations = 3_000
        # self.activations=  [sin,
        #                     gauss,
        #                     tanh,
        #                     sigmoid, 
        #                     clip,
        #                     torch.nn.Conv2d
                            # ] # MOVE
        self.activations=  [SinActivation,
                            IdentityActivation,
                            torch.nn.Tanh,
                            torch.nn.Sigmoid, 
                            ] # MOVE
        self.do_crossover = False
        self.population_elitism = 0
        self.tiebreak_novel = False
        self.num_upsamples = 0
        self.num_conv = 0
        self.max_weight = torch.inf
        self.weight_init_std = 2.0
        self.clamp_weights = False
        self.use_input_bias = False
        self.use_radial_distance = True
        self.num_inputs = 3 # x,y,d
        self.target_resize = (64,64)
        self.color_mode = "HSL"

        self.activation_mode = "node"
        # self.output_activation = identity
        self.output_activation = IdentityActivation
        self.normalize_outputs = "min_max"
        self.initial_mutations = 0
        
        self.diversity_mode = None # don't record diversity (it's slow)
        self.autoencoder_frequency = 0 # used for novelty, disabled for MOVE
        
        self.coord_range = (-0.5, 0.5)
        
        self.grad_every = 1
        
        self.batch_size = 50 # generational
        self.initial_batch_size = 50 # just for the initial population
        

        self.hidden_nodes_at_start = (4,4)
        self.init_connection_probability = 0.55
        self.prune_threshold = 0 # don't prune
        self.min_pruned = 0
        self.dense_init_connections = False
        self.enforce_initial_fill = False
        self.fourier_sin_and_cos = False
        
        self.use_fourier_features = True
        self.n_fourier_features = 8
        self.fourier_feature_scale = 2.0
        self.fourier_mult_percent = 0.05
        
        self.sgd_clamp_weights = 10
        self.sgd_lr = 1e-1
        self.prob_mutate_weight = 0.0
        
        self.mutate_sgd_lr_sigma = 0.1
        
        # MOVE specific:
        self.move_fns_per_cell = 3
        self.allow_jumps = torch.inf
        self.num_cells = 50
        self.population_size = self.num_cells
        self.objective_functions =  [
            "psnr",
            "mse",
            "ssim",
            "haarpsi",
            "vif",
            "msssim",
            "gmsd",
            "fsim", # nan in gradients
            "dss",
            "lpips",
            "dists",
            "style",
            "mdsi", # nan in gradients
            "vsi"   # nan in gradients
            ]
    
        self.prob_mutate_activation = .35
        self.prob_add_connection = .85 # 0.05 in the original NEAT
        self.prob_add_node = .85 # 0.03 in original NEAT
        self.prob_remove_node = 0.15
        self.prob_disable_connection = .15
        self.single_structural_mutation = True
        
        self.low_mem = False # don't record as much data to save memory
        self.thread_count = 1 # don't use multiple threads
        
        self.norm_df_path = 'data/target_fitness_fn_ranges.csv'
        
        self.record_frequency = 1 # record every offspring
        
        # Used by baseline:
        self.num_children = 5
        
        
        # force cppns to be regenerated if config type is changed
        self._make_dirty()
        