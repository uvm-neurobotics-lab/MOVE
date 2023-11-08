"""Stores configuration parameters for the MOVE algorithm."""
from cppn.activation_functions import *
# from evolution_torch import AlgorithmConfig
from cppn.config import CPPNConfig
import torch
import imageio.v2 as iio
from cppn.util import center_crop, resize
import logging

class MoveConfig(CPPNConfig):
    """Stores configuration parameters for the MOVE algorithm."""
    def __init__(self) -> None:
        # Initialize to default values:
        super().__init__()

        self.num_generations = 3_000
        
        #self.stop_condition = None
        #self.stop_condition_value = None
        self.stop_condition = "seconds"
        self.stop_condition_value = 60*60*8 # 8 hours
        

        self.activations=  [SinActivation,
                            IdentityActivation,
                            torch.nn.Tanh,
                            torch.nn.Sigmoid, 
                            ] # MOVE
        
        self.do_crossover = True
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
        self.target_resize = (128,128)
        self.color_mode = "HSL"

        self.activation_mode = "node"
        # self.output_activation = identity
        # self.output_activation = IdentityActivation
        self.output_activation = None
        self.normalize_outputs = "min_max"
        self.initial_mutations = 0
        
        self.diversity_mode = None # don't record diversity (it's slow)
        self.autoencoder_frequency = 0 # used for novelty, disabled for MOVE
        
        self.coord_range = (-1.0, 1.0)
        
        self.grad_every = 1
        
        self.batch_size = 1 # generational
        self.initial_batch_size = 1 # just for the initial population
        

        self.hidden_nodes_at_start = (16, )
        self.init_connection_probability = 0.55
        self.prune_threshold = 0 # don't prune
        self.min_pruned = 0
        self.dense_init_connections = False
        self.enforce_initial_fill = False
        self.fourier_sin_and_cos = False
        
        self.use_fourier_features = True
        self.n_fourier_features = 32
        self.fourier_feature_scale = 2.0
        self.fourier_mult_percent = 0.05
        
        self.sgd_clamp_weights = 10
        self.sgd_lr = 3e-1
        self.prob_mutate_weight = 0.0
        
        self.mutate_sgd_lr_sigma = 0.1
        
        # MOVE specific:
        self.move_fns_per_cell = 3
        self.allow_jumps = torch.inf
        self.num_cells = 50
        self.population_size = self.num_cells
        self.objective_functions =  None
        # self.objective_functions =  [
        #     "psnr",
        #     "mse",
        #     "ssim",
        #     "haarpsi",
        #     "vif",
        #     "msssim",
        #     "gmsd",
        #     "fsim", # nan in gradients
        #     "dss",
        #     "lpips",
        #     "dists",
        #     "style",
        #     "mdsi", # nan in gradients
        #     "vsi"   # nan in gradients
        #     ]
    
        self.prob_mutate_activation = .35
        self.prob_add_connection = .85 # 0.05 in the original NEAT
        self.prob_add_node = .85 # 0.03 in original NEAT
        self.prob_remove_node = 0.15
        self.prob_disable_connection = .15
        self.single_structural_mutation = True
        self.mutation_iters = 1
        
        self.low_mem = False # don't record as much data to save memory
        self.thread_count = 1 # don't use multiple threads
        
        self.norm_df_path = 'data/target_fitness_fn_ranges.csv'
        
        self.record_frequency = 1 # record every offspring
        
        # Used by baseline:
        self.num_children = 5
        
    
def resize_target(config):
    if not config.target_resize:
        return 
    device = config.target.device
    tar = config.target.cpu().numpy()
    
    # check if shape is already correct
    if tar.shape[:2] == config.target_resize:
        return
    
    res_fact = tar.shape[0] / config.target_resize[0], tar.shape[1] / config.target_resize[1]
    if res_fact[0] == 0.0 or res_fact[1] == 0.0:
        raise Exception("Target is too small to resize to target_resize")
    tar = resize(tar, (tar.shape[0] // int(res_fact[0]), tar.shape[1] // int(res_fact[1])))
    tar = center_crop(tar, config.target_resize[0], config.target_resize[1])
    
    config.target = torch.tensor(tar, dtype=torch.float32, device=device)
    config.set_res(*config.target_resize)
    
    
def apply_condition(config, controls, condition, name, name_to_function_map):
    config.name = name
    config.experiment_condition = name
    
    if len(controls) > 0:
        for k,v in controls.items():
            print("\t Control:", k, "->", v)
            config.apply(k, v)
            if k == "num_runs":
                config.num_runs = v
            if k == "target":
                config.target = v
                config.target_name = config.target
                if 'color_mode' in controls:
                    config.color_mode = controls['color_mode']
                pilmode = "RGB" if len(config.color_mode) == 3 else "L"
                config.target = torch.tensor(iio.imread(config.target, pilmode=pilmode), dtype=torch.float32, device=config.device)
                
                config.res_h, config.res_w = config.target.shape[:2]
        
    if len(condition) > 0:
        for k, v in condition.items():
            if k is not None:
                print(f"\t\tapply {k}->{v}")
                config.apply(k, v)
            if k == "target":
                if 'color_mode' in condition:
                    config.color_mode = condition['color_mode']
                config.target = v
                if isinstance(config.target, str):
                    config.target_name = config.target
                    pilmode = "RGB" if len(config.color_mode) == 3 else "L"
                    config.target = torch.tensor(iio.imread(config.target, pilmode=pilmode), dtype=torch.float32, device=config.device)

                config.res_h, config.res_w = config.target.shape[:2]
   
    # if config.color_mode=="L":
        # config.target = config.target.mean(dim=2)

    if config.target.max() > 1.0:
        config.target = config.target.to(torch.float32) /255.0
    
    resize_target(config)
    
    if len(config.target.shape) < len(config.color_mode):
        logging.warning("Color mode is RGB or HSV but target is grayscale. Setting color mode to L.")
        config.color_mode = "L"
        
    if config.color_mode == "L":
        if len(config.target.shape) == 2:
            config.target = config.target.unsqueeze(-1).repeat(1,1,3) # loss functions expect 3 channels
                
    
    if len(config.color_mode) != config.num_outputs:
        logging.warning("WARNING: color_mode does not match num_outputs. Setting num_outputs to len(color_mode)")
        config.num_outputs = len(config.color_mode)
    config.device = torch.device(config.device)
    config.target = config.target.to(config.device)     

    for i in range(len(config.activations)):
        if isinstance(config.activations[i], str):
            config.activations[i] = af.__dict__.get(config.activations[i])  
