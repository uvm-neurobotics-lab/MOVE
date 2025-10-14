"""Stores configuration parameters for the MOVE algorithm."""
import os
from typing import Callable
import uuid
import torch
import imageio.v2 as iio
import logging
from .cppn.activation_functions import *
from .cppn.config import CPPNConfig
from .cppn.util import center_crop, resize
from .fitness.name_to_fn import name_to_fn
from torchvision.transforms import Resize
from .fitness import fitness_functions as ff


class MOVEConfig(CPPNConfig):
    """Stores configuration parameters for the MOVE algorithm."""
    def __init__(self) -> None:
        # Initialize to default values:
        super().__init__()

        self.alg = "MOVE"
        self.name = "default" # name for this configuration
        self.run_id = uuid.uuid1().int>>64

        self.target = None # set later
        self.do_profile = False
        
        self.checkpoint_frequency = 0

        self.total_offspring = 30_000_000
        # self.total_offspring = 3000
        
        self.target_path = "default"
        
        
        # self.stop_condition = "batches"
        self.stop_condition = "evals"
        self.stop_condition_value = 100*100*3*10 # about 10 batches x 100 cells x 100 sgd steps (fwd + bwd + eval)


        # self.activations=  [SinActivation,
        #                     IdentityActivation,
        #                     TanhActivation,
        #                     SigmoidActivation, 
        #                     GaussActivation,
        #                     CosActivation
        #                     ] # MOVE
    
        self.activations=  [SinActivation,
                            IdentityActivation,
                            TanhActivation,
                            SigmoidActivation, 
                            ] # GECCO24
        
        # self.activations=  [SinActivation,
        #                     IdentityActivation,
        #                     TanhActivation,
        #                     SigmoidActivation, 
        #                     Conv3x3Activation,
        #                     Conv5x5Activation,
        #                     KernelBlurActivation,
        #                     KernelSharpenActivation,
        #                     KernelEdgeActivation,
        #                     KernelEmbossActivation,
        #                     torch.nn.Hardshrink,
        
        #                     ] 
        
        self.soft_mask_sigma = None # don't use a soft mask (binary mask)
        self.soft_mask_mu = 0.0 
        self.soft_mask_min = 0.0
        
        self.soft_replace = False # don't use soft replace
        self.soft_replace_mod = 2.0 # 2 times as likely to replace

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
        self.target_resize = (33,33)
        self.color_mode = "HSL"
        self.bloat_prune_ratio = 0.0

        self.activation_mode = "node"
        # self.output_activation = identity
        # self.output_activation = IdentityActivation
        self.output_activation = None
        self.normalize_outputs = "min_max"
        self.initial_mutations = 0
        
        self.diversity_mode = None # don't record diversity (it's slow)
        self.autoencoder_frequency = 0 # used for novelty, disabled for MOVE
        
        self.coord_range = (-0.5, 0.5)
        
        self.grad_every = 1
        self.evolve_every = 1

        self.evolve_only_after_sgd = False
        if self.evolve_only_after_sgd:
            self.evolve_every = self.grad_every
        

        
        self.comparison_batch_size = None # same as batch_size

        self.hidden_nodes_at_start = (16, )
        self.init_connection_probability = 0.50
        self.init_connection_probability_fourier = 0.10
        self.force_init_path_inputs_outputs = True
        self.prune_threshold = 0.0 # don't prune
        self.prune_threshold_nodes = 0.0 # don't prune nodes
        self.min_pruned = 0
        self.min_pruned_nodes = 0
        self.node_activation_prune_threshold = 0.0 # don't prune by activation
        self.dense_init_connections = False
        self.enforce_initial_fill = False
        self.fourier_sin_and_cos = False
        
        self.use_fourier_features = False
        self.n_fourier_features = 16
        self.fourier_feature_scale = 2.0
        self.fourier_mult_percent = 0.05
        

        
        # MOVE specific:
        self.move_fns_per_cell = 3
        self.allow_jumps = torch.inf
        self.num_cells = 100
        # self.batch_size = 1
        self.batch_size = self.num_cells
        self.initial_batch_size = self.num_cells # just for the initial population
        
        # self.objective_functions =  None
        self.objective_functions =  [
                "mse",
                "psnr",
                "lpips",
                "dists",
                "style",
                "vif",
                "dss",
                "ssim",
                "msssim",
                "haarpsi",
            ]
    
        self.prob_mutate_activation = .35
        self.prob_add_connection = .85 # 0.05 in the original NEAT
        self.prob_add_node = .85 # 0.03 in original NEAT
        self.prob_remove_node = 0.15
        self.prob_disable_connection = .15
        self.single_structural_mutation = False
        self.topology_mutation_iters = 1
        self.connection_bloat = 0 # don't bloat extra connections
        
        self.low_mem = False # don't record as much data to save memory
        self.thread_count = 1 # don't use multiple threads
        
        self.norm_df_path = 'data/target_fitness_fn_ranges.csv'
        
        self.record_frequency_batch = 1 # record every batch
    
    def intialize_linked_variables(self):
        if self.evolve_only_after_sgd:
            self.evolve_every = self.grad_every
        
        self.num_outputs = len(self.color_mode)

    def fns_to_strings(self):
        """Converts the activation functions to strings."""
        super().fns_to_strings()
        if hasattr(self, 'objective_functions') and self.objective_functions is not None:
            for i, fn in enumerate(self.objective_functions):
                if isinstance(fn, Callable):
                    self.objective_functions[i] = fn.__name__
            
        if hasattr(self, "fitness_schedule") and self.fitness_schedule is not None:
            for i, fn in enumerate(self.fitness_schedule):
                if isinstance(fn, Callable):
                    self.fitness_schedule[i] = fn.__name__
        
        if hasattr(self, "target_path") and self.target_path is not None:
            self.target = self.target_path
        
        self.dtype = str(self.dtype) # TODO deserialize 
        
        for fn in self.NO_GRADIENT:
            if isinstance(fn, Callable):
                self.NO_GRADIENT[self.NO_GRADIENT.index(fn)] = fn.__name__


    def strings_to_fns(self):
        """Converts the activation functions to functions."""
        super().strings_to_fns()
        if hasattr(self, "objective_functions") and self.objective_functions is not None:
            for i, fn in enumerate(self.objective_functions):
                if isinstance(fn, str):
                    self.objective_functions[i] = name_to_fn[fn]
    
        if hasattr(self, "fitness_schedule") and self.fitness_schedule is not None:
            for i, fn in enumerate(self.fitness_schedule):
                if isinstance(fn, str):
                    self.fitness_schedule[i] = name_to_fn[fn]

        for fn in self.NO_GRADIENT:
            if isinstance(fn, str):
                self.NO_GRADIENT[self.NO_GRADIENT.index(fn)] = name_to_fn[fn]
                
                
    def setup(self):
        super().setup()
        target_path_to_tensor(self)
        self.device = torch.device(self.device)
        for i in range(len(self.activations)):
            if isinstance(self.activations[i], str):
                self.activations[i] = name_to_fn[self.activations[i]]
            if hasattr(self, "fitness_function") and isinstance(self.fitness_function, str):
                if self.fitness_function in name_to_fn:
                    self.fitness_function = name_to_fn[self.fitness_function]
        
        for i in range(len(self.objective_functions)):
            if isinstance(self.objective_functions[i], str):
                self.objective_functions[i] = name_to_fn[self.objective_functions[i]]
        
        self.NO_GRADIENT = ff.NO_GRADIENT
        self.intialize_linked_variables()
        

def resize_image(image, size, device):
    # resize such that the smallest dimension is size
    h, w = image.shape[:2]
    if h < w:
        new_h = size[0]
        new_w = int(w * size[1] / h)
    else:
        new_w = size[1]
        new_h = int(h * size[0] / w)
        
    image = resize(image, (new_h, new_w))
    image = center_crop(image, size[0], size[1])
    resized_img = torch.tensor(image, dtype=torch.float32, device=device)
    return resized_img


def resize_target(config):
    if not config.target_resize:
        return 
    device = config.target.device
    tar = config.target.cpu().numpy()
    
    # check if shape is already correct
    if tar.shape[:2] == config.target_resize:
        return

    config.target = resize_image(tar, config.target_resize, device)
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

        
    if len(condition) > 0:
        for k, v in condition.items():
            if k is not None:
                print(f"\t\tapply {k}->{v}")
                config.apply(k, v)
    
    config.setup()


def target_path_to_tensor(config):
    if (config.target_path is None or config.target_path.strip()=="default") and (type(config.target)== str or type(config.target) == os.PathLike):
        config.target_path = config.target
        pilmode = "RGB" if len(config.color_mode) == 3 else "L"
        config.target = torch.tensor(iio.imread(config.target, pilmode=pilmode), dtype=torch.float32, device=config.device)
    else:
        pilmode = "RGB" if len(config.color_mode) == 3 else "L"
        config.target = torch.tensor(iio.imread(config.target_path, pilmode=pilmode), dtype=torch.float32, device=config.device)
        if config.target.max() > 1.0:
            config.target = config.target / 255.0

    resize_target(config)
    config.target = torch.stack([config.target.squeeze() for _ in range(config.initial_batch_size)])

    if len(config.target.shape) > 3:
        config.target = config.target.permute(0, 3, 1, 2) # move color channel to front
    else:
        config.target = config.target.unsqueeze(1).repeat(1,3,1,1) # add color channel
        
    if config.target.shape[-2] < 33 or config.target.shape[-1] < 33:
        config.target = Resize((33,33), antialias=True)(config.target)
    
    config.target = torch.clamp(config.target, 0, 1)

    config.target = config.target.to(config.device)
        
    if len(config.target.shape) < 3:
        # grayscale image
        if config.color_mode != "L":
            logging.warning("Target image is grayscale, but color_mode is not set to 'L'. Setting color_mode to 'L'")
            config.color_mode = "L"
            
    if config.res_w != config.target.shape[2]:
        config.res_w = config.target.shape[2]
        logging.warning("Target image width does not match config.res_w. Setting config.res_w to config.target image width")
    if config.res_h != config.target.shape[3]:
        config.res_h = config.target.shape[3]
        logging.warning("Target image height does not match config.res_h. Setting config.res_h to config.target image height")

    return config.target
