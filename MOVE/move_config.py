"""Stores configuration parameters for the MOVE algorithm."""
import os
import math
from typing import Callable
import uuid
import torch
import imageio.v2 as iio
import logging
import numpy as np
from .cppn.activation_functions import *
from .cppn.config import CPPNConfig
from .cppn.util import center_crop, resize
from .fitness.name_to_fn import name_to_fn
from torchvision.transforms import Resize
from .fitness import fitness_functions as ff
from .clip.semantic_targets import DEFAULT_STOP_WORDS


class MOVEConfig(CPPNConfig):
    """Stores configuration parameters for the MOVE algorithm."""
    def __init__(self) -> None:
        # Initialize to default values:
        super().__init__()

        self.alg = "MOVE"
        self.name = "default" # name for this configuration
        self.run_id = uuid.uuid1().int>>64

        self.target = None # set later
        self.clip_text_target = None
        self.clip_num_variants = 8
        self.clip_noise_scale = 0.2
        self.clip_random_seed = None
        self.clip_noise_anneal = False
        self.clip_noise_final_scale = 0.0
        self.clip_noise_anneal_start = 0.0
        self.clip_noise_anneal_end = 1.0
        self.clip_noise_anneal_power = 1.0
        self.clip_include_partials = True
        # Alias / clarity flag: when True, skip token-level (partial prompt) objectives.
        # This keeps CLIP objectives aligned to full prompt strings only.
        self.clip_disable_partials = False
        self.clip_partial_min_length = 3
        self.clip_partial_stopwords = list(DEFAULT_STOP_WORDS)
        self.clip_max_partial_prompts = 8
        self.clip_microbatch_size = 0
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

        # Functions that cannot be differentiated (copied so we can safely mutate)
        self.NO_GRADIENT = list(ff.NO_GRADIENT)
    
        self.prob_mutate_activation = .35
        self.prob_add_connection = .85 # 0.05 in the original NEAT
        self.prob_add_node = .85 # 0.03 in original NEAT
        self.prob_remove_node = 0.15
        self.prob_disable_connection = .15
        self.single_structural_mutation = False
        self.topology_mutation_iters = 1
        self.connection_bloat = 0 # don't bloat extra connections
        
        self.low_mem      = False # don't record as much data to save memory
        self.thread_count = 1 # don't use multiple threads
        
        # Performance optimizations
        self.use_channels_last   = True # Use channels_last memory format 
        self.use_torch_compile   = True # Enable torch.compile for PyTorch 2.0+ 
        self.use_fused_optimizer = True # Use fused AdamW optimizer on CUDA
        self.use_pinned_memory   = True # Use pinned memory for faster CPU-GPU
        
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
                if isinstance(fn, str) and fn in name_to_fn:
                    self.objective_functions[i] = name_to_fn[fn]
    
        if hasattr(self, "fitness_schedule") and self.fitness_schedule is not None:
            for i, fn in enumerate(self.fitness_schedule):
                if isinstance(fn, str) and fn in name_to_fn:
                    self.fitness_schedule[i] = name_to_fn[fn]

        for fn in self.NO_GRADIENT:
            if isinstance(fn, str) and fn in name_to_fn:
                self.NO_GRADIENT[self.NO_GRADIENT.index(fn)] = name_to_fn[fn]
                
                
    def setup(self):
        super().setup()
        target_path_to_tensor(self)
        self.device = torch.device(self.device)
        
        # Enable cuDNN benchmarking for consistent input sizes
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            logging.info("Enabled cuDNN benchmark mode for faster CUDA operations")
            
            # Enable TF32 for Ampere+ GPUs (A100, RTX 3090, etc.)
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logging.info("Enabled TF32 for faster matrix operations on Ampere+ GPUs")
            
            # Set float32 matmul precision for better performance
            if hasattr(torch, 'set_float32_matmul_precision'):
                try:
                    torch.set_float32_matmul_precision('high')  # Use TF32 when available
                    logging.info("Set float32 matmul precision to 'high' for better performance")
                except Exception:
                    pass
        
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
    """Resizes an image or batch of images, preserving aspect ratio before center cropping."""
    target_h = max(1, int(size[0]))
    target_w = max(1, int(size[1]))

    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = np.asarray(image)

    print(f"resize_image input shape: {image_np.shape}")

    if image_np.ndim < 2:
        raise ValueError(f"Unsupported image rank {image_np.ndim}; expected at least 2 dimensions.")

    def _resize_sample(sample: np.ndarray) -> np.ndarray:
        h, w = sample.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Cannot resize image with non-positive dimensions: {h}x{w}")

        if h < w:
            scale = target_h / h
            new_h = target_h
            new_w = max(target_w, int(math.ceil(w * scale)))
        else:
            scale = target_w / w
            new_w = target_w
            new_h = max(target_h, int(math.ceil(h * scale)))

        resized = resize(sample, (new_h, new_w))
        return center_crop(resized, target_h, target_w)

    if image_np.ndim == 2:
        channel_last = image_np[..., None]
        resized = _resize_sample(channel_last)[..., 0]
        return torch.from_numpy(resized).to(device=device, dtype=torch.float32)

    if image_np.ndim == 3:
        if image_np.shape[-1] in (1, 2, 3, 4) and image_np.shape[0] not in (1, 2, 3, 4):
            layout = "hwc"
            channel_last = image_np
        elif image_np.shape[0] in (1, 2, 3, 4):
            layout = "chw"
            channel_last = np.moveaxis(image_np, 0, -1)
        else:
            layout = "hwc"
            channel_last = image_np

        resized = _resize_sample(channel_last)
        if layout == "chw":
            resized = np.moveaxis(resized, -1, 0)
        return torch.from_numpy(resized).to(device=device, dtype=torch.float32)

    if image_np.ndim == 4:
        if image_np.shape[-1] in (1, 2, 3, 4) and image_np.shape[1] not in (1, 2, 3, 4):
            layout = "bhwc"
            channel_last = image_np
        elif image_np.shape[1] in (1, 2, 3, 4):
            layout = "bchw"
            channel_last = np.moveaxis(image_np, 1, -1)
        else:
            raise ValueError(f"Unsupported channel configuration for image shape {image_np.shape}")

        resized_samples = [_resize_sample(sample) for sample in channel_last]
        resized_stack = np.stack(resized_samples, axis=0)
        if layout == "bchw":
            resized_stack = np.moveaxis(resized_stack, -1, 1)
        return torch.from_numpy(resized_stack).to(device=device, dtype=torch.float32)

    raise ValueError(f"Unsupported image rank {image_np.ndim} for resizing.")


def resize_target(config):
    if not config.target_resize:
        return 
    if config.target is None:
        return
    device = config.target.device
    tar = config.target.cpu().numpy()
    
    # check if shape is already correct
    if tar.shape[-2:] == tuple(config.target_resize):
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
    raw_target = config.target if config.target is not None else config.target_path

    if raw_target is None or (isinstance(raw_target, str) and raw_target.strip() == "default"):
        return config.target

    # Support specifying multiple CLIP prompts as a list/tuple of strings.
    # In this case we enter CLIP mode and skip image target setup entirely.
    if isinstance(raw_target, (list, tuple)):
        prompts = []
        for item in raw_target:
            if item is None:
                continue
            if not isinstance(item, str):
                raise ValueError(
                    "Unsupported target list specification; expected a list of prompt strings"
                )
            text = item.strip()
            if not text:
                continue
            prompts.append(text)

        if not prompts:
            raise ValueError("Target list is empty; provide at least one prompt string")

        config.clip_text_target = prompts
        config.target = None
        config.target_path = None
        return None

    pilmode = "RGB" if len(config.color_mode) == 3 else "L"

    if isinstance(raw_target, (str, os.PathLike)):
        candidate_path = os.fspath(raw_target)
        if os.path.exists(candidate_path):
            config.target_path = candidate_path
            config.target = torch.tensor(iio.imread(candidate_path, pilmode=pilmode), dtype=torch.float32, device=config.device)
        else:
            config.clip_text_target = str(raw_target)
            config.target = None
            config.target_path = None
            return None
    elif isinstance(raw_target, torch.Tensor):
        config.target = raw_target.to(device=config.device)
    else:
        raise ValueError("Unsupported target specification; provide an image tensor, path, or text string")

    if config.target.max() > 1.0:
        logging.warning(f"Target image max value is {config.target.max()}, scaling to [0,1]")
        config.target = config.target / 255.0

    resize_target(config)

    target = config.target
    channel_counts = {1, 2, 3, 4}

    if not isinstance(target, torch.Tensor):
        target = torch.as_tensor(target, dtype=torch.float32, device=config.device)
    else:
        target = target.to(device=config.device, dtype=torch.float32)

    if target.ndim == 2:  # H, W
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        if target.shape[0] in channel_counts and target.shape[-1] not in channel_counts:
            target = target.unsqueeze(0)
        elif target.shape[-1] in channel_counts:
            target = target.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Ambiguous target shape {tuple(target.shape)}; unable to infer channels.")
    elif target.ndim == 4:
        if target.shape[1] in channel_counts and target.shape[-1] not in channel_counts:
            pass  # already B,C,H,W
        elif target.shape[-1] in channel_counts:
            target = target.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Ambiguous batched target shape {tuple(target.shape)}; unable to infer channels.")
    else:
        raise ValueError(f"Unsupported target tensor rank {target.ndim}; expected 2D-4D input.")

    batch_size = config.initial_batch_size or 1
    if target.shape[0] == 1 and batch_size > 1:
        target = target.repeat(batch_size, 1, 1, 1)
    elif target.shape[0] != batch_size:
        logging.warning(
            "Target batch dimension %d does not match initial_batch_size %d; duplicating first sample.",
            target.shape[0], batch_size)
        target = target[:1].repeat(batch_size, 1, 1, 1)

    if target.shape[1] == 1 and len(config.color_mode) == 3:
        target = target.repeat(1, 3, 1, 1)
    elif target.shape[1] != len(config.color_mode):
        logging.warning(
            "Target channel count %d does not match color_mode '%s'; adjusting via repeat as needed.",
            target.shape[1], config.color_mode)
        if target.shape[1] == 1:
            target = target.repeat(1, len(config.color_mode), 1, 1)

    config.target = target
        
    if config.target.shape[-2] < 33 or config.target.shape[-1] < 33:
        config.target = Resize((33,33), antialias=True)(config.target)
    
    config.target = torch.clamp(config.target, 0, 1)
    config.target = config.target.to(config.device)
        
    if len(config.target.shape) < 3:
        # grayscale image
        if config.color_mode != "L":
            logging.warning("Target image is grayscale, but color_mode is not set to 'L'. Setting color_mode to 'L'")
            config.color_mode = "L"

    if config.target_resize:
        config.set_res(*config.target_resize)
    else:
        config.set_res(config.target.shape[-2], config.target.shape[-1])

    return config.target
