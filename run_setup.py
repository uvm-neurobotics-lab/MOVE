
import json
import os
import imageio.v2 as iio
from evolution_torch import apply_condition
import torch
import matplotlib.pyplot as plt
import fitness.fitness_functions as ff
from move_config import MoveConfig
from cppn_torch import ImageCPPN

def fix_target_dimensions(config):
    dims = config.target.shape
    new_dims = []
    for i,d in enumerate(dims[:-1]):
        if d % 2**config.num_upsamples != 0:
            new_dims.append(d - d % 2**config.num_upsamples)
            print(f"Warning: target image dimension {d} is not divisible by 2^{config.num_upsamples}. Resizing to {new_dims[-1]}")
        else:
            new_dims.append(d)
    config.target = config.target[:new_dims[0], :new_dims[1], :]
    
    

def run_setup():
    import argparse
    import uuid
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='default.json', help='Path to config json file (default: default.json).')
    parser.add_argument('--generations', '-g', type=int, default=-1, help='Number of generations to run (default: 1000).')
    parser.add_argument('--population', '-p', type=int, default=-1, help='Population size.')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory.')
    parser.add_argument('-t','--target', action='store', help='Target image.')
    parser.add_argument('-v','--verbose', action='store_true', help='Print verbose output (default: False).')
    parser.add_argument('-d','--device', action='store', default=default_device, help=f'Device to run on (default: {default_device}).')
    parser.add_argument('-sgd','--sgd', type=bool, action='store', default=None, help=f'Use SGD to update weights (default: True).')
    parser.add_argument('-ff','--num_fourier_features',type=int, action='store', default=-1, help=f'Number of fourier features (default: 8).')
    parser.add_argument('-hn','--num_hidden_nodes',type=int, action='store', default=-1, help=f'Number of hidden nodes at initialization (default: 0).')
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        json_str = f.read()
    
    parsed = json.loads(json_str)

    
    # change controls based on command line args:
    if args.target is not None:
        parsed['controls']["target"] = args.target
    if args.population > 0:
        parsed['controls']["population_size"] = args.population
        parsed['controls']["num_cells"] = args.population
        parsed['controls']["num_children"] = args.population
    if args.generations > 0 :
        parsed['controls']["num_generations"] = args.generations
    if args.output is not None:
        parsed['controls']["output_dir"] = args.output
        
    if args.sgd:
        parsed['controls']["with_grad"] = True
    
    if args.num_fourier_features > 0:
        parsed['controls']["n_fourier_features"] = args.num_fourier_features
        parsed['controls']["use_fourier_features"] = True
        
    if args.num_hidden_nodes > 0:
        parsed['controls']["hidden_nodes_at_start"] = args.num_hidden_nodes
    
    config = MoveConfig()
    config.device = torch.device(args.device)
    apply_condition(config, parsed.get("controls", {}), {}, parsed.get("name", "Default"), ff.__dict__)
    
    conditions = parsed.get("conditions", [])

    for experiment in conditions:
        name = list(experiment.keys())[0]
        apply_condition(config, {}, experiment[name], name, ff.__dict__)

        if args.output is None:
            args.output = os.path.join("output", name)
        
        if config.allow_jumps == "inf":
            config.allow_jumps = torch.inf
        
        config.output_dir = args.output
        config.genome_type = ImageCPPN
        config.run_id = uuid.uuid1().int>>64 # generate a random (based on clock) 64-bit integer for id
        
    
        fix_target_dimensions(config)
        print("Target shape: {}".format(list(config.target.shape)))
        
        yield config, args.verbose
        
        