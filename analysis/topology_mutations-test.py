import json
import os
from typing import Callable
import random
import copy
import torch
import logging
import math
import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gc

from tqdm import tqdm
from torchvision.transforms import Resize

from MOVE.cppn.fourier_features import add_fourier_features

import imageio as iio


# from cppn_torch import ImageCPPN

# from cppn_torch.graph_util import activate_population
# from cppn.util import visualize_network, initialize_inputs
from MOVE.cppn.util import *
from MOVE.cppn.cppn import CPPN
from MOVE.move_config import MOVEConfig, resize_image
MoveConfig = MOVEConfig

from MOVE.util import *

from MOVE.move_map import MOVEMap
from MOVE.run_setup import run_setup, apply_condition, fix_target_dimensions
from MOVE.sgd_weights import sgd_weights, sgd_weights_imaml
from MOVE.record_keeping import Record

from MOVE.norm import norm_tensor, read_norm_data
from MOVE.fitness import fitness_functions as ff

from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load exiting checkpointed CPPNs from real runs to compare
# path_to_runs = "/users/j/s/jsdean/scratch/move/results/telo-24-growth/conditions/MOVE-SGD-Fast-Grow/"
path_to_conds = "/users/j/s/jsdean/scratch/move/results/telo-24-growth-sgd/conditions/"
conditions = "*"

targets = ["data/skull.png", "data/apple.png", "data/butterfly.png", "data/dolphin.png", "data/sunrise.png"]
# targets = ["data/butterfly.png"]

norm_df = read_norm_data('data/target_fitness_fn_ranges.csv')
# norm_df = pd.DataFrame(columns=['function', 'min', 'max'])


def activate_population(inputs, genomes):
    out_list = [g(inputs[i]) for i,g in enumerate(genomes)]
    for i, out in enumerate(out_list):
        if out_list[i].shape[0] == 1:
            out_list[i] = out_list[i].repeat(3, 1, 1)
    outputs = torch.stack(out_list)
        
    # outputs = outputs.clamp_(0,1)
    return outputs


@torch.no_grad()
def measure_fitness(config, inputs, genomes, genome_targets):
    fit_children = torch.zeros((len(genomes), len(config.objective_functions)), device=config.device, requires_grad=False)
    fc_normed = torch.zeros((len(genomes), len(config.objective_functions)), device=config.device, requires_grad=False)

    if config.dry_run:
        # Random fitness for dry run
        fit_children = torch.rand((len(genomes), len(config.objective_functions)), device=config.device, requires_grad=False)
        return fit_children, fit_children.clone()

    # pbar = tqdm(total=num_batches, desc="Measuring fitness")
    # self.correct_target_count(len(batch_genomes))
    # batch_start_idx = batch_index * config.batch_size

    genomes_key_list = list(genomes.keys())

    population = [genomes[genome_uuid] for genome_uuid in genomes_key_list]
    X = torch.stack([inputs[genome_uuid] for genome_uuid in genomes_key_list])
    Y = torch.vstack([all_targets[genome_targets[genome_uuid]] for genome_uuid in genomes_key_list])
    print(Y.shape)
    batch_imgs = activate_population(X, population)
    for i, fn in enumerate(config.objective_functions):
        # Evaluating based on images
        fitness = fn(batch_imgs, Y)
        if fitness.dim() == 0:
            fitness = fitness.unsqueeze(0)
        if fn not in ff.NO_NORM:
            normed_fitness = norm_tensor(fitness, norm_df, fn.__name__, warn=False)

        if not fn in ff.NO_MEAN:
            fc_normed[:, i] = normed_fitness

        fit_children[:, i] = fitness
        # pbar.update(1)
        
    # Normalizing collected fitness components
    if len(fc_normed) > 0:
        fc_normed = fc_normed
    else:
        fc_normed = fit_children.clone()  # Not normalizing anything

    # Assigning final fitness to genomes
    for i, genome_uuid in enumerate(genomes):
        genomes[genome_uuid].fitness = fc_normed[i].mean().item()
        
    agg_fc_normed   = fc_normed.detach().mean(dim=1)

    return agg_fc_normed


def filter_by_target(run):
    if not os.path.exists(os.path.join(run, "target.txt")):
        return False, None
    if not os.path.exists(os.path.join(run, "inputs.pt")):
        return False, None

    with open(os.path.join(run, "target.txt")) as f:
        target = f.read().strip()
    return (target in targets, target)


def load_cppn(run):
    genome_dir = os.path.join(run, "genomes/")
    genome_files = os.listdir(genome_dir)
    # sort by name
    genome_files.sort()
    if len(genome_files) == 0:
        return (None,None,None,None,None)
    
    # take the half way point
    # genome_file = genome_files[len(genome_files) // 2]
    genome_file = genome_files[len(genome_files) -1]
    g_batch = int(genome_file.split("_")[1].split(".")[0])  
    genome_file = genome_dir + genome_file
    with open(genome_file) as f:
        genomes = json.load(f)
    genome = genomes['genome']
    config = genomes["config"]
    config = MoveConfig.create_from_json(config, MoveConfig, device=device)
    # apply_condition(config, {}, {}, "top-muts", ff.__dict__)
    apply_condition(config, {}, {}, config.experiment_condition, ff.__dict__)

    fix_target_dimensions(config)

         # grayscale image
    if config.color_mode != "L":
            logging.warning("Target image is grayscale, but color_mode is not set to 'L'. Setting color_mode to 'L'")
            config.color_mode = "L"
            
    if config.res_w != config.target.shape[0]:
        config.res_w = config.target.shape[0]
        logging.warning("config.Target image width does not match config.res_w. Setting config.res_w to config.target image width")
    if config.res_h != config.target.shape[1]:
        config.res_h = config.target.shape[1]
        logging.warning("Target image height does not match config.res_h. Setting config.res_h to target image height")



    config.target = config.target.unsqueeze(0) 

    if len(config.target.shape) > 3:
        config.target = config.target.permute(0, 3, 1, 2) # move color channel to front
    else:
        config.target = config.target.unsqueeze(1).repeat(1,3,1,1) # add color channel
        
    if config.target.shape[-2] < 32 or config.target.shape[-1] < 32:
            config.target = Resize((32,32), antialias=True)(config.target)
    
    config.target = torch.clamp(config.target, 0, 1)

    for i, fn in enumerate(config.objective_functions):
            if isinstance(fn, str):
                config.objective_functions[i] = getattr(ff, fn)
                
    cppn = CPPN.create_from_json(genome, config)
    if os.path.exists(run + "inputs.pt"):
        inputs = torch.load(run + "inputs.pt", map_location=device)
    else:
        inputs = initialize_inputs_from_config(config)
        inputs = add_fourier_features(
            inputs,
            config.n_fourier_features,
            config.fourier_feature_scale,
            dims=2,
            include_original=True,
            )
    print(config.target.shape)
    return cppn, config, inputs, g_batch, config.target



# data = [target, condition, run,genome_uuid,  num_genome_cxs, num_genome_nodes, genome_start_fitness, num_mutation_iters, mutation_step, num_sgd_steps, final_fitness]
require_data=False
def build_df(path_to_conds, conds):
    data = []
    genomes = OrderedDict()
    all_inputs = OrderedDict()
    all_configs = OrderedDict()
    all_targets = OrderedDict()
    if conds == "*":
        conds = os.listdir(path_to_conds)
    for cond in tqdm(conds):
        cond_path = path_to_conds + cond + "/"
        runs = os.listdir(cond_path)
        runs = [cond_path + run + "/" for run in runs]
        for run in tqdm(runs):
            print(run)
            has_data, target = filter_by_target(run)
            if not has_data and require_data:
                continue
            genome,config,inputs,genome_batch,target_img = load_cppn(run)
            if genome is None:
                continue
            genome_uuid = uuid.uuid4()
            genomes[genome_uuid] = genome
            all_inputs[genome_uuid] = inputs
            all_configs[genome_uuid] = config
            if target not in all_targets:
                print(target_img.shape)
                all_targets[target] = target_img
            this_data = {
                "target": target,
                "condition": config.experiment_condition,
                "run": run,
                "genome_uuid": genome_uuid,
                "num_genome_cxs": len(genome.connections),
                "num_genome_nodes": len(genome.nodes),
                "genome_start_fitness": torch.nan,
                "change_in_fitness": torch.nan,
                "num_mutation_iters": 0,
                "mutation_step": 0,
                "num_sgd_steps": 0,
                "final_fitness": 0,
                "genome_batch": genome_batch
            }
            data.append(this_data)
    df = pd.DataFrame(data, columns=["target", "condition", "run", "genome_uuid", "genome_batch", "num_genome_cxs", "num_genome_nodes", "genome_start_fitness", "change_in_fitness", "num_mutation_iters", "mutation_step", "num_sgd_steps", "final_fitness"])

    return df, genomes, all_inputs, all_configs, all_targets

df, genomes, all_inputs, all_configs, all_targets = build_df(path_to_conds, conditions)

use_config = all_configs[list(all_configs.keys())[0]] # TODO check this

fns = [
    ff.mse,
    ff.psnr,
    ff.lpips,
    ff.dists,
    ff.style,
    ff.vif,
    ff.dss,
    ff.ssim,
    ff.msssim,
    ff.haarpsi,
]

use_config.objective_functions = fns

pilmode = "RGB" if len(use_config.color_mode) == 3 else "L"
# all_targets = {t:
                # torch.tensor(resize_image(iio.imread(t, pilmode=pilmode), use_config.target_resize, device), dtype=torch.float32, device=use_config.device)
            #    for t in targets
            #    }

for t in all_targets:
#     # if all_targets[t].max() > 1.0:
#         # all_targets[t] = all_targets[t].to(torch.float32) / 255.0
#     # repeat L to RGB
#     if len(all_targets[t].shape) == 3:
#         all_targets[t] = all_targets[t].repeat(0, 3, 1, 1)
    # all_targets[t] = all_targets[t].permute(2, 0, 1)
    all_targets[t] = all_targets[t][:,:3,:,:]

iter_conds = [0, 1, 5, 10]
# iter_conds = [0, 1]
sgd_step_conds = [0, 1, 10, 100, 1000]
# sgd_step_conds = [0,1]
these_genomes = copy.deepcopy(genomes)
genome_targets  = {genome_uuid:df.loc[df['genome_uuid'] == genome_uuid, 'target'].item() for genome_uuid in these_genomes}
_ = measure_fitness(use_config, all_inputs, these_genomes, genome_targets)
for genome_uuid in these_genomes:
    df.loc[df['genome_uuid'] == genome_uuid, 'genome_start_fitness'] = these_genomes[genome_uuid].fitness


# TODO NOT SURE IF THE ABOVE WORKED, NEED TO TEST WITH:
# python -c "import torch; print(torch.load('/users/j/s/jsdean/scratch/move/results/telo-24-growth/conditions/MOVE-SGD-Fast-Grow/run_3750977589289292271/agg_fitness_by_batch.pt').max(dim=0)[0][69])"
    
print(df.filter(['target', 'condition', 'run', 'genome_batch', 'genome_start_fitness']))
# show an image
genome_key = list(these_genomes.keys())[3]
genome = these_genomes[genome_key]
genome_batch = int(df.loc[df['genome_uuid'] == genome_key, 'genome_batch'].values[0])
X = torch.stack([all_inputs[genome_uuid] for genome_uuid in [genome_key]])
img = activate_population(X, [genome])
img = img[0].permute(1, 2, 0).detach().cpu().numpy()
# print the run path
run_path = df.loc[df['genome_uuid'] == genome_key, 'run'].values[0]
print(run_path)
plt.imsave("analysis/test_img.png", img)
print("genome_key", genome_key)
print("target", genome_targets[genome_key])
print("start fitness", genome.fitness)
# print("saved", torch.load(run_path + 'agg_fitness_by_batch.pt').max(dim=0)[0][genome_batch])


# save the df as a csv 
df.to_csv("analysis/partial_fitness.csv", index=False)

import seaborn as sns
# plot the start_fitness by condition as bar
sns.barplot(data=df, x="condition", y="genome_start_fitness")

plt.savefig("analysis/partial_fitness.png")

exit()


use_config.no_param_mutations =True
use_config.topology_mutation_iters=10
use_config.bloat_prune_rate= 0.0
use_config.connection_prune_threshold= 0.0
use_config.min_pruned= 0

for iter_c in iter_conds:
    for sgd_c in sgd_step_conds:
        genomes_copy = copy.deepcopy(genomes)
        # make iter_c mutations
        for genome_uuid in genomes_copy:
            for _ in range(iter_c):
                genomes_copy[genome_uuid].mutate(use_config)
        # apply sgd_c steps
        population = [(_,_,genomes_copy[genome_uuid]) for genome_uuid in genomes_copy]
        X = torch.stack([all_inputs[genome_uuid] for genome_uuid in genomes_copy])
        Y = torch.stack([all_targets[genome_targets[genome_uuid]] for genome_uuid in genomes_copy])
        record = torch.ones(sgd_c, device=use_config.device)
        use_config.sgd_steps = sgd_c
        use_config.sgd_early_stop = False
        sgd_weights(population, None, X, Y, use_config.objective_functions, norm_df, use_config, False, record, unequal_shape=True)
        final_fitnesses = measure_fitness(use_config, all_inputs, genomes_copy, genome_targets)
        # TODO USE RECORD (BUT IT'S MEAN OF POP NOW)
        new_data = []
        for i, genome_uuid in enumerate(genomes_copy):
            new_data.append({
                "target": genome_targets[genome_uuid],
                "condition": use_config.experiment_condition,
                "run": df.loc[df['genome_uuid'] == genome_uuid, 'run'].values[0],
                "genome_uuid": genome_uuid,
                "num_genome_cxs": len(genomes_copy[genome_uuid].connections),
                "num_genome_nodes": len(genomes_copy[genome_uuid].nodes),
                "genome_start_fitness": df.loc[df['genome_uuid'] == genome_uuid, 'genome_start_fitness'].values[0],
                "change_in_fitness": final_fitnesses[i].item() - df.loc[df['genome_uuid'] == genome_uuid, 'genome_start_fitness'].values[0],
                "num_mutation_iters": iter_c,
                "mutation_step": 0,
                "num_sgd_steps": sgd_c,
                "final_fitness": final_fitnesses[i].item(),
                "genome_batch": df.loc[df['genome_uuid'] == genome_uuid, 'genome_batch'].values[0]
            })
        new_df = pd.DataFrame(new_data, columns=["target", "condition", "run", "genome_uuid", "genome_batch", "num_genome_cxs", "num_genome_nodes", "genome_start_fitness", "change_in_fitness","num_mutation_iters", "mutation_step", "num_sgd_steps", "final_fitness"])
        df = pd.concat([df, new_df], ignore_index=True)
                    
        
print(df)
print(df.run.tolist())


df.to_csv("analysis/n_mutations_by_sgd.csv", index=False)

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("analysis/n_mutations_by_sgd.csv")


sns.barplot(data=df, x="num_mutation_iters", y="final_fitness", hue="num_sgd_steps")
plt.savefig("analysis/n_mutations_by_sgd.png")
plt.show()