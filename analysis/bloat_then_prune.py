from cv2 import add
import matplotlib.pyplot as plt
import numpy as np
import tqdm

plt.style.use("seaborn-v0_8-whitegrid")
import torch
import sys 
sys.path.append('..')
import json
import os
import pandas as pd
import seaborn as sns
import imageio.v2 as iio

from cppn import CPPN
from cppn.cppn import Node
from move_config import MoveConfig, resize_target
from cppn.graph_util import get_disjoint_connections, get_excess_connections, get_matching_connections, genetic_difference
from norm import read_norm_data
from sgd_weights import sgd_weights

def load_genome(path)->tuple[CPPN, MoveConfig]:
    with open(path, "r") as f:
        genome = json.load(f)
    if "config" in genome:
        # TODO REMOVE:
        genome["config"] = json.loads(genome["config"])
        if not "version" in genome["config"]:
            genome["config"]["version"] = "0.0.0"
        # END REMOVE
        config = MoveConfig.create_from_json(genome["config"], MoveConfig)
        genome = genome["genome"]
    else:
        config_path = os.path.join(os.path.dirname(path), "../config.json")
        config = MoveConfig.create_from_json(config_path, MoveConfig)
        
    cppn = CPPN.create_from_json(genome, config=config)
    cppn.to(config.device)
    
    return (cppn, config)


def get_cppns_by_batch(path, modulo=1):
    batches = {}
    path = os.path.join(path, "genomes")
    sorted_list = sorted(os.listdir(path), key=lambda x: int(x.split("_")[1].split(".")[0]))
    pbar = tqdm.tqdm(total=len(sorted_list))
    for i, genome_path in enumerate(sorted_list):
        if i % modulo != 0:
            continue
        if genome_path.endswith(".json"):
            cppn, config = load_genome(os.path.join(path, genome_path))
            # print(f"{genome_path}: {len(cppn.nodes)} nodes, {len(cppn.connections)} connections")
            gen = genome_path.split("_")[1].split(".")[0]
            # this_cppn = cppn.clone(new_id=False, config=config)
            batches[int(gen)] = cppn
            assert isinstance(cppn, CPPN)
            assert isinstance(config, MoveConfig)
            for node in cppn.nodes.values():
                assert isinstance(node, Node)
            pbar.update(1)
    pbar.close()
    print("\nLoaded", len(batches), "genomes")
    return batches, config


def plot_cxs_over_time(batches, show=True):
    cxs_by_gen = {k:len(v.connections) for k,v in batches.items()}
    df = pd.DataFrame(cxs_by_gen.items(), columns=["gen", "cxs"])
    df = df.sort_values(by="gen")
    df.plot.line(x="gen", y="cxs", title="Connections over time")
    if show:
        plt.show()
    
    
def plot_nodes_over_time(batches, show=True):
    nodes_by_gen = {k:len(v.nodes) for k,v in batches.items()}
    df = pd.DataFrame(nodes_by_gen.items(), columns=["gen", "nodes"])
    df = df.sort_values(by="gen")
    df.plot.line(x="gen", y="nodes", title="Nodes over time")
    if show:
        plt.show()


def bloat(cppn, config, n=100, node_cx_ratio=0.5):
    config = config.clone()
    cppn = cppn.clone(new_id=False, config=config)
    
    config.topology_mutation_iters = n
    config.prob_remove_node = 0.0
    config.prob_disable_connection = 0.0
    config.prob_add_connection = 1.0
    config.prob_mutate_activation = 0.0
    config.prob_mutate_bias=0.0
    config.prob_add_node = 1.0*node_cx_ratio
    config.single_structural_mutation = False
    
    cppn.mutate(config)
    
    return cppn


def load_inputs(path):
    path = os.path.join(path, "inputs.pt")
    inputs = torch.load(path)
    return inputs
    
    
def get_image(cppn, inputs):
    img = cppn(inputs, channel_first=False)
    return img.detach().cpu().numpy()


def show_image(cppn, inputs):
    plt.style.use("default")
    img = get_image(cppn, inputs)
    plt.imshow(img.detach().cpu().numpy(), cmap="gray")
    plt.show()


def show_image_grid(images, titles=[]):
    num_images = len(images)
    
    # Calculate rows and cols based on the number of images
    cols = int(num_images**0.5)
    rows = cols if cols**2 == num_images else cols + 1

    # Adjust cols if necessary
    if rows * cols < num_images:
        cols += 1

    fig, axs = plt.subplots(rows, cols, figsize=(10,10), squeeze=False) # Ensure axs is always a 2D array
    
    # Hide any extra subplots
    for ax in axs.flat[num_images:]:
        ax.set_visible(False)

    for i in range(len(images)):
        ax = axs[i // cols, i % cols]
        ax.imshow(images[i], cmap="gray")
        if i < len(titles):
            ax.set_title(titles[i])
        ax.axis("off")
        
    plt.tight_layout()
    plt.show()
    
    
def show_img_progression(gens, inputs):
    plt.style.use("default")
    plt.rcParams.update({'font.size': 12})
    sorted_keys = sorted(gens.keys())
    imgs = []
    titles = []
    for k in sorted_keys:
        imgs.append(get_image(gens[k], inputs))
        num_nodes = len(gens[k].nodes)
        num_cxs = len(gens[k].connections)
        titles.append(r"$\bf{"+str(k)+"}$: "+f"{num_nodes};{num_cxs}")
    
    show_image_grid(imgs, titles)
    
        
def set_max_node_id(gens):
    max_id = max([max([int(node.id) for node in cppn.nodes.values()]) for cppn in gens.values()])
    CPPN.current_node_id = max_id + 1
    
    
def plot_genetic_difference_after_bloat(start, config, n=100):
    genetic_differences = []
    for _ in range(n):
        cppn_bloated = bloat(start, config)
        genetic_differences.append(genetic_difference(start, cppn_bloated))
    sns.displot(genetic_differences, kde=True)
    plt.show()
    
    
def load_target(path, config):
    with open(os.path.join(path, "target.txt"), "r") as f:
        target_path = f.read()
    config.target_name = target_path
    target_path = os.path.join("../",target_path) # move out of analysis directory
    pilmode = "RGB" if len(config.color_mode) == 3 else "L"
    target = torch.tensor(iio.imread(target_path, pilmode=pilmode), dtype=torch.float32, device=config.device)
    if target.max()>1.0:
        target /= 255.0
    config.target = target
    target = resize_target(config)
    return config.target


def test_bloat_prune(initial_cppn, target, config, args):
    if args.plot:
        plot_genetic_difference_after_bloat(initial_cppn, config)
        
    num_added_connections = len(initial_cppn.connections) // 4
        
    cppn_bloated = bloat(initial_cppn, config, num_added_connections, 0.0)
    cppn_bloated_after_sgd_prune = cppn_bloated.clone(new_id=False, config=config)
    print(f"initial_cppn: {len(initial_cppn.nodes)} nodes, {len(initial_cppn.connections)} connections")
    print(f"cppn_bloated: {len(cppn_bloated.nodes)} nodes, {len(cppn_bloated.connections)} connections")
    
    print("Genetic difference between cppn_0 and cppn_bloated:")
    print(genetic_difference(initial_cppn, cppn_bloated))
    print(genetic_difference(cppn_bloated, cppn_bloated_after_sgd_prune))
    
    
    norm = read_norm_data(os.path.join("../",config.norm_df_path), config.target_name)
    target = target.unsqueeze(0)
    if len(target.shape) == 3:
        # L -> RGB
        target = target.unsqueeze(1)
        target = target.repeat(1, 3, 1, 1)
    else:
        # RGB -> RGB
        target = target.permute(0,3,1,2)
    
    config.sgd_l2_reg = .1
    config.prune_threshold = 0.01
    config.min_pruned = 0
    config.sgd_steps = 1000
    config.sgd_learning_rate = 0.001
    
    cppn_bloated_after_sgd_prune.sgd_lr = config.sgd_learning_rate # override learned lr
    
    sgd_weights([cppn_bloated_after_sgd_prune], None, inputs, target, config.objective_functions, norm, config, early_stop=10)
    n_pruned,n_pruned_nodes = cppn_bloated_after_sgd_prune.prune(config)
    
    print(f"Pruned {n_pruned} connections")
    print(f"Pruned {n_pruned_nodes} nodes")
    
    print("Min weight:", min([abs(cx.weight.item()) for cx in cppn_bloated_after_sgd_prune.connections.values()]))
    
    if args.plot:
        imgs = [
            target.squeeze(0).permute(1,2,0).cpu(),
            get_image(initial_cppn, inputs),
            get_image(cppn_bloated, inputs),
            get_image(cppn_bloated_after_sgd_prune, inputs)
        ]
        show_image_grid(imgs, ["target", "initial_cppn", "bloated", "bloated_then_sgd_prune"])
        
    
    print()
    print(f"initial: \t\t{len(initial_cppn.nodes)} nodes, {len(initial_cppn.connections)} connections")
    print(f"bloated: \t\t{len(cppn_bloated.nodes)} nodes, {len(cppn_bloated.connections)} connections")
    print(f"bloated_pruned: \t{len(cppn_bloated_after_sgd_prune.nodes)} nodes, {len(cppn_bloated_after_sgd_prune.connections)} connections")
    print()
    
    
    print("Shared cxs initial : bloated", end="\t\t\t\t")
    print(len(get_matching_connections(initial_cppn.connections, cppn_bloated.connections)[0]))
    print("Shared cxs bloated : bloated_pruned", end="\t\t\t")
    print(len(get_matching_connections(cppn_bloated.connections, cppn_bloated_after_sgd_prune.connections)[0]))
    print("Shared cxs initial : bloated_pruned", end="\t\t\t")
    print(len(get_matching_connections(initial_cppn.connections, cppn_bloated_after_sgd_prune.connections)[0]))
    
    added_connections =             set(cppn_bloated.connections.keys()) - set(initial_cppn.connections.keys())
    pruned_connections =            set(cppn_bloated.connections.keys()) - set(cppn_bloated_after_sgd_prune.connections.keys())
    pruned_from_added = added_connections.intersection(pruned_connections)
    print("Pruned from added / total pruned", end="\t\t\t")
    print(len(pruned_from_added),"/",n_pruned)
    
    print("-"*80)
    
    if n_pruned == 0:
        return 0.0
    ratio = len(pruned_from_added) / n_pruned
    return ratio
  


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to folder containing genome files")
    parser.add_argument("--modulo", "-m", type=int, default=1, help="Only load every m-th genome")
    parser.add_argument("--plot", "-p", action="store_true", help="Plot connections over time")
    args = parser.parse_args()
    
    batches, config = get_cppns_by_batch(args.path, args.modulo)
   
    target = load_target(args.path, config)
    
    if args.plot:
        plot_cxs_over_time(batches,show=False)
        plot_nodes_over_time(batches)
    
    
    inputs = load_inputs(args.path).to(config.device)
    
    
    if args.plot:
        show_img_progression(batches, inputs)
    
    
    set_max_node_id(batches)
    
    batch_index = -1
    test_batch = list(batches.keys())[batch_index]
    print("Test batch:", test_batch)

    initial_cppn = batches[test_batch]

    results = []
    n_trials = 100
    
    batch = list(batches.keys())[list(batches.values()).index(initial_cppn)]

    for i in tqdm.tqdm(range(n_trials)):
        results.append(test_bloat_prune(initial_cppn, target, config, args))
        
    print("Average ratio:", sum(results)/len(results))
    print("Min ratio:", min(results))
    print("Max ratio:", max(results))
    
    torch.save(torch.tensor(results), os.path.join(args.path, "bloat_prune_ratio.pt"))
    
    plt.style.use("default")
    sns.displot(results, kde=True, bins=10)
    plt.savefig(os.path.join(args.path, f"bloat_prune_ratio_batch_{batch:04d}.png"))
    
    plt.show()
    
    
