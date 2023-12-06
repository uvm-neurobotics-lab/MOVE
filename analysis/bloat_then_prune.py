import matplotlib.pyplot as plt
import multiprocessing as mp

plt.style.use("seaborn-v0_8-whitegrid")
import torch
import sys 
sys.path.append('..')
import os
import pandas as pd
import seaborn as sns

from cppn import CPPN

from cppn.graph_util import get_matching_connections, genetic_difference
from norm import read_norm_data
from sgd_weights import sgd_weights
from analysis_util import *


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


def mutate_bloat(cppn, config, n=100, node_cx_ratio=0.5):
    config = config.clone()
    cppn = cppn.clone(new_id=False, config=config)
    
    config.topology_mutation_iters = n
    config.prob_remove_node = 0.0
    config.prob_disable_connection = 0.0
    config.prob_add_connection = 1.0
    config.prob_mutate_activation = 0.0
    config.prob_mutate_bias=0.0
    config.prob_add_node = config.prob_add_connection*node_cx_ratio
    config.single_structural_mutation = False
    
    cppn.mutate(config, pbar=True)
    
    return cppn

def bloat_connections(cppn, config, n):
    bloated_cppn = cppn.clone(new_id=False, config=config)
    bloated_cppn.to('cuda')
    for _ in range(n):
        bloated_cppn.add_connection(config)
    return bloated_cppn


    
def plot_genetic_difference_after_bloat(start, config, n=100):
    genetic_differences = []
    for _ in range(n):
        cppn_bloated = mutate_bloat(start, config)
        genetic_differences.append(genetic_difference(start, cppn_bloated))
    sns.displot(genetic_differences, kde=True)
    plt.show()
    
    


def config_for_bloat_prune(config):
    config_cloned = config.clone()   
    config_cloned.sgd_l2_reg = .1
    config_cloned.prune_threshold = 0.01
    config_cloned.min_pruned = 0
    config_cloned.sgd_steps = 1000
    config_cloned.sgd_learning_rate = 0.01 
    return config_cloned

def test_bloat_prune(initial_cppn, target, config, args, inputs):
    # if args.plot:
        # plot_genetic_difference_after_bloat(initial_cppn, config)
        
    num_added_connections = len(initial_cppn.connections)
    # cppn_bloated = mutate_bloat(initial_cppn, config, num_added_connections, 0.0) # extra bloat
    
    cppn_bloated = bloat_connections(initial_cppn, config, num_added_connections)
        
    cppn_bloated_after_sgd_prune = cppn_bloated.clone(new_id=False, config=config)
    
    config = config_for_bloat_prune(config)
    
    cppn_bloated_after_sgd_prune.sgd_lr = config.sgd_learning_rate # override learned lr
    
    sgd_weights([cppn_bloated_after_sgd_prune], None, inputs, target, config.objective_functions, norm, config, early_stop=10)
    n_pruned,n_pruned_nodes = cppn_bloated_after_sgd_prune.prune(config)
    
    print(f"Pruned {n_pruned} connections")
    print(f"Pruned {n_pruned_nodes} nodes")
    
    print("Min weight:", min([abs(cx.weight.item()) for cx in cppn_bloated_after_sgd_prune.connections.values()]))
    
    cppn_bloated.to('cuda')
    cppn_bloated_after_sgd_prune.to('cuda')
    initial_cppn.to('cuda')
    inputs = inputs.to('cuda')
        
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
  

def show_prune_effect(initial_cppn, config, inputs, target, norm):
    imgs,titles = [],[]
    config = config_for_bloat_prune(config)
    initial_cppn.sgd_lr = config.sgd_learning_rate # override learned lr
    initial_cppn.to('cuda')
    inputs = inputs.to('cuda')
    sgd_weights([initial_cppn], None, inputs, target, config.objective_functions, norm, config, early_stop=10)
    imgs.append(get_image(initial_cppn, inputs))
    titles.append(f"initial : {len(initial_cppn.connections)} cxs")
    
    num_added_connections = len(initial_cppn.connections)
    cppn_bloated = bloat_connections(initial_cppn, config, num_added_connections)
    cppn_bloated.to('cuda')

    # for min_pruned in [0, 8, 16, 32, 128, 1024, 2048, 4096, 8192, 16384, 32768]:
    for min_pruned_pt in [0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 0.85, 0.90, 0.95, 0.99]:
        min_pruned = int(min_pruned_pt * len(cppn_bloated.connections))
        if min_pruned > len(cppn_bloated.connections):
            continue
        config.min_pruned = min_pruned
        config.prune_threshold=0.0
        test_cppn = cppn_bloated.clone(new_id=False, config=config)
        test_cppn.to('cuda')
        test_cppn.sgd_lr = config.sgd_learning_rate # override learned lr
        sgd_weights([test_cppn], None, inputs, target, config.objective_functions, norm, config, early_stop=10)
        n_pruned,n_pruned_nodes = test_cppn.prune(config)
        print(f"Pruned {n_pruned} connections")
        print(f"Pruned {n_pruned_nodes} nodes")
        imgs.append(get_image(test_cppn, inputs))
        titles.append(f"n_pruned={min_pruned} : {len(test_cppn.connections)} cxs")
    
    show_image_grid(imgs, titles)

    
if __name__ == "__main__":
    mp.set_start_method('spawn')
    
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
    
    
    norm = read_norm_data(os.path.join("../",config.norm_df_path), config.target_name)
   
    
    set_max_node_id(batches)
    
    batch_index = len(batches)//2
    test_batch = list(batches.keys())[batch_index]
    print("Test batch:", test_batch)

    initial_cppn = batches[test_batch]

    show_prune_effect(initial_cppn, config, inputs, target, norm)

    results = []
    n_trials = 30
    
    batch = list(batches.keys())[list(batches.values()).index(initial_cppn)]


    pool = mp.Pool(mp.cpu_count()//2)
    results = pool.starmap(test_bloat_prune, [(initial_cppn.clone(config, new_id=False), target.clone(), config, args, inputs.clone()) for _ in range(n_trials)])

    # for i in tqdm.tqdm(range(n_trials)):
        # results.append(test_bloat_prune(initial_cppn, target, config, args, inputs))
        
    print("Average ratio:", sum(results)/len(results))
    print("Min ratio:", min(results))
    print("Max ratio:", max(results))
    
    torch.save(torch.tensor(results), os.path.join(args.path, "bloat_prune_ratio.pt"))
    
    plt.style.use("default")
    sns.displot(results, kde=True, bins=10)
    plt.savefig(os.path.join(args.path, f"bloat_prune_ratio_batch_{batch:04d}.png"))
    
    plt.show()
    
    
