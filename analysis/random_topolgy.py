# %%
from plot import *


results_path = "../../results/sgd-steps"


save_path = os.path.join(results_path, "plots")
if os.path.exists(save_path):
    os.makedirs(os.path.join(save_path), exist_ok=True)

metrics = [
    "evals_by_batch",
    "normed_fitness_by_batch",
    # "pruned_cxs",
    # "offspring_by_batch",
    # "fitness_by_batch",
    # "lr_by_batch",
    "cx_by_batch",
    "nodes_by_batch",

]


condition_filter = ["100_steps_"]

results = read_tensor_results(results_path, metrics, max_runs=None, condition_filter=condition_filter)

# %%
results['condition_path'] = results['condition'].copy()

# %%
avg_nodes_at_max_evals = []
avg_cx_at_max_evals = []

def get_max_evals(group):
    return group.loc[group.evals_by_batch.idxmax()]

results_max_evals = results.groupby(["condition", "run", "target"]).apply(get_max_evals).reset_index(drop=True)

print(results_max_evals)    


# %%
avg_nodes_at_max_evals = results_max_evals.groupby("target").nodes_by_batch.mean()
avg_cx_at_max_evals = results_max_evals.groupby("target").cx_by_batch.mean()
# convert to dict
avg_nodes_at_max_evals = avg_nodes_at_max_evals.to_dict()
avg_cx_at_max_evals = avg_cx_at_max_evals.to_dict()
print(f"avg_nodes_at_max_evals: {avg_nodes_at_max_evals}")
print(f"avg_cx_at_max_evals: {avg_cx_at_max_evals}")

# %%
import sys
sys.path.append("../")
from analysis_util import *
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


cppns, all_inputs, configs, targets = get_cppns(results, os.path.join(results_path, "conditions"), device)


# %%
random_topology_results = {}

# %%
N_TRIALS = 2


# %%
from norm import read_norm_data
from sgd_weights import sgd_weights

from cppn import CPPN
from tqdm import trange
from move_config import MoveConfig
from cppn.visualize import visualize_network
import copy 



for target_name in avg_nodes_at_max_evals.keys():
    inputs = torch.tensor(all_inputs[target_name][condition_filter[0]]).to(device)
    config = configs[target_name][list(configs[target_name].keys())[0]] # TODO assumes only one condition
    norm = read_norm_data(os.path.join("../", config.norm_df_path), config.target_name)

    if isinstance(config.target, str):
        target = load_target_from_config(config)
    else:
        target = config.target
    
    if config.output_activation is not None and len(config.output_activation) == 0:
        config.output_activation = None
    
    n_hidden = int(avg_nodes_at_max_evals[target_name])
    n_cx = int(avg_cx_at_max_evals[target_name])
    
    these_cppns = []
    for trial in trange(N_TRIALS):
        config_copy = copy.deepcopy(config)
        if False:
            cppn = CPPN(config_copy)
            
            
            tries = 0
            while len(cppn.hidden_nodes) != n_hidden:
                if len(cppn.hidden_nodes) > n_hidden:
                    cppn.remove_node(config)
                if len(cppn.hidden_nodes) < n_hidden:
                    cppn.add_node(config)
                
                if tries > 100:
                    print("breaking")
                    break
                
                cppn.update_layers()
            
            print("nodes:", len(cppn.hidden_nodes))

            
            for _ in range(20):
                cppn.mutate_activations(config.prob_mutate_activation, config)
            
            print("cxs:", len(cppn.enabled_connections))
            tries = 0
            while len(cppn.enabled_connections) != n_cx:
                if len(cppn.enabled_connections) > n_cx:
                    cppn.disable_connection()
                    cppn.prune(config)
                else:
                    cppn.add_connection(config)
                cppn.update_layers()
                cppn.disable_invalid_connections(config)
                tries += 1
                if tries > 10000:
                    print("breaking")
                    break
                
                print("after cxs:", len(cppn.enabled_connections))
        else:
            total_connections = config.num_inputs * config.hidden_nodes_at_start[0]
            for n_hd in config.hidden_nodes_at_start[1:]:
                total_connections += n_hd * n_hd
            total_connections += config.hidden_nodes_at_start[-1] * config.num_outputs
            config_copy.init_connection_probability = n_cx / total_connections
            config_copy.init_connection_probability_fourier = n_cx / total_connections
            cppn = CPPN(config_copy)
        
        these_cppns.append(cppn)
    
    # config_copy.sgd_steps = 100_000
    config_copy.sgd_steps = 100
    config_copy.sgd_early_stop = 10
    
    target_losses = []
    for i, cppn in enumerate(these_cppns):
        plt.title("Before")
        plt.imshow(cppn(inputs, channel_first=False).cpu().detach().numpy(),cmap="gray")
        plt.show()
        losses = torch.zeros(config_copy.sgd_steps)
        
        sgd_weights([cppn], None, inputs, target, config_copy.objective_functions, norm, config_copy, config_copy.sgd_early_stop, losses)
        
        target_losses.append(losses)
        plt.title("After")
        plt.imshow(cppn(inputs, channel_first=False).cpu().detach().numpy(),cmap="gray")
        plt.show()
    target_losses = torch.stack(target_losses)
    random_topology_results[target_name] = target_losses
        

# %%
print(random_topology_results)
random_top_df = pd.DataFrame(columns=["target", "trial", "loss", "step"])

random_top_results = []
for target_name, losses in random_topology_results.items():
    for i, loss in enumerate(losses):
        l = loss[-1]
        l = 1.0 - l
        random_top_results.append({"target": target_name, "trial": i, "normed_fitness_by_batch": l.item(), "step": len(losses)-1})
            
random_top_df = pd.DataFrame(random_top_results)
random_top_df['condition'] = "random_top"

            

# compare results to random topology

import seaborn as sns

move_results = results_max_evals.drop(columns=["condition_path", "condition"])
move_results['condition'] = "move"
print(move_results)
final_df = pd.concat([move_results, random_top_df])


final_df.to_csv(os.path.join(save_path, "move_vs_random_top_fitness_at_end.csv"))

# %%

sns.barplot(data=final_df, x="target", y="normed_fitness_by_batch", hue="condition")


plt.show()



