# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fitness.fitness_functions import GENOTYPE_FUNCTIONS
import math
from norm import read_norm_data, norm

import argparse

#%%
parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("-g", "--gen", type=int, default=-1)
parser.add_argument("-t", "--target", type=str, default=None)
parser.add_argument("-c", "--condition", type=str, default=None)
parser.add_argument("--collect", action="store_true")
args = parser.parse_args()
 

# %%
path = args.path

if not os.path.exists(path):
    print("Error: path does not exist")
    exit()

if args.collect:
    os.system(f"python collect_data.py {path} -w {os.cpu_count()} -f fits")
    

norm_df = read_norm_data("data/target_fitness_fn_ranges.csv", "")
conds_dir = os.path.join(args.path, "conditions")

os.makedirs(os.path.join(args.path, "plots"), exist_ok=True)

# %%
fits_path = os.path.join(conds_dir, "fits.pkl")

if os.path.exists(args.path) and not os.path.exists(fits_path):
    print("Error: data has not been parsed yet.")
    print("Collect data by using the --collect flag")
    exit()

df = pd.read_pickle(fits_path)

if args.gen is not None and args.gen >= 0:
    df = df[df["gen"] == args.gen]

if args.condition is not None:
    df = df[df["condition"] == args.condition]
    
if args.target is not None:
    df = df[df["target"] == args.target]
    
# remove functions in GENOTYPE_FUNCTIONS
g_f = GENOTYPE_FUNCTIONS.copy()
g_f = [f.__name__ for f in g_f]
df = df[~df["function"].isin(g_f)]

norm(df, norm_df)


pop_size = df[df['run'] == df['run'].min()]["cell"].count()
mean_df = df.groupby(["gen", "condition", "run"])["fitness"].mean().reset_index()
use_df = mean_df.copy()


# %%
adjust_for_batch_size = False
if adjust_for_batch_size:
    import json
    for run in use_df["run"].unique():
        config = None
        for cond in use_df["condition"].unique():
            if config: 
                break
            run_name = f"run_{run}"
            if run_name in os.listdir(os.path.join(conds_dir, cond)):
                config = json.load(open(os.path.join(conds_dir, cond, run_name, "config.json")))
                config = json.loads(config)
                break
        assert config is not None, f"Could not find config file for run {run}"
        print(type(config))
        batch_size = config["batch_size"]
        initial_batch_size = config["initial_batch_size"]
        # add total_offspring column
        gens_with_init = math.ceil(config["num_cells"]/initial_batch_size)
        offspring_with_init = gens_with_init * initial_batch_size
        
        def gen_to_offspring(gen):
            if gen < gens_with_init:
                return gen * initial_batch_size
            else:
                return offspring_with_init + (gen - gens_with_init) * batch_size
        
        use_df.loc[use_df["run"] == run, "total_offspring"] = use_df.apply(lambda row: gen_to_offspring (row["gen"]), axis=1)

use_df["total_offspring"] = use_df["gen"].apply(lambda x: x * pop_size)

# %%
ax = sns.lineplot(x="total_offspring", y="fitness", hue="condition", data=use_df)
plt.savefig(os.path.join(args.path, "plots", "avg_fits_over_time.png"))
plt.show()


