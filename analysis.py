# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from norm import read_norm_data, norm

import argparse

#%%
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, default="output/default")
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

# population_size = 70

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
    
norm(df, norm_df)

mean_df = df.groupby(["gen", "condition", "run"])["fitness"].mean().reset_index()


# %%
# Adjust the Steady State condition to match the Without Steady State condition
use_df = mean_df.copy()

if "without-steady-state" in df["condition"].unique():
    population_size = df[(df["condition"] == "without-steady-state") & (df['run'] ==  df['run'].min())]["cell"].unique().shape[0]
    # split conditions to separate dfs
    df_with = use_df[use_df["condition"] == "with-steady-state"]
    df_without = use_df[use_df["condition"] == "without-steady-state"]
    # keep every 70th gen of df_with
    df_with = df_with[df_with["gen"] % population_size == 0].reset_index()
    # rename gens to match df_without
    df_with["gen"] = df_with["gen"] // population_size
    use_df = pd.concat([df_with, df_without])
# %%
ax = sns.lineplot(x="gen", y="fitness", hue="condition", data=use_df)
plt.savefig(os.path.join(args.path, "plots", "avg_fits_over_time.png"))
plt.show()


