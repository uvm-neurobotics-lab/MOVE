
"""Metrics for evaluating the performance of MOVE."""
# %%
import multiprocessing as mp
from scipy.stats import ranksums
from tqdm import tqdm, trange
import pandas as pd
import warnings
import argparse
import os
import torch
import json
from scikits import bootstrap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42 
plt.rcParams['font.size'] = 15
import seaborn as sns
sns.set(font_scale=1.15)
sns.set_style("ticks")
import textwrap
sns.color_palette("colorblind")

import dask.dataframe as dd

from norm import norm_from_saved, read_norm_data

try:
    from .collect_data import collect_data
except ImportError:
    from collect_data import collect_data
    
# globals:
current_experiment_dir = ""
cache_dir = ""
save_dir = ""

warnings.filterwarnings("ignore")  # for bootstrap CI
# plt.style.use('seaborn-paper')

# BASELINE_PATH = "map_elites_voting/results/baseline-equal/"
BASELINE_PATH = "results/baseline/"
# BASELINE_PATH = "/users/j/s/jsdean/scratch/masters/map_elites_voting/results/baseline-equal/"
# BASELINE_PATH = "results/baseline-all-images/"
# BASELINE_PATH = "results/ff-sgd-baseline"


pretty_labels = {
    "baseline": "Single-objective hillclimber",
    "baseline-aggregate": "All-objective hillclimber",
    "0-jumps-70-cells-1ea": "No (1 fn/cell)",
    "yes-100-cells-1ea": "Yes (1 fn/cell)",
    "yes-100-cells-1ea-no-mp": "Yes (no multiple replacements, 1 fn/cell)",
    "yes-100-cells-1ea-mp": "Yes (with multiple replacements; 1 fn/cell)",

    "0-jumps-70-cells-03ea": "No (3 fns/cell)",
    "yes-100-cells-03ea": "Yes (3 fns/cell)",
    "yes-100-cells-03ea-no-mp": "Yes (no multiple replacements, 3 fns/cell)",
    "yes-100-cells-03ea-mp": "Yes (with multiple replacements; 3 fns/cell)",

    "0-jumps-70-cells-05ea": "No (5 fns/cell)",
    "yes-100-cells-05ea": "Yes (5 fns/cell)",
    "yes-100-cells-05ea-no-mp": "Yes (no multiple replacements, 5 fns/cell)",
    "yes-100-cells-05ea-mp": "Yes (with multiple replacements; 5 fns/cell)",

    "0-jumps-70-cells-07ea": "No (7 fns/cell)",
    "yes-100-cells-07ea": "Yes (7 fns/cell)",
    "yes-100-cells-07ea-no-mp": "Yes (no multiple replacements, 7 fns/cell)",
    "yes-100-cells-07ea-mp": "Yes (with multiple replacements; 7 fns/cell)",

    "0-jumps-70-cells-09ea": "No (9 fns/cell)",
    "yes-100-cells-09ea": "Yes (9 fns/cell)",
    "yes-100-cells-09ea-no-mp": "Yes (no multiple replacements, 9 fns/cell)",
    "yes-100-cells-09ea-mp": "Yes (with multiple replacements; 9 fns/cell)",

    "0-jumps-70-cells-11ea": "No (11 fns/cell)",
    "yes-100-cells-11ea": "Yes (11 fns/cell)",
    "yes-100-cells-11ea-no-mp": "Yes (no multiple replacements, 11 fns/cell)",
    "yes-100-cells-11ea-mp": "Yes (with multiple replacements; 11 fns/cell)",

    "worst-50-03ea": "Worst 50 cells",
    "best-50-03ea": "Best 50 cells",
    "01-fns-per-cell": "1 function per cell",
    "03-fns-per-cell": "3 functions per cell",
    "05-fns-per-cell": "5 functions per cell",
    "07-fns-per-cell": "7 functions per cell",
    "09-fns-per-cell": "9 functions per cell",
    "11-fns-per-cell": "11 functions per cell",
    "fixed (5 fns per cell)": "Fixed (5 fns per cell)",
    "0-jumps-70-cells": "No jumping",
    "yes-100-cells-mp": "Jumping with multiple placements",
    "yes-100-cells-no-mp": "Jumping without multiple placements",
    "use radial distance": "Use radial distance",
    "no radial distance": "No radial distance",
    "25-cells-05ea": "25 cells (5 fns/cell)",
    "50-cells-05ea": "50 cells (5 fns/cell)",
    "100-cells-05ea": "100 cells (5 fns/cell)",
    "50-cells-03ea": "50 cells (3 fns/cell)",
    "50-cells-11ea": "50 cells (11 fns/cell)",
    "50-cells-09ea": "50 cells (9 fns/cell)",
    "50-cells-07ea": "50 cells (7 fns/cell)",
    "use_avg_fit_3": "Average fitness (3 fns/cell)",
    "use_avg_fit_5": "Average fitness (5 fns/cell)",
    "use_avg_fit_7": "Average fitness (7 fns/cell)",
    "use_avg_fit_9": "Average fitness (9 fns/cell)",
    "use_avg_fit_11": "Average fitness (11 fns/cell)",
}

pretty_function_names = {
    "psnr": "PSNR",
    "mse": "MSE",
    "gmsd": "GMSD",
    "lpips": "LPIPS",
    "dists": "DISTS",
    "ssim": "SSIM",
    "style": "Style",
    "fsim": "FSIM",
    "mdsi": "MDSI",
    "haarpsi": "HaarPSI",
    "vif": "VIF",
    "dss": "DSS",
    "vsi": "VSI",
    "msssim": "MS-SSIM",
}


# %%

def save_stats(df, col, additional_name=None, directory=None):
    if directory is None:
        directory = save_dir
    # summary stats
    stats = {}
    for cond in df["condition"].unique():
        stats[cond] = {}
        stats[cond]["std"] = df[df["condition"] == cond][col].std()
        stats[cond]["mean"] = df[df["condition"] == cond][col].mean()
        stats[cond]["quantiles_975"] = df[df["condition"]
                                          == cond][col].quantile(0.975)
        stats[cond]["quantiles_025"] = df[df["condition"]
                                          == cond][col].quantile(0.025)
        stats[cond]["max"] = df[df["condition"] == cond][col].max()
        stats[cond]["min"] = df[df["condition"] == cond][col].min()
        stats[cond]["count"] = df[df["condition"] == cond][col].count()
                
        stats[cond]["CI_95_high"] = stats[cond]["mean"] + 1.96 * stats[cond]["std"] / np.sqrt(stats[cond]["count"])
        stats[cond]["CI_95_low"] = stats[cond]["mean"] - 1.96 * stats[cond]["std"] / np.sqrt(stats[cond]["count"])
        stats[cond]["CI_95"] = stats[cond]["CI_95_high"] - stats[cond]["CI_95_low"]
        stats[cond]["std_error"] = stats[cond]["std"] / np.sqrt(stats[cond]["count"])
            
    total_std = df[col].std()

    # calculate p-values
    pvals_ranksums = {}
    dvals = {}

    for cond1 in df["condition"].unique():
        for cond2 in df["condition"].unique():
            if cond1 != cond2:
                if df[df["condition"] == cond1][col].empty or df[df["condition"] == cond2][col].empty:
                    continue
                dvals[(cond1, cond2)] = {"d": (stats[cond1]["mean"] - stats[cond2]["mean"])/(
                    np.sqrt((stats[cond1]["std"]**2 + stats[cond2]["std"]**2)/2))}
                t_w = ranksums(df[df["condition"] == cond1]
                                [col], df[df["condition"] == cond2][col])
                pvals_ranksums[(cond1, cond2)] = {
                    "w": t_w.statistic, "p": t_w.pvalue}
    

    with open(os.path.join(directory, f"{col}{('_'+additional_name) if additional_name is not None else ''}_stats.txt"), "w") as f:
        f.write(f"Summary stats for {col}:\n")
        for cond, stats in stats.items():
            f.write(f"{cond}:\n")
            for k, v in stats.items():
                f.write(f"\t{k}: {v}\n")

        f.write(f"Total std: {total_std}\n")

        f.write(f"\n\nComparison p-values (Wilcoxon rank-sum):\n")
        for k, v in pvals_ranksums.items():
            f.write(f"{k}: {v}")
            if v["p"] < 0.05:
                f.write(" *")
            f.write("\n")
        f.write(f"\n\nComparison effect size (Cohen's d):\n")
        for k, v in dvals.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\n\n")
        f.write("All data:\n")
        f.write(f"{df.to_string()}\n")


def save_fig(name):
    os.makedirs(os.path.join(save_dir, "plots", "pdfs"), exist_ok=True)
    
    sns.set(font_scale=1.15)
    sns.set_style("ticks")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plots", f"{name}.png"))
    plt.savefig(os.path.join(save_dir, "plots", "pdfs", f"{name}.pdf"))

    plt.show()
    plt.close()


def cache_df(df, name):
    print(f"Saving {name} to cache at {cache_dir}/{name}.pkl")
    df.to_pickle(os.path.join(cache_dir, f"{name}.pkl"))

def try_get_cached_df(name, directory=None):
    if directory is None:
        directory = cache_dir
    path = os.path.join(directory, f"{name}.pkl")
    if os.path.exists(path):
        print(f"Loading {name} from cache at {path}")
        return pd.read_pickle(path)
    return None

def validate_baseline(df_path):    
    if not os.path.exists(df_path):
        print(f"Missing df at {df_path}")
        return False
    print("\t\tFound, validating...")
    df = pd.read_pickle(df_path)
    
    cond_path = os.path.join(df_path, "..", "conditions")
    conds = df["condition"].unique()
    for dir in os.listdir(cond_path):
        if dir not in conds:
            print(f"Missing condition {dir} in df")
            return False
        if len(os.listdir(os.path.join(cond_path, dir))) != len(df[df["condition"] == dir]["run"].unique()):
            print(f"Missing runs for condition {dir}",
                  f"{len(os.listdir(os.path.join(cond_path, dir)))} vs {len(df[df['condition'] == dir]['run'].unique())}")
            return False
    print("\t\tBaseline data validated")
    return True


def add_baseline_to_df(df):
    if df is not None and 'baseline' in df['condition'].unique():
        print("Baseline already in df")
        return df
    
    df_path = os.path.join(BASELINE_PATH, "fits.pkl")
    print("\tAdding baseline data to df...")
    fits_df = pd.DataFrame(columns=["fitness", "gen", "run", "condition"])
    if os.path.exists(df_path):
        fits_df = pd.read_pickle(df_path)
    else:
        print(f"No baseline data at {df_path}")
        print("Create it first by running baseline.py and analysis/compile_baseline.py")
        exit(1)
    
    if False:
        print("\nDEV WARN: *NOT* Validating baseline data\n");
        if not validate_baseline(df_path):
            print("\t\tBaseline data invalid, exiting...")
            exit(1)
        

    baseline_df = fits_df.copy()

    # the function is the condition for baselines
    baseline_df.loc[:, "function"] = baseline_df.loc[:, "condition"]

    # replace all the conditions with "baseline" except if the condition includes "baseline" already
    # leaves 'baseline' and 'baseline-aggregate' as is
    baseline_df.loc[:, "condition"] = baseline_df.loc[:, "condition"].apply(
        lambda x: "baseline" if "baseline" not in x else x)
    
    print("\t\tConditions:", baseline_df["condition"].unique())
    if df is None:
        return baseline_df
    else:
        # only include the baseline data up to the max generation of the df
        max_df_gen = df["gen"].max()
        baseline_df = baseline_df[baseline_df["gen"] <= max_df_gen]
        # add the baseline data to the df
        df = pd.concat([df, baseline_df],
                       ignore_index=True).reset_index(drop=True)
        
        return df

def norm_fitness(df, norm_df):
    """
    Normalizes fitness values in df using the min and max fitness values in norm_df.\
    Modifies df in-place

    Args:
        df (pandas.DataFrame): DataFrame containing fitness values to normalize (in column "fitness")
        
        norm_df (pandas.DataFrame): DataFrame containing min and max fitness values for each function\
        (in columns "min_fitness_mean" and "max_fitness_mean", respectively)

    """
    print("Applying fitness normalization...")

    # cache the mins and maxes for speed:
    normed_fns = list(norm_df['function'].unique())
    mins =  {fn: norm_df[norm_df["function"] == fn]["min_fitness_mean"].values[0] for fn in normed_fns}
    maxes = {fn: norm_df[norm_df["function"] == fn]["max_fitness_mean"].values[0] for fn in normed_fns} 
    
    already_normed = ['baseline-aggregate'] # skip these
    
    def norm(data):
        fn = data['function']
        if not fn in normed_fns:
            print(f"Function {fn} not in normed_fns")
            return pd.NA
        if fn in already_normed:
            return data['fitness'] # don't norm
        
        return (data['fitness'] - mins[fn]) / (maxes[fn] - mins[fn])
    
    df['normed_fitness'] = df.apply(norm, axis=1)


def measure_by_function(df):
    df = replacements_to_removals(df.reset_index(drop=True))
    # df columns: ,condition,cell,run,gen,removals,cell-from,cell-to
    fns = ["psnr","gmsd","mse","lpips","dists","ssim","style","fsim","mdsi","haarpsi","vif","dss","vsi","msssim",]
    for fn in fns:
        df[fn] = df["cell-from"].apply(lambda x: fn in x.split("_"))
        # find the average number of removals per gen
    
    # plot
    mean_df = df.groupby(['condition', 'run', 'gen', 'function', 'target'])['removals'].agg('mean').reset_index()
    sum_df = mean_df.groupby(['gen', 'function'])['removals'].agg('sum').reset_index()
    sns.scatterplot(x="gen", y="removals", data=df_fn, label=fn, hue="function")

    save_fig("survivors_per_function_scatter")
    
    # as bar:
    # sum across gens and plot bar
    all_df_summed = []
    for fn in fns:
        df_fn = df[df[fn]]
        df_fn = df_fn.groupby(['condition', 'run'])['removals'].agg('sum').reset_index(name="removals_for_fn")
        df_fn["fn"] = fn
        df_fn['condition'] = df_fn['condition'].apply(lambda x: x+"_"+fn)
    
        all_df_summed.append(df_fn)
    all_df_summed = pd.concat(all_df_summed)
    
    print(all_df_summed)
    save_stats(all_df_summed, "removals_for_fn")
    sns.barplot(x="fn", y="removals_for_fn", data=all_df_summed)
    save_fig("survivors_per_function_bar_all")
    # remove 'fn' column
    all_df_summed = all_df_summed.drop(columns=['fn'])
    ax=sns.barplot(x="condition", y="removals_for_fn", data=all_df_summed)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    save_fig("survivors_per_function_bar_by_cond")
    

def add_same_function_info(df):
    if "shared_fns" in df.columns:
        print("shared_fns already in df")
        return
    def count_shared(row):
        return len(set(row['cell-from'].split('_')).intersection(set(row['cell-to'].split('_'))))
    df['shared_fns'] = df.apply(lambda row: count_shared(row), axis=1)


def replacements_to_removals(df):
    # each cell-to can only be removed once per gen
    # take only the last replacement for a cell in a gen
    # any cell-to that is replaced at least once in a gen is a 1
    # any cell-to that is not replaced in a gen is a 0
    # if a cell is replaced multiple times in a gen, only the last replacement should count
    r_df = df[df["replacements"] > 0]
    r_df = r_df.drop_duplicates(
        subset=['condition', 'run', 'gen', 'cell-to'], keep='last')

    # copy back to df, where missing are 0
    df.loc[:, "removals"] = 0
    df.loc[r_df.index, "removals"] = r_df["replacements"]

    return df


def plot_replacements_same_fn(df, run_id, cond=None, removals=True):
    if cond is not None:
        df = df[df["condition"] == cond]
    if removals:
        df = replacements_to_removals(df)
        varname = "removals"
    else:
        varname = "replacements"

    add_same_function_info(df)
    # percent of replacements that are from cells that share a with the to-cell
    percent_same_fn = df.copy()
    percent_same_fn['same_fn_at_least_one'] = percent_same_fn["shared_fns"] > 0
    percent_same_fn_overall = percent_same_fn.copy()
    percent_same_fn = percent_same_fn[percent_same_fn[varname] > 0]
    percent_same_fn = percent_same_fn.groupby(['condition', 'run'])[
        'same_fn_at_least_one'].mean().reset_index(name='percent_same_fn')
    percent_same_fn_overall = percent_same_fn_overall.groupby(['condition', 'run'])[
        'same_fn_at_least_one'].mean().reset_index(name='percent_same_fn')
    percent_same_fn_overall['condition'] = percent_same_fn_overall['condition'] + "+overall"
    percent_same_fn['condition'] = percent_same_fn['condition'] + f"+{varname}"
    result = pd.concat([percent_same_fn_overall, percent_same_fn])
    save_stats(result, "percent_same_fn", varname)
    print(f"\nPercent of {varname} that are from cells with shared functions\n")
    print(result)
    

    total_mean_shared_fns = df.groupby(['condition', 'run'])[
        'shared_fns'].mean().reset_index(name='mean_shared_fns')
    total_mean_shared_fns['condition'] = total_mean_shared_fns['condition'] + "+total"
    df = df[df[varname] > 0]
    removal_mean_shared_fns = df.groupby(['condition', 'run'])[
        'shared_fns'].mean().reset_index(name='mean_shared_fns')
    removal_mean_shared_fns['condition'] = removal_mean_shared_fns['condition'] + \
        f"+{varname}"
    result = pd.concat([total_mean_shared_fns, removal_mean_shared_fns])
    save_stats(result, "mean_shared_fns", varname)

    ax = sns.barplot(x="condition", y="mean_shared_fns", data=result)
    ax.set_title(f"Mean shared functions in {varname}")
    ax.set_ylabel("Mean shared functions")
    # ax.set_xlabel("Condition")
    ax.set(xlabel=None)
    save_fig(f"mean_shared_fns_{varname}")

    ax = sns.barplot(x="condition", y="percent_same_fn", data=percent_same_fn)
    ax.set_title(
        f"Percent of {varname} that are from cells with shared functions")
    ax.set_ylabel("Percent")
    ax.set(xlabel=None)

    save_fig(f"percent_same_fn_{varname}")


def plot_replacements_heatmap(df, run_id, cond=None, removals=True, plot_by_function=False):
    print("plot_replacements_heatmap")
    if cond is not None:
        df = df[df["condition"] == cond]
    if removals:
        df = replacements_to_removals(df)
    if run_id != -1 and run_id is not None:
        df = df[df["run"] == run_id]

    varname = "replacements" if not removals else "removals"
    
    if not plot_by_function:
        print("plotting by cell")
        for c in df["condition"].unique():
            fig, ax = plt.subplots(figsize=(30, 30))
            this_df = df[df["condition"] == c]
            # plot by cell
            # plot by cell
            cells = this_df["cell-to"].unique().tolist()
            matrix = np.zeros((len(cells), len(cells)))

            matrix = this_df.pivot_table(
                index="cell-from", columns="cell-to", values=varname, aggfunc=np.sum).fillna(0).values
            labels = this_df.pivot_table(
                index="cell-from", columns="cell-to", values=varname, aggfunc=np.sum).index.tolist()

            # to int
            # replace nan with 0
            matrix = np.nan_to_num(matrix)
            matrix = matrix.astype(int)

            # matrix = matrix / matrix.sum(axis=1)[:, None]
            g = sns.heatmap(matrix, ax=ax, cmap="Blues", xticklabels=labels,
                            yticklabels=labels, annot=False, fmt="d")
            # plt.title(f"{'Removals' if removals else 'Wins'} heatmap for {c}" + f"run {run_id}" if run_id != -1 else "all runs")
            # set axis titles
            ax.set(xlabel="Cell to", ylabel="Cell from")

            save_fig(f"{'removals' if removals else 'wins'}_heatmap_{c}_{run_id}")
            
    if plot_by_function:
        print("plotting by function")
        
        functions = set()
        for cell in df["cell-to"].unique():
            functions.update(cell.split("_"))
        functions = list(functions)
        functions = np.array(functions)
        
        
        # works:
        this_df = df[df['removals'] > 0].copy()
        this_df[['cell-to','cell-from']] = this_df[['cell-to','cell-from']].apply(lambda x: x.str.split('_'))
        this_df = this_df.explode('cell-to')
        this_df = this_df.explode('cell-from')
        this_df.rename(columns={'cell-to': 'function-to', 'cell-from': 'function-from'}, inplace=True)
        this_df = this_df.groupby(['function-to','function-from', 'run','condition', 'gen']).agg({varname: 'sum'})
        this_df = this_df.reset_index()

        # save as csv
        this_df.to_csv(os.path.join(save_dir, f"{'removals' if removals else 'wins'}_heatmap_all_{run_id}_gen_all.csv"))
        return # handled in another file
        

def plot_replacements(df, removals=True):
    title = "Removals over time" if removals else "Wins over time"

    varname = "removals" if removals else "replacements"
    
    if removals:
        c_df = replacements_to_removals(c_df)
        c_df = c_df.rename(columns={"removals": "Removals"})
        title = "Removals"
    else:
        c_df = c_df.rename(columns={"replacements": "Wins"})
        title = "Wins"

    df_grp = c_df.groupby(['condition', 'run', 'gen'])[
        title].sum().reset_index()
    df_grp = df_grp.rename(columns={"gen": "Generation"})

    sns.lineplot(data=df_grp, x="Generation", y=title, label=cond, hue="condition")

    save_fig("removals_t" if removals else "replacements_t")
    
    # plot sum as bar
    df_grp = df.groupby(['condition', 'run']).agg({varname: "sum"}).reset_index()
    save_stats(df_grp, varname, "total_replacements")
    df_grp['condition'] = df_grp['condition'].apply(lambda x: pretty_labels[x])
    df_grp = df_grp.rename(columns={varname: "Total replacements", "condition": "Condition"})
    
    ax = sns.barplot(data=df_grp, x="Condition", y="Total replacements")
    ax.set_xticklabels([1, 3, 5, 7, 9, 11])
    ax.set(xlabel="Functions per cell")
    save_fig("replacements_total")
    

def plot_ratio_of_same_cell_replacements(df, removals=True):
    title = "Ratio of same cell removals" if removals else "Ratio of same cell wins"

    # plt.title(title)
    varname = "removals" if removals else "replacements"
    if removals:
        df = replacements_to_removals(df)

    df["same_cell"] = df["cell-to"] == df["cell-from"]
    
    df = df[df[varname] > 0] # only care about removals that actually happened
    print(df.head())
    # calculate ratio of same cell removals
    df_sum = df.groupby(['condition', 'run', 'same_cell'], as_index=False).agg({varname: "sum"})
    ratio = df_sum.groupby(['condition', 'run'], as_index=False).apply(lambda x: x[x['same_cell'] == True][varname].values[0] / x[varname].sum())
    # ratio = ratio.reset_index()
    # ratio = ratio.rename(columns={"": "ratio-of-same-cell"})
    ratio.columns.values[len(ratio.columns.values)-1] = "ratio-of-same-cell"
    save_stats(ratio, "ratio-of-same-cell", varname)

    df_grp = df.groupby(['condition', 'run', 'same_cell'])[
        varname].sum().reset_index()
    df_grp = df_grp.rename(columns={varname: "Count"})
    df_grp['condition'] = df_grp['condition'].apply(lambda x: pretty_labels[x])

    ax = sns.barplot(data=df_grp, x="condition", y="Count", hue="same_cell")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20,
                       ha="right")
    ax.set(xlabel=None)

    save_fig(f"same_cell_{varname}")
    


def plot_one_fit(df, fn, cond=None, cell_agg=None, save_dir_=None, compare_to_baseline=False):
    plt.title(f"{fn.upper()} over time")
    df_fn = df[df["function"] == fn]
    # df_fn = pd.concat([df_fn, baseline])
    if cond is not None:
        df_fn = df_fn[df_fn["condition"] == cond]
        df_fn = df_fn.rename(columns={"gen": "Generation"})

        sns.lineplot(data=df_fn, x="Generation", y="Fitness", label=cond)
    else:
        if cell_agg is not None:
            df_fn = df_fn.groupby(['condition', 'run', 'gen'])[
                'fitness'].agg(cell_agg).reset_index()
            df_fn = df_fn.rename(columns={"fitness": cell_agg+" fitness"})
            df_fn = df_fn.rename(columns={"gen": "Generation"})

            ax = sns.lineplot(data=df_fn, x="Generation", y=cell_agg +
                              " fitness", hue="condition", style="condition", dashes=False)
            for t in ax.legend_.texts:
                if t.get_text()  in pretty_labels:
                    t.set_text(pretty_labels[t.get_text()])
        else:
            df_fn = df_fn.rename(columns={"gen": "Generation"})
            ax = sns.lineplot(data=df_fn, x="Generation", y="fitness",
                              hue="condition", style="condition", dashes=False)
            for t in ax.legend_.texts:
                t.set_text(pretty_labels[t.get_text()])

    if save_dir_ is None:
        save_dir_ = save_dir

    save_fig(f"fn_performance/{fn}_t" if cell_agg is None else f"fn_performance/{fn}_{cell_agg}_t")
    
    if compare_to_baseline:
        df = add_baseline_to_df(df)
        last_gen = df['gen'].max()
        df_last_gen = df[df['gen'] == last_gen]
        # max by function 
        df_last_gen_max = df_last_gen.groupby(['condition', 'function', 'run'], as_index=False).agg({'fitness': 'max'})
        # make each function it's own condition
        df_last_gen_max['condition'] = df_last_gen_max.apply(lambda x: x['condition'] + "_" + x['function'], axis=1)
        
            
        save_stats(df_last_gen_max, f"fitness", "max_fn_performance_last_gen")
        if 'max_global_performance' in df_last_gen_max.columns:
            sns.boxplot(data=df_last_gen_max, x="condition", y="max_global_performance", hue="function")
            save_fig(f"fn_performance/_all_last_gen_max")
        
        
        # mean by function 
        df_last_gen_mean = df_last_gen.groupby(['condition', 'function', 'run'], as_index=False).agg({'fitness': 'mean'})
        # make each function it's own condition
        df_last_gen_mean['condition'] = df_last_gen_mean.apply(lambda x: x['condition'] + "_" + x['function'], axis=1)
        
        save_stats(df_last_gen_mean, f"fitness", "mean_fn_performance_last_gen")
        if 'mean_global_performance' in df_last_gen_mean.columns:
            sns.boxplot(data=df_last_gen_mean, x="condition", y="fitness", hue="function")
            save_fig(f"fn_performance/_all_last_gen_mean")
        
        

def plot_fit(df, fn, cond=None, agg='max'):
    if cond is not None:
        df = df[df["condition"] == cond]
        # else show from all conditions
    df_fn = df[df["function"] == fn]
    # plot max at each gen
    if agg == 'max':
        df_fn = df_fn.groupby(['gen', 'condition', 'run'])[
            'fitness'].max().reset_index()
    elif agg == 'mean':
        df_fn = df_fn.groupby(['gen', 'condition', 'run'])[
            'fitness'].mean().reset_index()
    # normalize 0-1
    norm_from_saved(df_fn, BASELINE_PATH)
    # df_fn["fitness"] = (df_fn["fitness"] - df_fn["fitness"].min()) / \
        # (df_fn["fitness"].max() - df_fn["fitness"].min())
        
    df_fn = df_fn.rename(columns={"gen": "Generation"})

    ax = sns.lineplot(data=df_fn, x="Generation", y="fitness",
                      label=fn if cond is None else f"{cond} - {fn}")
    for t in ax.legend_.texts:
        t.set_text(pretty_labels[t.get_text()])


def print_glob_performance(df, stats):
    print(f'{"Global performance":-^65}')
    b_max, b_min = get_baseline_max_min_fits()
    print(f'{"baseline":^65}')
    for (kmax, vmax), (kmin, vmin) in zip(b_max.items(), b_min.items()):
        print(f"\t{kmax:20} min: {vmin:0.3f},\t max: {vmax:0.3f}")
    print(f'\n{"experiment":^65}')
    print(stats)
    print(f'{"":-^65}')


def plot_performance_bar(df, stat):
    stat_to_label = {"max": "Max global", "agg": "Overall", "mean": "Mean"}
    plt.close()
    if stat == "agg":
        df = df[df["condition"] != "baseline"]
        print(df.columns)
        print(df.condition.unique())
        print(df)
        # copy "agg_global_fitness_baseline" to "agg_global_performance" only for "baseline-aggregate" condition
        df.loc[df["condition"] == "baseline-aggregate", "agg_global_performance"] = df.loc[df["condition"]
                                                                                           == "baseline-aggregate", "agg_global_fitness_baseline"]

    # remove "agg_global_fitness_baseline"
    df = df[df["condition"] != "agg_global_fitness_baseline"]
    # plt.title(stat_to_label[stat] + " performance by condition")
    plt.ylabel(stat_to_label[stat] + " performance")

    last_gen = df["gen"].max()
    last_gen_df = df[df["gen"] == last_gen]
    means = last_gen_df.groupby(['condition', 'run'])[
        stat+"_global_performance"].mean().reset_index()
    print(means)
    print(means.columns)
    print(means["condition"].unique())

    save_stats(means, stat+"_global_performance")

    means["condition"] = means["condition"].apply(lambda x: pretty_labels[x])
    ax = sns.barplot(data=means, x="condition", y=stat+"_global_performance")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=10,
                       ha="right")
    ax.set(xlabel=None)

    plt.tight_layout()
    save_fig(f"{stat}")


def plot_glob_performance_over_time(df, cond=None, use_cache=None):
    plt.close()
    name = "glob_t" if cond is None else f"glob_t_{cond}"
    
    if use_cache is not None:
        cached_df = try_get_cached_df(name, use_cache)
        if cached_df is not None:
            this_df = cached_df
            
    if cond is not None:
        df = df[df["condition"] == cond]
      
    this_df = df.rename(
        columns={"max_global_performance": "Max global performance"})
    this_df = this_df.groupby(['gen', 'run'])['Max global performance'].mean().reset_index()  # mean/max/min should be same
    this_df = this_df.rename(columns={"gen": "Generation"})
    ax = sns.lineplot(data=this_df, x="Generation",
                        y="Max global performance", hue="condition")
    if ax.legend_ is not None:
        for t in ax.legend_.texts:
            t.set_text(pretty_labels[t.get_text()])
                
    cache_df(this_df, name)
    save_fig(name)


def plot_agg_performance_over_time(df, cond=None, use_cache=None):
    plt.close()
    name = "agg_glob_t" if cond is None else f"agg_glob_t_{cond}"
    
    # try to load from cache
    plot_df = None
    if use_cache is not None:
        plot_df = try_get_cached_df(name, use_cache)
    if plot_df is None:
        plot_df = df.copy()
        
    if cond is not None:
        plot_df = plot_df[plot_df["condition"] == cond]

    plot_df = plot_df[plot_df['condition'] != 'baseline'] # no way to get aggregated performance for baseline (could do post hoc?)
    
    conds = plot_df["condition"].unique()
   
    if '11-fns-per-cell' in conds:
        # sort by number of functions per cell
        conds = ['01-fns-per-cell', '03-fns-per-cell', '05-fns-per-cell', '07-fns-per-cell', '09-fns-per-cell', '11-fns-per-cell', 'baseline-aggregate']

    plot_df['condition'] = pd.Categorical(plot_df['condition'], conds)


    plot_df.loc[plot_df['condition'] == 'baseline-aggregate', 'cell'] = 'no-cell' # no cell for baseline-aggregate
    
    # average fitness within cells
    plot_df = plot_df.groupby(['gen', 'run', 'condition', 'cell'])['normed_fitness'].mean().reset_index()
    
    # find max within gen
    plot_df = plot_df.groupby(['gen', 'run', 'condition'])['normed_fitness'].max().reset_index()
    
    # rename columns
    plot_df = plot_df.rename(
        columns={"normed_fitness": "Overall global performance", "gen": "Generation"})

    # plot
    ax = sns.lineplot(data=plot_df, x="Generation",
                        y="Overall global performance", hue="condition")
    
    # fix legend labels
    if ax.legend_ is not None:
        for t in ax.legend_.texts:
            try:
                t.set_text(pretty_labels[t.get_text()])

            except KeyError:
                print("no pretty label for", t.get_text())
                continue
            
    ax.legend() # remove seaborn title from legend
    
    # save
    cache_df(plot_df, name)
    save_fig(name)
                
                    
    if '0-jumps-70-cells-1ea' in df['condition'].unique():
        # combine all no conditions to one 
        df_copy = df.copy()
        df_copy.loc[df_copy['condition'] == '0-jumps-70-cells-1ea', 'condition'] = '0-jumps-70-cells-'
        df_copy.loc[df_copy['condition'] == '0-jumps-70-cells-03ea', 'condition'] = '0-jumps-70-cells-'
        df_copy.loc[df_copy['condition'] == '0-jumps-70-cells-05ea', 'condition'] = '0-jumps-70-cells-'
        df_copy.loc[df_copy['condition'] == '0-jumps-70-cells-07ea', 'condition'] = '0-jumps-70-cells-'
        df_copy.loc[df_copy['condition'] == '0-jumps-70-cells-09ea', 'condition'] = '0-jumps-70-cells-'
        df_copy.loc[df_copy['condition'] == '0-jumps-70-cells-11ea', 'condition'] = '0-jumps-70-cells-'
        
        df_copy.loc[df_copy['condition'] == 'yes-100-cells-1ea-mp', 'condition'] = 'yes-100-cells-mp'
        df_copy.loc[df_copy['condition'] == 'yes-100-cells-03ea-mp', 'condition'] = 'yes-100-cells-mp'
        df_copy.loc[df_copy['condition'] == 'yes-100-cells-05ea-mp', 'condition'] = 'yes-100-cells-mp'
        df_copy.loc[df_copy['condition'] == 'yes-100-cells-07ea-mp', 'condition'] = 'yes-100-cells-mp'
        df_copy.loc[df_copy['condition'] == 'yes-100-cells-09ea-mp', 'condition'] = 'yes-100-cells-mp'
        df_copy.loc[df_copy['condition'] == 'yes-100-cells-11ea-mp', 'condition'] = 'yes-100-cells-mp'
        
        df_copy.loc[df_copy['condition'] == 'yes-100-cells-1ea-no-mp', 'condition'] = 'yes-100-cells-no-mp'
        df_copy.loc[df_copy['condition'] == 'yes-100-cells-03ea-no-mp', 'condition'] = 'yes-100-cells-no-mp'
        df_copy.loc[df_copy['condition'] == 'yes-100-cells-05ea-no-mp', 'condition'] = 'yes-100-cells-no-mp'
        df_copy.loc[df_copy['condition'] == 'yes-100-cells-07ea-no-mp', 'condition'] = 'yes-100-cells-no-mp'
        df_copy.loc[df_copy['condition'] == 'yes-100-cells-09ea-no-mp', 'condition'] = 'yes-100-cells-no-mp'
        df_copy.loc[df_copy['condition'] == 'yes-100-cells-11ea-no-mp', 'condition'] = 'yes-100-cells-no-mp'
        all = []
        
        # conds = df_copy["condition"].unique()
        conds = ['yes-100-cells-mp', 'yes-100-cells-no-mp', '0-jumps-70-cells-', 'baseline-aggregate']
        for cond in conds:
            this_df = df_copy[df_copy["condition"] == cond]
            if cond == "baseline":
                continue  # no way to get aggregated performance for baseline
            elif cond == "baseline-aggregate":
                this_df = this_df.groupby(['gen', 'run'])[
                    'fn_global_performance'].mean().reset_index()
                this_df.loc[:, "agg_global_performance"] = this_df["fn_global_performance"].apply(
                    lambda x: np.mean(x))
            else:
                this_df = this_df.groupby(['gen', 'run'])['agg_global_performance'].mean(
                ).reset_index()  # mean/max/min should be same
            this_df = this_df.rename(
                columns={"agg_global_performance": "Overall global performance"})
            this_df = this_df.rename(columns={"gen": "Generation"})

            ax = sns.lineplot(data=this_df, x="Generation",
                              y="Overall global performance", label=cond)
            if ax.legend_ is not None:
                for t in ax.legend_.texts:
                    t.set_text(pretty_labels[t.get_text()])
            all.append(this_df)
        
        save_fig("agg_glob_t_combined")
        all_df = pd.concat(all)
        last_gen = all_df[all_df["gen"] == all_df["gen"].max()]
        save_stats(last_gen, "Overall global performance")


def plot_average_performance_over_time(df, cond=None):
    plt.close()

    if cond is not None:
        df = df[df["condition"] == cond]
        sns.lineplot(data=df, x="gen", y="mean_global_performance", label=cond)
    else:
        # plt.title("Average performance over time")
        # df = add_baseline_to_df(df)
        for cond in df["condition"].unique():
            this_df = df[df["condition"] == cond]
            this_df = this_df.groupby(['gen', 'run'])['mean_global_performance'].mean(
            ).reset_index()  # mean/max/min should be same
            this_df = this_df.rename(
                columns={"mean_global_performance": "Mean global performance"})

            ax = sns.lineplot(data=this_df, x="gen",
                              y="Mean global performance", label=cond)
            if ax.legend_ is not None:
                for t in ax.legend_.texts:
                    t.set_text(pretty_labels[t.get_text()])

    # plot_baseline_performance(show=False, split_fns=False)
    save_fig("average_t")


def plot_unique_cells_in_genealogy(parents_df, ids_df):
    df = pd.DataFrame(columns=["run", "condition", "unique_cells", "id"])
    # ids: ,condition,cell,run,id,gen
    # parents: ,condition,cell,run,parent0,parent1,gen
    # fits: ,condition,cell,run,function,gen,fitness
    last_gen = parents_df["gen"].max()
    for condition in ids_df["condition"].unique():
        for run in ids_df["run"].unique():
            run_df_ids = ids_df.loc[(ids_df["condition"] == condition) & (
                ids_df["run"] == run)]
            run_df_parents = parents_df.loc[(
                parents_df["condition"] == condition) & (parents_df["run"] == run)]
            for cell in tqdm(run_df_ids["cell"].unique(), desc=f"{condition} | {run}/{len(ids_df['run'].unique())} genealogy", leave=False, mininterval=0, miniters=1):
                total_count = 0
                final_id_in_cell = run_df_ids.loc[(run_df_ids["cell"] == cell) & (
                    run_df_ids["gen"] == last_gen), "id"].values[0]
                df = df.append({"run": run, "condition": condition,
                               "unique_cells": 1, "id": final_id_in_cell}, ignore_index=True)
                current_gen = last_gen
                current_cell = cell
                genealogy = [None] * int(last_gen + 1)
                genealogy[int(current_gen)] = current_cell
                # print("cell", cell, "final_id_in_cell", final_id_in_cell)
                while current_gen > 1:
                    # NOTE: ASSUMES 1 PARENT
                    # get parent id
                    # faster way to look up parent id:
                    parent_id = run_df_parents.loc[(run_df_parents["cell"] == current_cell) & (
                        run_df_parents["gen"] == current_gen), "parent0"].values[0]
                    if parent_id == -1:
                        break  # parent is -1, so we are at the root
                    # get cell of parent
                    cell_values = run_df_ids.loc[(run_df_ids["id"] == parent_id) & (
                        run_df_ids["gen"] == current_gen - 1), "cell"].values
                    if len(cell_values) == 0:
                        # not born in this gen
                        current_gen -= 1
                        continue
                    current_cell = cell_values[0]
                    total_count += 1
                    # is this cell already in genealogy?
                    if not current_cell in genealogy:
                        # add 1 to unique cells
                        # df.loc[(df["id"] == final_id_in_cell) & (df["run"] == run) & (df["condition"] == condition), "unique_cells"] += 1
                        genealogy[int(current_gen)] = current_cell
                    current_gen -= 1
                genealogy = [x for x in genealogy if x is not None]
                df.loc[(df["id"] == final_id_in_cell) & (df["run"] == run) & (
                    df["condition"] == condition), "unique_cells"] = len(set(genealogy))
                df.loc[(df["id"] == final_id_in_cell) & (df["run"] == run) & (
                    df["condition"] == condition), "total_jumps"] = total_count

    new_df = df.groupby(['condition', 'run'])[
        'unique_cells'].mean().reset_index(name='unique_cells_mean')
    total_df = df.groupby(['condition', 'run'])[
        'total_jumps'].mean().reset_index(name='total_jumps_mean')
    save_stats(total_df, 'total_jumps_mean')
    save_stats(new_df, 'unique_cells_mean')

    new_df = new_df.rename(
        columns={"unique_cells_mean": "Mean unique cells in genealogy"})
    # replace condition names with pretty names
    new_df["condition"] = new_df["condition"].apply(lambda x: pretty_labels[x])
    ax = sns.barplot(data=new_df, x="condition",
                     y="Mean unique cells in genealogy")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20,
                       ha="right")
    ax.set(xlabel=None)

    save_fig("unique_cells_in_genealogy")
    print("Done.")


def plot_unique_ids_over_time(df):
    plt.close()
    unique_count = df.groupby(['gen', 'run', 'condition'])['id'].nunique().reset_index()
    unique_count.rename(columns={'id': 'unique_ids'}, inplace=True)
    unique_count = unique_count.rename(
        columns={"unique_ids": "Unique genomes"})

    ax = sns.lineplot(data=unique_count, x="gen",
                        y="Unique genomes", hue="condition")
    for t in ax.legend_.texts:
        t.set_text(pretty_labels[t.get_text()])

    cache_df(unique_count, "ids_t")
    save_fig("ids_t")


def plot_final_unique_ids(df):
    plt.close()
    # plt.title("Final unique genomes")
    max_gen = df["gen"].max()
    df = df[df["gen"] == max_gen]
    # get counts
    unique_count = df.groupby(['run', 'condition'])[
        'id'].nunique().reset_index()
    unique_count.rename(columns={'id': 'Unique genomes'}, inplace=True)
    unique_count['condition'] = unique_count['condition'].apply(
        lambda x: pretty_labels[x])
    save_stats(unique_count, "Unique genomes")
    ax = sns.barplot(data=unique_count, x="condition", y="Unique genomes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20,
                       ha="right")
    ax.set(xlabel=None)

    cache_df(unique_count, "ids_final")
    save_fig("ids_final")


def count_removals_per_cell(df, filename):
    # if os.path.exists(filename):
    #     cell_removals_df = pd.read_csv(filename)
    # else:
    cell_removals_df = pd.DataFrame(
        columns=["experiment", "condition", "run", "cell", "removals-to", "removals-from"])
    if not "removals" in df.columns:
        df = replacements_to_removals(df)

    for cond in df["condition"].unique():
        cond_df = df[df["condition"] == cond]
        for run in cond_df["run"].unique():
            run_df = cond_df[cond_df["run"] == run]
            for cell in set(run_df["cell-from"].unique().tolist() + run_df["cell-to"].unique().tolist()):
                removals_to_cell = run_df[run_df["cell-to"]
                                          == cell]["removals"].sum()
                removals_from_cell = run_df[run_df["cell-from"]
                                            == cell]["removals"].sum()
                cell_removals_df = cell_removals_df.append({"experiment": current_experiment_dir, "condition": cond, "run": run,
                                                           "cell": cell, "removals-to": removals_to_cell, "removals-from": removals_from_cell}, ignore_index=True)
    cell_removals_df.to_csv(filename, index=False)
    return cell_removals_df


def max_fits_by_function(df):
    last_gen = df['gen'].max()
    df_lg = df[df['gen'] == last_gen]
    cache_df(df_lg, "last_gen_fits")
    return 
    
    
def print_removals_per_cell(cell_removals_df):
    print(cell_removals_df.groupby("cell").mean().to_string())


def save_removals_per_cell(cell_removals_df):
    mean_removals = cell_removals_df.groupby("cell").mean()
    mean_removals = mean_removals.sort_values(
        by=["removals-from"], ascending=False)

    # save best half
    mean_removals_best = mean_removals.iloc[:int(len(mean_removals)/2)]
    with open(os.path.join(save_dir, "best_half.txt"), "w") as f:
        for cell in mean_removals_best.index:
            f.write(f"{cell}\n")
    # save worst half
    mean_removals_worst = mean_removals.iloc[int(len(mean_removals)/2):]
    with open(os.path.join(save_dir, "worst_half.txt"), "w") as f:
        for cell in mean_removals_worst.index:
            f.write(f"{cell}\n")


num_workers = 1

def main():
    parser = argparse.ArgumentParser(
        description='Show the performance of the algorithm.')
    parser.add_argument('experiment_dir', action='store',
                        help='Saved results json file.')
    parser.add_argument('-ow', '--overwrite', action='store_true',
                        help='Overwrite existing results (runs).', default=False)
    parser.add_argument('--load', action='store_true',
                        help='Load existing results (dfs).', default=False)
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Debug mode.', default=False)
    parser.add_argument("-w", "--workers", type=int, default=mp.cpu_count()-1)
    parser.add_argument("-f", "--fns", type=str, nargs="+",
                        default=["fits", "replacements", "ids", "parents"], help="Functions to plot.")
    parser.add_argument('--plot_output', action='store',
                        help='Where to save the plots. Defaults to experiment dir.', default=None)
    parser.add_argument("-c", "--cond_dir", type=str, required=False, help="Directory of one condition (defaults to all).")
    parser.add_argument("-r", "--run_dir", type=str, required=False, help="Directory of one run (defaults to all.")
    parser.add_argument("-b", "--baseline_dir", type=str, required=False, help="Directory of baseline runs.")

    args = parser.parse_args()
    if args.workers > 0:
        global num_workers
        num_workers = args.workers
    if args.baseline_dir is not None:
        global BASELINE_PATH
        BASELINE_PATH = args.baseline_dir
    
    global current_experiment_dir, save_dir, cache_dir

    current_experiment_dir = args.experiment_dir
    assert os.path.exists(
        args.experiment_dir), f"Directory {args.experiment_dir} does not exist"
    
    assert not (
        args.overwrite and args.load), "Can't overwrite and load at the same time."
    
    if args.plot_output is None:
        args.plot_output = args.experiment_dir

    save_dir = args.plot_output

    cache_dir = os.path.join(current_experiment_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # norm_df = pd.read_pickle(os.path.join(BASELINE_PATH, "norm.pkl"))
    norm_df = read_norm_data("data/target_fitness_fn_ranges.csv", "none")


    dfs = collect_data(
        dir=args.experiment_dir,
        overwrite_run_dfs=args.overwrite,
        one_run=args.run_dir,
        one_cond=args.cond_dir,
        load_overall_dfs=args.load,
        num_workers=args.workers,
        fns=args.fns
    )
    print("Done collecting data, got the following dfs:", [
          f"{k}: {v is not None}" for k, v in dfs.items()])

    for df in dfs.values():
        if df is not None:
            # reset index to make it easier to work with
            df.reset_index(inplace=True)

    print("Done collecting data")
    print("Plotting...")

    # fits
    if "fits" in args.fns:
        dir = os.path.join(current_experiment_dir, "plots", "pdfs", "fn_performance")
        os.makedirs(dir,exist_ok=True)
        dfs["fits"] = dfs["fits"][dfs["fits"]
                                  ["fitness"] < torch.inf]  # remove inf
        dfs["fits"] = dfs["fits"][dfs["fits"]
                                  ["fitness"] > -torch.inf]  # remove -inf

        fit_df = dfs.get("fits", None)
        
        fit_df = add_baseline_to_df(fit_df)

        norm_fitness(fit_df, norm_df)
        
        plot_agg_performance_over_time(fit_df)
        
        max_fits_by_function(fit_df)
        
        # plot_performance_bar(fit_df, "agg")
        # plot_performance_bar(fit_df, "mean")
        # plot_performance_bar(fit_df, "max")

        # plot_baseline_performance(show=True, split_fns=True)

        # print_glob_performance(fit_df, stats)
        print()

        # plot_average_performance_over_time(fit_df)

    # parents
    if "parents" in args.fns:
        assert "ids" in args.fns, "can't do parents without ids"
        # assert "replacements" in args.fns, "can't do parents without replacements for now"

        # condition = dfs["ids"]["condition"].unique()[0]
        # run = 0
        # genealogy = get_genealogies(dfs["ids"], dfs["parents"], dfs["replacements"], args.debug)
        plot_unique_cells_in_genealogy(dfs["parents"], dfs["ids"])
        # id = dfs["ids"][(dfs["ids"]["condition"] == condition) & (dfs["ids"]["run"] == run)]["id"].iloc[-1]
        # print_genealogy(genealogy, condition, run, id)
        # condition = dfs["ids"]["condition"].unique()[-1]
        # id = dfs["ids"][(dfs["ids"]["condition"] == condition) & (dfs["ids"]["run"] == run)]["id"].iloc[-1]
        # print_genealogy(genealogy, condition, run, id)

    # ids
    if "ids" in args.fns:
        # plot_unique_ids_over_time(dfs["ids"])
        plot_final_unique_ids(dfs["ids"])

    # replacements
    if "replacements" in args.fns:
        df = dfs["replacements"]
        # plot_replacements(df)
        # plot_replacements_heatmap(df, run_id=None, plot_by_function=True, removals=True)
        plot_ratio_of_same_cell_replacements(df, removals=True)
        # measure_by_function(df)
        # r_per_cell_df = count_removals_per_cell(df, os.path.join("map_elites_voting", "results", "overall", "removals_per_cell.csv"))
        # r_per_cell_df = count_removals_per_cell(
            # df, os.path.join(save_dir, "removals_per_cell.csv"))
        # print_removals_per_cell(r_per_cell_df)
        save_removals_per_cell(r_per_cell_df)

        plot_replacements_same_fn(df, run_id=-1, removals=True)
        # plot_replacements_heatmap(df, run_id=0)
        # plot_replacements_heatmap(df, run_id=0, removals=False)

    # plot global performance for each function
    if "fits" in args.fns:
        if not os.path.exists(dir):
            os.makedirs(dir)
        for fn in dfs["fits"]["function"].unique():
            plot_one_fit(dfs["fits"], fn=fn, cell_agg="max",  save_dir_=dir, compare_to_baseline=True)
            plot_one_fit(dfs["fits"], fn=fn, cell_agg="mean", save_dir_=dir, compare_to_baseline=True)

# %%


if __name__ == "__main__":
    main()
