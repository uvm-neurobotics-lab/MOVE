import argparse
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import multiprocessing as mp
import traceback

def process_tensor(result, name):
    result = result.clone()
    result[result == float('inf')] = float('nan')
    result[result == float('-inf')] = float('nan')
    if name in ['offspring_by_batch']:
        return result
    if name in ['pruned_cxs']:
        if len(result.shape) > 1:
            return result[:,0] # only pruned connections (1 is nodes)
        else:
            return result
    if name in ['lr_by_batch']:
        return result.nanmean(dim=0)
    if name in ["fitness_by_batch", "normed_fitness_by_batch"]:
        #replace inf with nan
        return result.nanmean(dim=(0,1))
    if name in ['evals_by_batch']:
        fwds_sgd = result[:,1] # only n_fwds_incl_sgd
        # cumlative sum:
        fwds_sgd = fwds_sgd.cumsum(dim=0)
        return fwds_sgd
    if name in ['nodes_by_batch', 'cx_by_batch']:
        return result[:,1] # average over pop


def read_tensor_results(results_path, names, max_runs=None):
    cond_dir = os.path.join(results_path, "conditions")
    results = []
    for name in tqdm(names, desc="Loading"):
        try:
            df_index_offset = 0
            for cond in os.listdir(cond_dir):
                cond_path = os.path.join(cond_dir, cond)
                if not os.path.isdir(cond_path):
                            continue
                runs = os.listdir(cond_path)[:max_runs] if max_runs is not None else os.listdir(cond_path)
                for run in runs:
                    pt_path = os.path.join(cond_path, run, f"{name}.pt")
                    if not os.path.exists(pt_path):
                        continue
                    t = torch.load(pt_path)
                    t = process_tensor(t, name)
                    if t is None:
                        print("Did not find tensor for", name, "at", pt_path)
                        continue

                    with open(os.path.join(cond_path, run, "target.txt"), 'r') as f:
                        target = f.read().strip()
                    
                    names = [name]
                    if len(t.shape) > 1:
                        print("Found multiple columns for", name)
                        while len(names) < t.shape[1]:
                            names.append(name + f"_{len(names)}")
                    df = pd.DataFrame(t.numpy(), columns=names)
                    df["condition"] = cond
                    df["run"] = str(run)
                    df["target"] = target
                    df["batch"] = df.index

                    df = df[df[name].notna()]
                    df = df[df[name] != float('inf')]
                    df = df[df[name] != float('-inf')]
                    df = df[df[name] != float('nan')]
                    df = df[df[name] != float('NaN')]
                    df = df[df[name] != 'NaN']
                    df = df.reset_index(drop=True)


                    df.index = df.index + df_index_offset
                    df_index_offset += len(df)
                    results.append(df)
        except Exception as e:
            print("Error processing", name)
            print(e)
            print(traceback.format_exc())
            continue
            
    final_df = results[0]
    for df in tqdm(results[1:], desc="Merging"):
        final_df = final_df.combine_first(df)
    return final_df



def plot_xy(results, x, y, save_path=None, show=False, x_label=None, y_label=None):
    plt.figure(figsize=(10,10))
    # use_save_path = os.path.join(save_path, f"{y}.png")
    use_save_path = save_path
    if 'target' not in results.columns:
        sns.lineplot(results, x=x, y=y, hue="condition")
        use_save_path = use_save_path.replace(".png", "_avg.png")
    else:
        sns.lineplot(results, x=x, y=y, hue="condition", style="target")

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    # plt.tight_layout(rect=[0, 0, .8, 1])
    if save_path:
        plt.savefig(use_save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_vs_batches(results, y, save_path=None, show=False, max_ofs=None):
    if max_ofs is not None:
        results = results[(results["offspring_by_batch"] <= max_ofs) & (results["offspring_by_batch"] > 0)]
    plot_xy(results, "batch", y, save_path, show)

def plot_vs_evals(results, y, save_path=None, show=False, mean_by_target=False,smooth=None):
    results = results.drop(columns=[c for c in results.columns if c not in[ "condition", "target", 'run', 'batch', 'evals_by_batch', y]])
    
    num_points = len(results['batch'].unique())
    if smooth is not None:
        num_points = int(num_points * (1.0-smooth))
    evals_by_batch = np.linspace(results['evals_by_batch'].min(),
                                 results['evals_by_batch'].max(),
                                 num_points)
    # find closest evals 
    results['evals'] = results['evals_by_batch'].apply(lambda x: evals_by_batch[np.argmin(np.abs(evals_by_batch - x))])
    
    save_name = f"{y}_v_evals.png"
    if mean_by_target:
        results = results.groupby(["condition", 'run', 'batch']).mean(numeric_only=True).reset_index()
        save_name = f"{y}_v_evals_avg.png"
    else:
        results = results.groupby(["condition", "target", 'run','batch']).mean(numeric_only=True).reset_index()
    
    
    save_path = os.path.join(save_path, save_name)
    plot_xy(results, "evals", y, save_path, show, x_label="Forward passes")


def plot_vs_offspring(results, y, save_path=None, show=False, mean_by_target=False):
    results = results.drop(columns=[c for c in results.columns if c not in[ "condition", "target", 'run', 'batch', 'offspring_by_batch', y]])
    if mean_by_target:
        results = results.groupby(["condition", 'run', 'batch']).mean(numeric_only=True).reset_index()
        # save_path = save_path.split(".")[-2] + "_avg" + save_path.split(".")[-1]
    else:
        results = results.groupby(["condition", "target", 'run','batch']).mean(numeric_only=True).reset_index()

    plot_xy(results, "offspring_by_batch", y, save_path, show)

def plot_vs_time(results, y, save_path=None, show=False, mean_by_target=False):
    results = results.groupby(["condition", "target", 'run', 'batch']).mean(numeric_only=True).reset_index()
    # get the average time at each batcheration
    results = results.groupby(["condition", "target", "batch"]).agg({"time": "mean", y:"mean"}).reset_index()

    save_name = f"{y}_v_time.png"
    print(results)
    if mean_by_target:
        results = results.groupby(["condition", "batch"]).agg({"time": "mean", y:"mean"}).reset_index()
        sns.lineplot(results, x="time", y=y, hue="condition")
        save_name = f"{y}_v_time_avg.png"
    else:
        sns.lineplot(results, x="time", y=y, hue="condition", style="target")
    if save_path:
        plt.savefig(os.path.join(save_path, save_name))
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", type=str, help="Path to results directory")
    parser.add_argument("-s", "--show", action="store_true", help="Show plots")
    parser.add_argument("-t", "--target", type=str, default="", help="Target image name (to filter results)")
    args = parser.parse_args()

    save_path = os.path.join(args.results_path, "plots")
    os.makedirs(os.path.join(save_path), exist_ok=True)

    metrics = [
        "evals_by_batch",
        "normed_fitness_by_batch",
        "pruned_cxs",
        # "offspring_by_batch",
        # "fitness_by_batch",
        "cx_by_batch",
        "nodes_by_batch",

    ]
    # offs_results = read_tensor_results(args.results_path, "offspring_by_batch")
    results = read_tensor_results(args.results_path, metrics, max_runs=None)
    # results = results.dropna()
    print(results)

    for m in tqdm(['normed_fitness_by_batch', 'cx_by_batch', 'nodes_by_batch', 'pruned_cxs'], desc="Plotting"):
        plot_vs_evals(results, m, save_path, args.show)
        plot_vs_evals(results, m, save_path, args.show, mean_by_target=True)

