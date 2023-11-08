import argparse
import pandas as pd
import os 
import seaborn as sns
import matplotlib.pyplot as plt
import torch

def read_results(results_path):
    fits_path = os.path.join(results_path, "results.pkl")
    df = pd.read_pickle(fits_path)
    return df

def read_tensor_results(results_path, name):
    cond_dir = os.path.join(results_path, "conditions")
    results = []
    for cond in os.listdir(cond_dir):
        cond_path = os.path.join(cond_dir, cond)
        if not os.path.isdir(cond_path):
            continue
        for run in os.listdir(cond_path):
            pt_path = os.path.join(cond_path, run, f"{name}.pt")
            pt_off_path = os.path.join(cond_path, run, f"offspring_over_time.pt")
            if not os.path.exists(pt_path):
                continue
            with open(os.path.join(cond_path, run, "target.txt"), 'r') as f:
                target = f.read().strip()
            df = pd.DataFrame(torch.load(pt_path).numpy(), columns=[name])
            df["condition"] = cond
            df["run"] = str(run)
            df["target"] = target
            df["gen"] = df.index
            df["total_offspring"] = torch.load(pt_off_path).numpy()
            results.append(df)
    return pd.concat(results)

def plot_vs_gens(results, y, save_path=None, show=False, max_ofs=None):
    if max_ofs is not None:
        results = results[(results["total_offspring"] <= max_ofs) & (results["total_offspring"] > 0)]
    sns.lineplot(results, x="gen", y=y, hue="condition", style="target")
    if save_path:
        plt.savefig(os.path.join(save_path, f"{y}_v_gens.png"))
    if show:
        plt.show()
    plt.close()
    
def plot_vs_offspring(results, y, save_path=None, show=False, mean_by_target=False):
    results = results.drop(columns=[c for c in results.columns if c not in[ "condition", "target", 'run', 'gen', 'total_offspring', y]])
    if mean_by_target:
        results = results.groupby(["condition", 'run', 'gen']).mean().reset_index()
        sns.lineplot(results, x="total_offspring", y=y, hue="condition")
    else:
        results = results.groupby(["condition", "target", 'run','gen']).mean().reset_index()
        sns.lineplot(results, x="total_offspring", y=y, hue="condition", style="target")
    if save_path:
        plt.savefig(os.path.join(save_path, f"{y}_v_offspring.png"))
    if show:
        plt.show()
    plt.close()
    
def plot_vs_time(results, y, save_path=None, show=False, mean_by_target=False):
    results = results.groupby(["condition", "target", 'run', 'gen']).mean(numeric_only=True).reset_index()
    # get the average time at each generation
    results = results.groupby(["condition", "target", "gen"]).agg({"time": "mean", y:"mean"}).reset_index()
    
    save_name = f"{y}_v_time.png"
    print(results)
    if mean_by_target:
        results = results.groupby(["condition", "gen"]).agg({"time": "mean", y:"mean"}).reset_index()
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
    
    results = read_results(args.results_path)
    # remove nan rows
    results = results[results['condition'].notna()]
    results['run'] = results['run'].astype(str)
    assert len(args.target)==0  or args.target in results["target"].unique()
    if len(args.target) > 0:
        results = results[results["target"] == args.target]

    plot_vs_time(results, "fitness", save_path, args.show)
    plot_vs_time(results, "fitness", save_path, args.show, mean_by_target=True)
    plot_vs_time(results, "total_offspring", save_path, args.show, mean_by_target=True)
    plot_vs_offspring(results, "fitness", save_path, args.show, mean_by_target=True)
    exit()
    
    
    plot_vs_offspring(results, "diversity", save_path, args.show)
    plot_vs_offspring(results, "avg_num_connections", save_path, args.show)
    plot_vs_offspring(results, "avg_num_hidden_nodes", save_path, args.show)
    plot_vs_offspring(results, "time", save_path, args.show)
    plot_vs_offspring(results, "population", save_path, args.show)
    plot_vs_offspring(results, "fitness", save_path, args.show)
    # plot_vs_offspring(results, "diversity", save_path, args.show)
    
    
    metrics = [
        "lr_over_time",
        
    ]
    offs_results = read_tensor_results(args.results_path, "offspring_over_time")
    plot_vs_gens(offs_results, "offspring_over_time", save_path, args.show, results["total_offspring"].max())
    
    for m in metrics:
        tensor_results = read_tensor_results(args.results_path, m)
        plot_vs_offspring(tensor_results, m, save_path, args.show)