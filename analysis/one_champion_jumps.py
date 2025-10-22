import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patches as patches

from collect_data import collect_data
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42 
matplotlib.rcParams.update({'font.size': 20})
from matplotlib import colors
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm


save_dir = ""
args = object()
num_to_plot = 4


pretty_names = {
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

ordered_names = list(pretty_names.keys())
ordered_names = sorted(ordered_names, key=lambda x: x.lower())


def plot_trajectory(parents_df, ids_df, condition, run=None, do_all_cells=False, skip_plot=False):
    df = pd.DataFrame(columns=["run", "condition", "unique_cells", "id"])
    # ids: ,condition,cell,run,id,gen
    # parents: ,condition,cell,run,parent0,parent1,gen
    # fits: ,condition,cell,run,function,gen,fitness
    
    jumps_df = pd.DataFrame(columns=["run", "condition", "unique_cells", "jumps", "id"])
    runs = []
    if run is None:
        runs = parents_df["run"].unique()
    else:
        runs = [run]
    for run_id in runs:
        print("run", run_id)
        last_gen = parents_df["gen"].max()
        
        run_df_ids = ids_df.loc[(ids_df["condition"] == condition) & (
            ids_df["run"] == run_id)]
        run_df_parents = parents_df.loc[(
            parents_df["condition"] == condition) & (parents_df["run"] == run_id)]
        
        
        if do_all_cells:
            cells = run_df_ids["cell"].unique()
        else:
            # cells = np.random.choice(run_df_ids["cell"].unique(), num_to_plot)
            # cells = [run_df_ids["cell"].unique()[50]]
            cells = run_df_ids[run_df_ids["id"] == run_df_ids['id'].max()]["cell"].unique()

        for cell in tqdm(cells, leave=False, mininterval=0, miniters=1):
            # print("cell", cell)
            total_count = 0
            final_id_in_cell = run_df_ids.loc[(run_df_ids["cell"] == cell) & (
                run_df_ids["gen"] == last_gen), "id"].values[0]
            df = df.append({"run": run_id, "condition": condition,
                            "unique_cells": 1, "id": final_id_in_cell}, ignore_index=True)
            current_gen = last_gen
            current_cell = cell
            trajectories = []
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
                prev_cell = current_cell
                current_cell = cell_values[0]
                total_count += 1
                trajectories.append((prev_cell, current_cell))
                # is this cell already in genealogy?
                if not current_cell in genealogy:
                    # add 1 to unique cells
                    # df.loc[(df["id"] == final_id_in_cell) & (df["run"] == run) & (df["condition"] == condition), "unique_cells"] += 1
                    genealogy[int(current_gen)] = current_cell
                current_gen -= 1
            genealogy = [x for x in genealogy if x is not None]
            df.loc[(df["id"] == final_id_in_cell) & (df["run"] == run_id) & (
                df["condition"] == condition), "unique_cells"] = len(set(genealogy))
            df.loc[(df["id"] == final_id_in_cell) & (df["run"] == run_id) & (
                df["condition"] == condition), "total_jumps"] = total_count
            
            jumps_df = jumps_df.append({"run": run_id, "condition": condition,
                            "unique_cells": len(set(genealogy)), "jumps": total_count, "id": final_id_in_cell}, ignore_index=True)

            trajectories = trajectories[::-1]
            
            if not skip_plot:
                all_cells = np.array(run_df_ids["cell"].unique())
                all_cells = all_cells.reshape((10, 10))

                # # put a green dot on the first cell
                # x0 = np.where(all_cells == trajectories[-1][1])[1][0]
                # y0 = np.where(all_cells == trajectories[-1][1])[0][0]
                # ax.scatter(x0, y0, c="g", s=300)
                # # put a red dot on the last cell
                # x0 = np.where(all_cells == trajectories[0][0])[1][0]
                # y0 = np.where(all_cells == trajectories[0][0])[0][0]
                # ax.scatter(x0, y0, c="r", s=300)
                print(save_dir)
                cmap =  matplotlib.cm.get_cmap('plasma')
                norm = matplotlib.colors.Normalize(vmin=0, vmax=int(len(trajectories)*1.25))
                os.makedirs(os.path.join(save_dir, "trajectories", "anim"), exist_ok=True)
                for idx, tr in enumerate(trajectories):
                    # plot as grid of text labels
                    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
                    # set axes limits
                    ax.set_xlim(-0.5, 10-.5)
                    ax.set_ylim(-0.5, 10-.5)
                    for i in range(10):
                        for j in range(10):
                            name = all_cells[i, j]
                            name = name.split("_")
                            name= [f" ({ordered_names.index(n)}) {pretty_names[n]}" for n in name]
                            name = "\n".join(name)
                            color  = (0,0,0,.65)
                            ax.text(j-.45, i, name , ha="left", va="center", color=color, zorder=9)
                            ax.vlines(j-.5, -.5, 10-.5, color="k", zorder=0)
                        ax.hlines(i-.5, -.5, 10-.5, color="k", zorder=0)
                    ax.hlines(10-.5, -.5, 10-.5, color="k", zorder=0)
                    ax.vlines(10-.5, -.5, 10-.5, color="k", zorder=0)
                    ax.set_title("")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for t in trajectories[:idx]:
                        
                        
                        # t = (prev_cell, current_cell)
                        # draw arrow from t[0] to t[1]
                        # get coordinates of t[0]
                        x0 = np.where(all_cells == t[1])[1][0]
                        y0 = np.where(all_cells == t[1])[0][0]
                        # get coordinates of t[1]
                        x1 = np.where(all_cells == t[0])[1][0]
                        y1 = np.where(all_cells == t[0])[0][0]
                        
                        # curved
                        style = "Simple, tail_width=15, head_width=40, head_length=40"
                        kw = dict(arrowstyle=style, color=cmap(norm(trajectories.index(t))))
                        a = patches.FancyArrowPatch((x0, y0), (x1, y1), connectionstyle="arc3,rad=0.1", zorder=10, **kw)
                        ax.add_patch(a)

                        # not curved
                        # ax.arrow(x0, y0, x1-x0, y1-y0, head_width=0.15, head_length=0.15, fc=cmap(norm(trajectories.index(t))), ec=cmap(norm(trajectories.index(t))),
                                #  length_includes_head=True, width=0.05)

                    
                    plt.tight_layout()
                    print(os.path.join(save_dir, f"trajectories", "anim", f"all_cells_{condition}_{run}_{final_id_in_cell}_{idx:04}.png"))
                    plt.savefig(os.path.join(save_dir, f"trajectories", "anim", f"all_cells_{condition}_{run}_{final_id_in_cell}_{idx:04}.png"))
                    plt.close()

    if do_all_cells and run==None:
        jumps_df.to_csv(os.path.join(save_dir, "trajectories/jumps.csv"), index=False)
    else:
        jumps_df.to_csv(os.path.join(save_dir, f"trajectories/jumps_{condition}_{run}.csv"), index=False)
        import imageio
        images = []
        for filename in sorted(list(os.path.join(save_dir, "trajectories", "anim"))):
            if filename.endswith(".png"):
                images.append(imageio.imread(os.path.join(save_dir, "trajectories", "anim/", filename)))
        imageio.mimsave(os.path.join(save_dir, "trajectories", "all.gif"), images)
    
    print("Done.")
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--condition", "-c", type=str, default=None)
    parser.add_argument("--run_dir", "-r", type=str, default=None)
    parser.add_argument("--all_cells", "-a", action="store_true")
    args = parser.parse_args()
    save_dir = args.save_dir  
    
    dfs = collect_data(
        dir=save_dir,
        overwrite_run_dfs=False,
        one_run=args.run_dir,
        one_cond=args.condition,
        load_overall_dfs=True,
        num_workers=1,
        fns=['parents', 'ids']
    )  
    
    os.makedirs(os.path.join(save_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "trajectories","anim"), exist_ok=True)
    if args.run_dir is None:
        plot_trajectory(dfs['parents'], dfs['ids'], args.condition, None, args.all_cells, skip_plot=True)
    else:
        plot_trajectory(dfs['parents'], dfs['ids'], args.condition, int(args.run_dir.split('_')[-1]), args.all_cells)
    
    
# id = 195866