import csv
import pickle
from queue import Queue
import random
from time import sleep
import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm, trange
import multiprocessing as mp
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
ProgressBar().register()

CHECKPOINT_DF_FREQ = 5
device = torch.device("cpu")

def checkpoint_run(df, cond_name, run_id, run_dir, name):
    run_df = df[df["run"] == run_id]
    run_df = run_df[run_df["condition"] == cond_name]
    run_df.to_pickle(os.path.join(run_dir, name))
    
def checkpoint_df(args):
    df, path = args
    df.to_pickle(path)

def try_load_checkpoint(run_dir, name):
    if os.path.exists(os.path.join(run_dir, name)):
       try:
            df = pd.read_pickle(os.path.join(run_dir, name))
            return df
       except Exception as e:
           print(e)
           print(f"Error reading {name}, rebuilding")
           return None
           
def build_fits(cond, target, run_id, run_dir, cell_names, function_names, overwrite=False):
    if not overwrite:
        checkpoint = try_load_checkpoint(run_dir, "fits.pkl")
        if checkpoint is not None:
            return checkpoint

    # (n_fns, n_cells, num_generations)
    fit = torch.load(os.path.join(
        run_dir, "fitness_by_batch.pt"), map_location=device)

    # Initialize fits_df dataframe with data from fit tensor
    fits_df = pd.DataFrame(
        [(cond, cell_names[cell], run_id, function_names[fn], gen, fit[fn, cell, gen].item())
         for (fn, cell, gen), _ in np.ndenumerate(fit)],
        columns=["condition", "cell", "run", "function", "gen", "fitness"]
    )
    fits_df["gen"] = fits_df["gen"].astype(int)
    fits_df["run"] = fits_df["run"].astype(np.uint)
    fits_df["target"] = target

    checkpoint_run(fits_df, cond, run_id, run_dir, "fits.pkl")
    return fits_df

def build_ids(cond, target, run_id, run_dir, cell_names, _, overwrite=False):
    if not overwrite:
        checkpoint = try_load_checkpoint(run_dir, "ids.pkl")
        if checkpoint is not None:
            return checkpoint
        
    # (n_cells, num_generations)
    ids = torch.load(os.path.join(
        run_dir, "ids_over_time.pt"), map_location=device)
    total_ids = ids.shape[0] * ids.shape[1]

    ids_df = pd.DataFrame(
    [(cond, cell_names[cell], run_id, gen, id_.item()) for (cell, gen), id_ in np.ndenumerate(ids)],
        columns=["condition", "cell", "run", "gen", "id"]
    )
    ids_df["gen"] = ids_df["gen"].astype(int)
    ids_df["run"] = ids_df["run"].astype(int)
    ids_df["id"] = ids_df["id"].astype(int)
    ids_df["target"] = target

    checkpoint_run(ids_df, cond, run_id, run_dir, "ids.pkl")
    return ids_df

def build_replaces(cond, target, run_id, run_dir, cell_names, _, overwrite=False):
    if not overwrite:
        checkpoint = try_load_checkpoint(run_dir, "replaces.pkl")
        if checkpoint is not None:
            return checkpoint
   
    # (cells_from, cells_to, num_generations)
    replace = torch.load(os.path.join(
        run_dir, "replacements_over_time.pt"), map_location=device)
    
    total_replaces = replace.shape[0] * replace.shape[1] * replace.shape[2]
    list_ = [None] * total_replaces
    dict_ = {}
    pbar = tqdm(total=total_replaces, desc=f"replaces {cond} {run_id}")
    i = 0
    for (cell_from, cell_to, gen), rep in np.ndenumerate(replace.numpy()):
        if rep == 0.0:
            i+=1
            pbar.update(1)
            continue
        dict_ = {
            "cell-from":cell_names[cell_from],
            "cell-to": cell_names[cell_to],
            "gen": gen,
            "replacements": rep,
        }
        list_[i] = dict_
        i+=1
        pbar.update(1)

    pbar.close()
    list_ = [x for x in list_ if x is not None]
    replaces_df = pd.DataFrame(list_, columns=["condition", "cell-from", "cell-to", "run", "gen", "replacements"])
    replaces_df["replacements"] = replaces_df["replacements"].astype(int)
    
    replaces_df['condition'] = cond
    replaces_df['run'] = run_id
    
    # TODO:
    replaces_df = replaces_df[replaces_df["replacements"] > 0]
    
    replaces_df["gen"] = replaces_df["gen"].astype(int)
    replaces_df["run"] = replaces_df["run"].astype(int)
    replaces_df["target"] = target
    
    checkpoint_run(replaces_df, cond, run_id, run_dir, "replaces.pkl")
    return replaces_df

def build_parents(cond, target, run_id, run_dir, cell_names, function_names, overwrite=False):
    if not overwrite:
        checkpoint = try_load_checkpoint(run_dir, "parents.pkl")
        if checkpoint is not None:
            return checkpoint

    # (2, n_cells, num_generations)
    parents = torch.load(os.path.join(
        run_dir, "parents_over_time.pt"), map_location=device)

    # Initialize parents_df dataframe with data from parents tensor
    parents_df = pd.DataFrame(
        [(cond, cell_names[cell], run_id, parents[0, cell, gen].item(), parents[1, cell, gen].item(), gen)
        for (cell, gen), _ in np.ndenumerate(parents[0])],
        columns=["condition", "cell", "run", "parent0", "parent1", "gen"]
    )
    parents_df["gen"] = parents_df["gen"].astype(int)
    parents_df["run"] = parents_df["run"].astype(int)
    parents_df["parent0"] = parents_df["parent0"].astype(int)
    parents_df["parent1"] = parents_df["parent1"].astype(int)
    parents_df["target"] = target

    checkpoint_run(parents_df, cond, run_id, run_dir, "parents.pkl")
    return parents_df

class Worker(mp.Process):
    def __init__(self, worker_id, in_queue, out_queue, stop):
        self.worker_id = worker_id
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop = stop 
        super(Worker, self).__init__()
    
    def run(self):  
        while not self.stop.is_set():
            if not self.in_queue.empty():
                try:
                    fn, cond, target, run_id, run_dir, cell_names, function_names, overwrite = self.in_queue.get()
                    print(f"Worker {self.worker_id} starting", cond, run_id, "with function", fn.__name__)
                    df = fn(cond, target, run_id, run_dir, cell_names, function_names, overwrite)
                    self.out_queue.put((fn, df, self.worker_id))
                except Exception as e:
                    print(e, "in worker", self.worker_id)
                    self.out_queue.put((None, None, None)) # signal to stop
            sleep(0.1 + random.random() * .5)

def collect_data(dir, overwrite_run_dfs=False, one_run=None, one_cond=None, load_overall_dfs=False, num_workers=1, fns=None):
    dir = os.path.join(dir, "conditions")
    assert os.path.isdir(dir), f"Directory {dir} does not exist"

    fits_save_path = os.path.join(dir, "fits.pkl")
    replace_save_path = os.path.join(dir, "replacements.pkl")
    ids_save_path = os.path.join(dir, "ids.pkl")
    parents_save_path = os.path.join(dir, "parents.pkl")
    fits_df = pd.DataFrame(
        columns=["condition", "cell", "run", "function", "gen", "fitness"])
    ids_df = pd.DataFrame(columns=["condition", "cell", "run", "id"])
    replaces_df = pd.DataFrame(
        columns=["condition", "child_sub_id", "cell", "run", "gen", "replacements"])
    parents_df = pd.DataFrame(
        columns=["condition", "cell", "run", "parent0", "parent1"])

    expected_runs = 0

    if one_cond is not None:
        cond_dirs = [os.path.join(dir, one_cond)]
    else:
        cond_dirs = [os.path.join(dir, o) for o in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, o))]
    
    for cond_dir in cond_dirs:
        if one_run is not None:
            if "/" in one_run:
                run_dirs = [one_run]
            else:
                run_dirs = [os.path.join(cond_dir, one_run)]
        else:
            run_dirs = [os.path.join(cond_dir, o) for o in os.listdir(
                cond_dir) if os.path.isdir(os.path.join(cond_dir, o))]
        expected_runs += len(run_dirs)
    
    if (not "fits" in fns or os.path.isfile(fits_save_path))\
        and (not "replacements" in fns or os.path.isfile(replace_save_path))\
        and (not "ids" in fns or os.path.isfile(ids_save_path))\
        and (not "parents" in fns or os.path.isfile(parents_save_path)):
        print("Found saved dataframes")
        if load_overall_dfs:
            print("Loading saved dataframes...")
            try:
                if "fits" in fns:
                    use_cols = ["condition","cell","run","function","gen","fitness"]
                    dtypes = {
                        'condition': str,
                        'cell': str,
                        'function': str,
                        'run': np.uint8,
                        'gen': np.int16,
                        'fitness': np.float32
                        
                    }
                    f = pd.read_pickle(fits_save_path)
                    f = f.filter(use_cols).astype(dtypes)
                    # print("\tread (dask)")
                    # f = f.compute(num_workers=num_workers, scheduler="threads") # dask->pandas
                    # f = pd.read_csv(fits_save_path)
                    print("\tLoaded fits with shape", f.shape)
                    # check "Saved fits dataframe matches expected number of runs"
                    assert len(f.groupby(["condition", "run"])) == expected_runs, "Saved fits dataframe does not match expected number of runs"
                else:
                    f = None
                if "replacements" in fns:
                    use_cols = ["condition","cell-to", "cell-from", "run", "gen", "replacements"]
                    dtypes = {
                        'condition': str,
                        'cell-to': str,
                        'cell-from': str,
                        'run': np.uint8,
                        'gen': np.int16,
                        'replacements': np.bool_
                        
                    }
                    
                    r = pd.read_pickle(replace_save_path)
                    r = r.filter(items=use_cols)
                    r = r.astype(dtypes)
                    r = r[r['replacements'] == True]
                    # print("\tread (dask)")
                    # r= r.compute(num_workers=num_workers, scheduler="threads") # dask->pandas
                    print("\tLoaded replacements with shape", r.shape)
                    assert len(r.groupby(["condition", "run"])) == expected_runs, "Saved replacements dataframe does not match expected number of runs"
                else:
                    r = None
                if "ids" in fns:
                    i = pd.read_pickle(ids_save_path)
                    print("\tLoaded ids with shape", i.shape)
                    assert len(i.groupby(["condition", "run"])) == expected_runs, "Saved ids dataframe does not match expected number of runs"
                else:
                    i = None
                if "parents" in fns:
                    p = pd.read_pickle(parents_save_path)
                    print("\tLoaded parents with shape", p.shape)
                    assert len(p.groupby(["condition", "run"])) == expected_runs, "Saved parents dataframe does not match expected number of runs"
                else:
                    p = None
                print("Done loading saved dataframes")
                return {
                    "fits": f,
                    "replacements": r,
                    "ids": i,
                    "parents": p
                }
                
            except Exception as e:
                print(f"Error ({e}) reading saved dataframes, rebuilding")
            
    print("Rebuilding dataframes...")


    if fns is None:
        fns = [build_fits, build_replaces, build_ids, build_parents]
    else:
        use_fns = []
        for f in fns:
            if f == "fits":
                use_fns.append(build_fits)
            elif f == "replacements":
                use_fns.append(build_replaces)
            elif f == "ids":
                use_fns.append(build_ids)
            elif f == "parents":
                use_fns.append(build_parents)
            else:
                raise ValueError(f"Unknown function {f}")
        fns = use_fns
    
    
    
    print("Found conditions: ", cond_dirs)
        
    # initialize workers
    if num_workers > 1:
        in_queue = mp.Queue()
        out_queue = mp.Queue()
        stop = mp.Event()
        # inputs to workers : fn, cond, run_id, run_dir, cell_names, function_names, overwrite, worker_id
        # outputs from workers : fn, df, worker_id
        workers = []
        for i in range(num_workers):
            workers.append(Worker(i, in_queue, out_queue, stop))
            workers[-1].start()   
    else:
        out_queue = Queue() 
        
    run_count = 0
    df_count = 0
    for cond_dir in tqdm(cond_dirs):
        if one_run is not None:
            if "/" in one_run:
                run_dirs = [one_run]
            else:
                run_dirs = [os.path.join(cond_dir, one_run)]
        else:
            run_dirs = [os.path.join(cond_dir, o) for o in os.listdir(
                cond_dir) if os.path.isdir(os.path.join(cond_dir, o))]
        cond = os.path.basename(cond_dir)
        # print("\tFound runs: ", run_dirs)
        for run_dir in run_dirs:
            run_id = np.uint(os.path.basename(run_dir).split("_")[-1])
            # (n_cells)
            try:
                cell_names = list(csv.reader(
                    open(os.path.join(run_dir, "cell_names.csv"))))[0]
            except FileNotFoundError:
                print("No cell_names.csv found in", run_dir)
                continue
            # (n_fns)
            function_names = list(csv.reader(
                open(os.path.join(run_dir, "function_names.csv"))))[0]
            
            target = os.path.join(run_dir, "target.txt")
            with open(target, "r") as f:
                target = f.read().strip()


            for fn in fns:
                if num_workers > 1:
                    in_queue.put((fn, cond, target, run_id, run_dir, cell_names, function_names, overwrite_run_dfs))
                else:
                    # call function directly
                    out_queue.put((fn, fn(cond, target, run_id, run_dir, cell_names, function_names, overwrite_run_dfs), -1))
                df_count += 1
            run_count += 1
            
    pbar = tqdm(total=df_count, desc="Collecting dfs...")
    fits_all = [None] * run_count
    ids_all = [None] * run_count
    replaces_all = [None] * run_count
    parents_all = [None] * run_count
    fits_count, ids_count, replaces_count, parents_count = 0, 0, 0, 0
    # wait for workers to finish
    while df_count > 0:
        if not out_queue.empty():
            fn, df, worker_id = out_queue.get()
            if worker_id is None:
                print("Worker died")
                break
            if fn == build_fits:
                fits_all[fits_count] = df
                fits_count += 1
                # fits_df = pd.concat([fits_df, df])
            elif fn == build_replaces:
                replaces_all[replaces_count] = df
                replaces_count += 1
                # replaces_df = pd.concat([replaces_df, df])
            elif fn == build_ids:
                ids_all[ids_count] = df
                ids_count += 1
                # ids_df = pd.concat([ids_df, df])
            elif fn == build_parents:
                parents_all[parents_count] = df
                parents_count += 1
                # parents_df = pd.concat([parents_df, df])
            df_count -= 1
            pbar.update(1)
        sleep(0.2)
    pbar.close()
    print("Finished collecting dfs")
    assert df_count == 0, "Not all dataframes were collected"
    if num_workers > 1:
        print("Stopping workers...")
        # stop workers
        stop.set()
        for worker in workers:
            worker.join()
        
    
        for worker in workers:
            assert worker.in_queue.empty(), "Worker input queue not empty"
            assert worker.out_queue.empty(), "Worker output queue not empty"
            assert not worker.is_alive(), "Worker still alive"
    print("Concatenating dataframes...")
    # print(fits_count, replaces_count, ids_count, parents_count)
    # print(len([f for f in fits_all if f is not None]), len([f for f in replaces_all if f is not None]), len([f for f in ids_all if f is not None]), len([f for f in parents_all if f is not None]))
    if build_fits in fns:
        assert fits_count == run_count, "Not all fits dfs were collected"
        fits_df = pd.concat(fits_all).reset_index(drop=True)
    if build_replaces in fns:
        assert replaces_count == run_count, "Not all replaces dfs were collected"
        replaces_df = pd.concat(replaces_all).reset_index(drop=True)
    if build_ids in fns:
        assert ids_count == run_count, "Not all ids dfs were collected"
        ids_df = pd.concat(ids_all).reset_index(drop=True)
    if build_parents in fns:
        assert parents_count == run_count, "Not all parents dfs were collected"
        parents_df = pd.concat(parents_all).reset_index(drop=True)
    
    if not one_cond and not one_run:
        print("Checkpointing dataframes...")
        for fn in fns:
            if fn == build_fits:
                checkpoint_df((fits_df, fits_save_path))
            elif fn == build_replaces:
                checkpoint_df((replaces_df, replace_save_path))
            elif fn == build_ids:
                checkpoint_df((ids_df, ids_save_path))
            elif fn == build_parents:
                checkpoint_df((parents_df, parents_save_path))

    return {"fits": fits_df, "replacements": replaces_df, "ids": ids_df, "parents": parents_df}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str)
    parser.add_argument("-c", "--cond_dir", type=str, required=False)
    parser.add_argument("-r", "--run_dir", type=str, required=False)
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("-w", "--workers", type=int, default=mp.cpu_count()-1)
    parser.add_argument("-l", "--load", action="store_true")
    parser.add_argument("-f", "--fns", type=str, nargs="+", default=["fits", "replacements", "ids", "parents"])
    args = parser.parse_args()
    print("using {} workers".format(args.workers))
    
    collect_data(
        dir=args.experiment_dir,
        overwrite_run_dfs=args.overwrite,
        one_run=args.run_dir,
        one_cond=args.cond_dir,
        load_overall_dfs=args.load,
        num_workers=args.workers,
        fns = args.fns
    )
    