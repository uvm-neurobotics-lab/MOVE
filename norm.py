import logging
import os
import pandas as pd
import torch

def read_norm_data(norm_df_path, target=None):
    assert os.path.exists(norm_df_path), f"Norm data not found at {norm_df_path}"
    if norm_df_path.endswith('.csv'):
        norm_df = pd.read_csv(norm_df_path)
    elif norm_df_path.endswith('.pkl'):
        norm_df = pd.read_pickle(norm_df_path)
    else:
        raise ValueError(f"Unknown norm data format {norm_df_path}")
        
    if target is not None:
        if target in norm_df['target'].unique():
            norm_df = norm_df[norm_df['target'] == target]
        else:
            # print(f"WARNING: target {target} not found in {norm_df_path}, using mean of all targets")
            norm_df = norm_df.groupby('function').mean(numeric_only=True).reset_index()
            
    return norm_df

def read_norm_data_to_tensor(norm_df_path, target=None, run_agg='mean')->torch.Tensor:
    norm_df = read_norm_data(norm_df_path, target)
    output = torch.zeros(len(norm_df.function.unique()), 2)
    for i, fn in enumerate(norm_df.function.unique()):
        output[i,0] = norm_df[norm_df['function'] == fn][f'min_fitness_{run_agg}'].values[0]
        output[i,1] = norm_df[norm_df['function'] == fn][f'max_fitness_{run_agg}'].values[0]
    return output
    


def norm_tensor(input, norm_df, fn_name, run_agg='mean', method: str='minmax', clamp=False, warn=True):
    eps = 1e-8
    if isinstance(norm_df, str) or isinstance(norm_df, os.PathLike):
        norm_df = read_norm_data(norm_df)
    if fn_name not in norm_df['function'].unique():
        if warn:
            logging.warning(f"No norm for function {fn_name}, out of options: {norm_df['function'].unique()}")
        return input
    if method == 'minmax':
        col_name_min = f"min_fitness_{run_agg}"
        col_name_max = f"max_fitness_{run_agg}"
        if col_name_min not in norm_df.columns:
            col_name_min = "min_fitness"
        if col_name_max not in norm_df.columns:
            col_name_max = "max_fitness"
        min_ = norm_df[norm_df["function"] == fn_name][col_name_min].values[0]
        max_ = norm_df[norm_df["function"] == fn_name][col_name_max].values[0]
        
        normed = (input - min_) / (eps + max_ - min_)
        if not((normed > 0).all() and (normed < 1).all()):
            if warn:
                logging.warning(f"Normed value [{normed.min().item()}-{normed.max().item()}] out of range for function {fn_name} original was: [{input.min().item()}-{input.max().item()}]")
        
        if clamp:
            normed = torch.clamp(normed, 0, 1)
            
        return normed
    elif method == 'zscore':
        mean = norm_df[norm_df["function"] == fn_name][f"fitness_mean"].values[0]
        std = norm_df[norm_df["function"] == fn_name][f"fitness_std"].values[0]
        normed = (input - mean) / std
        if not((normed > 0).all() and (normed < 1).all()):
            logging.warning(f"Normed value [{normed.min().item()}-{normed.max().item()}] out of range for function {fn_name} original was: [{input.min().item()}-{input.max().item()}]")
        if clamp:
            normed = torch.clamp(normed, 0, 1)
        return normed
        
    else:
        raise ValueError(f"Unknown method {method}")
    
    return normed

def norm_tensor_by_tensor(input, norm):
    # not sure we should use this, since there's no guarantee that the functions are in the same order
    raise NotImplementedError
    min_ = norm[:,0][:,None]
    max_ = norm[:,1][:,None]
    normed = (input.T - min_) / (max_ - min_)
    return normed.T


def norm(df, norm, excluded_from_norm=['baseline-aggregate'], skip_conditions=[], run_agg='mean'):
    df["normed_fitness"] = df["fitness"]
    
    for fn in df["function"].unique():
        if fn in excluded_from_norm:
            continue
        this_norm = norm.loc[norm["function"] == fn]
        if len(this_norm) == 0:
            print(f"No norm for function {fn}")
            continue
        
        postfix = f"_{run_agg}" if run_agg is not None else ""
        
        if f"fitness_mean{postfix}" not in this_norm.columns:
            postfix = ""
            
        max_fitness = this_norm[f"max_fitness{postfix}"].values[0]
        min_fitness = this_norm[f"min_fitness{postfix}"].values[0]
        
        # gross:
        df.loc[(df["function"] == fn) &(~ df["condition"].isin(skip_conditions)), "normed_fitness"] = (
            (df.loc[(df["function"] == fn) &(~ df["condition"].isin(skip_conditions)), "normed_fitness"] - min_fitness) /
            (max_fitness - min_fitness))
        
        
def norm_from_saved(df, baseline_dir, excluded_from_norm=['baseline-aggregate'], skip_conditions=[],run_agg='mean'):
    baseline_norm = pd.read_pickle(os.path.join(baseline_dir, "norm.pkl"))
    norm(df, baseline_norm, excluded_from_norm, skip_conditions=skip_conditions, run_agg=run_agg)