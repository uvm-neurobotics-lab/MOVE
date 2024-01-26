from plot import *
import os
import sys

sys.path.append(os.path.abspath('../'))
from analysis_util import *

from norm import read_norm_data
from sgd_weights import sgd_weights

config=None
inputs=None
target=None
norm=None

import torch

def get_sgd_loss(cppn, config=config, inputs=inputs,target=target, norm=norm, num_steps=500):
    losses = torch.zeros(num_steps)
    sgd_weights([cppn], None, inputs, target, config.objective_functions, norm, config, None, losses)
    return losses

def run_one(run_path, cond_dir, device, modulo, num_steps):
    print(run_path, cond_dir)
    this_data =  pd.DataFrame(columns=['run', 'batch', 'loss'])
    try:
        genomes_by_batch, config = get_cppns_by_batch(run_path, modulo, always_include_last=True, override_batch='mid')
    except Exception as e:
        print(e)
        return pd.DataFrame()
    config.sgd_steps = num_steps
    if not os.path.exists(os.path.join(run_path, 'inputs.pt')):
        print('no inputs')
        return pd.DataFrame()
    inputs = load_inputs(run_path, device)
    target = load_target(run_path, config)
    norm = read_norm_data(os.path.join("../", config.norm_df_path), config.target_name)
    
    # apply mutation
    set_max_node_id(genomes_by_batch)
    
    
    all_losses = torch.ones(len(genomes_by_batch), num_steps) * -torch.inf
    batches = []
    for index, (batch, cppn) in enumerate(genomes_by_batch.items()):
        cppn.mutate(config)
        all_losses[index] = get_sgd_loss(cppn, config, inputs, target, norm, num_steps)
        batches.append(batch)
    data = []
    for index, (batch, cppn) in enumerate(genomes_by_batch.items()):
        batch_name = batch
        # if batch == max(genomes_by_batch.keys()):
            # batch_name = 'last'
        # elif batch == min(genomes_by_batch.keys()):
            # batch_name = 'first'
        # else:
        batch_name = 'mid'
        run_dir = os.path.basename(os.path.normpath(run_path))
        this_data = (pd.DataFrame({'run': run_dir, 'condition_path': cond_dir, 'target':config.target_name, 'batch': batch_name, 'step': torch.arange(num_steps), 'loss': all_losses[index].cpu().numpy()}))
        data.append(this_data)
    return  pd.concat(data)


def full_experiment(experiment_path, num_steps=100, modulo=500):
    all_results = pd.DataFrame(columns=['run', 'batch', 'loss'])
    cond_path = os.path.join(experiment_path, 'conditions')
    # cond_dirs = [cond_dir for cond_dir in os.listdir(cond_path) if os.path.isdir(os.path.join(cond_path, cond_dir))]
    cond_dirs = ['100_steps_']
    run_paths = []
    for cond_dir in cond_dirs:
        run_paths += [os.path.join(cond_path, cond_dir, run_dir) for run_dir in os.listdir(os.path.join(cond_path, cond_dir)) if os.path.isdir(os.path.join(cond_path, cond_dir, run_dir))]
        # run_paths += [os.path.join(cond_path, cond_dir, "run_373404468125110766")]
        # run_paths += [os.path.join(cond_path, cond_dir, "run_1255773183754310126")]
    
    MAX_RUNS = 4
    run_paths = run_paths[:MAX_RUNS]
    
    jobs = [(run_path, cond_dir, device, modulo, num_steps) for cond_dir in cond_dirs for run_path in run_paths]
    print(len(jobs), 'jobs')
    print(jobs)
    
    mp_return = []
    # for job in tqdm(jobs):
        # mp_return.append(run_one(*job))
    
    pool = mp.Pool(mp.cpu_count()-1)
    mp_return = pool.starmap(run_one, jobs)
    
    
    all_results = pd.concat(mp_return)
    
    return all_results

from tqdm import tqdm
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

def mean_confidence_interval(data, confidence=0.95):
    from scipy import stats
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def stats(df, batch1, batch2, var):
    from scipy.stats import wilcoxon
    print(var, batch1, batch2)
    batch1_df = df[df['batch']==batch1]
    batch2_df = df[df['batch']==batch2]
    batch1_losses = batch1_df[var].values
    batch2_losses = batch2_df[var].values
    print('mean', batch1_losses.mean(), batch2_losses.mean())
    print('std', batch1_losses.std(), batch2_losses.std())
    
    ci_1 = mean_confidence_interval(batch1_losses)
    ci_2 = mean_confidence_interval(batch2_losses)
    
    print('95% CI ', batch1, batch1_losses.mean() - ci_1, '-', batch1_losses.mean() + ci_1, f'${batch1_losses.mean()} \pm ' + str(ci_1) + '$')
    print('95% CI ', batch2, batch2_losses.mean() - ci_2, '-', batch2_losses.mean() + ci_2, f'${batch2_losses.mean()} \pm ' + str(ci_2) + '$')
    
    return wilcoxon(batch1_losses, batch2_losses)

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # results = full_experiment('../../results/sgd-steps', num_steps=500, modulo=10000)

    # results.to_csv('sgd-steps-inner-loop.csv')
    results = pd.read_csv('100-steps-inner-loop-all.csv')
    
    max_step = results.groupby(['run', 'condition_path', 'batch']).max().reset_index()
    min_step = results.groupby(['run', 'condition_path', 'batch']).min().reset_index()
    
    max_step['step'] = max_step['step'].astype(int)
    min_step['step'] = min_step['step'].astype(int)
    
    diff_first_step_last_step = pd.merge(max_step, min_step, on=['run', 'condition_path', 'batch'], suffixes=('_max', '_min'))
    
    diff_first_step_last_step['loss_diff'] = diff_first_step_last_step['loss_max'] - diff_first_step_last_step['loss_min']
    diff_first_step_last_step['loss_diff'] = diff_first_step_last_step['loss_diff'].abs()    
    
    print(diff_first_step_last_step)
    sns.barplot(data=diff_first_step_last_step, x='condition_path', y='loss_diff', hue='batch')
    plt.savefig('sgd-steps-inner-loop-diff.png')
    
    print(stats(diff_first_step_last_step, 'first', 'last', 'loss_diff'))
    exit()
    
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.lineplot(data=results, x='step', y='loss', hue='condition_path', legend='full', style='batch')
    plt.savefig('sgd-steps-inner-loop.png')