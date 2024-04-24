from analysis_util import *
from scipy.stats import wilcoxon

def mean_confidence_interval(data, confidence=0.95):
    from scipy import stats
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def get_stats_by_batch(df, batch1, batch2, var):
    from scipy.stats import wilcoxon
    print(var, batch1, batch2)
    batch1_df = df[df['batch']==batch1]
    batch2_df = df[df['batch']==batch2]
    batch1_losses = batch1_df[var].values
    batch2_losses = batch2_df[var].values
    print('mean', batch1_losses.mean(), batch2_losses.mean())
    print('std', batch1_losses.std(), batch2_losses.std())
    
    if len(batch1_losses) != len(batch2_losses):
        print("Warning: different number of samples for batches", batch1, batch2)
        if len(batch1_losses) < len(batch2_losses):
            batch2_losses = batch2_losses[:len(batch1_losses)]
        else:
            batch1_losses = batch1_losses[:len(batch2_losses)]

    ci_1 = mean_confidence_interval(batch1_losses)
    ci_2 = mean_confidence_interval(batch2_losses)
    
    print('95% CI ', batch1, batch1_losses.mean() - ci_1, '-', batch1_losses.mean() + ci_1, f'${batch1_losses.mean()} \pm ' + str(ci_1) + '$')
    print('95% CI ', batch2, batch2_losses.mean() - ci_2, '-', batch2_losses.mean() + ci_2, f'${batch2_losses.mean()} \pm ' + str(ci_2) + '$')
    
    return wilcoxon(batch1_losses, batch2_losses)


def get_stats_by_condition(df, condition_col_name, var):

 

    conditions = df[condition_col_name].unique()

    # pair-wise comparisons
    means = {}
    stds = {}
    cis = {}
    confidence = {}
    p_values = {}

    for conditionA in conditions:
        means[conditionA] = df[df[condition_col_name]==conditionA][var].mean()
        stds[conditionA] = df[df[condition_col_name]==conditionA][var].std()
        ci_1 = mean_confidence_interval(df[df[condition_col_name]==conditionA][var].values)
        cis[conditionA] = (means[conditionA] - ci_1, means[conditionA] + ci_1)
        confidence[conditionA] = ci_1

        for conditionB in conditions:
            if conditionA == conditionB:
                continue
            conditionA_df = df[df[condition_col_name]==conditionA]
            conditionB_df = df[df[condition_col_name]==conditionB]

            conditionA_vals = conditionA_df[var].values
            conditionB_vals = conditionB_df[var].values
            if len(conditionA_vals) != len(conditionB_vals):
                print("Warning: different number of samples for conditions", conditionA, conditionB)
                if len(conditionA_vals) < len(conditionB_vals):
                    conditionB_vals = conditionB_vals[:len(conditionA_vals)]
                else:
                    conditionA_vals = conditionA_vals[:len(conditionB_vals)]
            p_values[(conditionA, conditionB)] = wilcoxon(conditionA_vals, conditionB_vals).pvalue

    return means, stds, cis, p_values, confidence


def compare_dfs(dfA, dfB, var):
    means = {}
    stds = {}
    cis = {}
    confidence = {}
    p_values = {}
    means['A'] = dfA[var].mean()
    means['B'] = dfB[var].mean()
    stds['A'] = dfA[var].std()
    stds['B'] = dfB[var].std()
    ci_1 = mean_confidence_interval(dfA[var].values)
    ci_2 = mean_confidence_interval(dfB[var].values)
    cis['A'] = (means['A'] - ci_1, means['A'] + ci_1)
    cis['B'] = (means['B'] - ci_2, means['B'] + ci_2)
    confidence['A'] = ci_1
    confidence['B'] = ci_2
    p_values['A vs B'] = wilcoxon(dfA[var].values, dfB[var].values).pvalue
    return means, stds, cis, p_values, confidence


    


def print_stats_by_condition(df, condition_col_name, var, save_path=None):
    means, stds, cis, p_values, confidence = get_stats_by_condition(df, condition_col_name, var)
    print("Means")
    for k, v in means.items():
        print("\t", str(k)+":", v)

    print("Stds")
    for k, v in stds.items():
        print("\t", str(k)+":", v)

    print("CIs")
    for k, v in cis.items():
        print("\t", str(k)+":", v)

    print("Confidence")
    for k, v in confidence.items():
        print("\t", str(k)+":", v)

    print("P Values")
    for k, v in p_values.items():
        print(f"\t{k[0]} vs {k[1]}: {v} {'*'if v < 0.05 else ''}")


    if save_path is None:
        return means, stds, cis, p_values, confidence
    # save as tex table
    strng = "mean & lower & upper \\\\ \n"
    for k, v in means.items():
        strng += f"{k} & {cis[k][0]:.3f} & {cis[k][1]:.3f} \\\\ \n"
        
    with open(os.path.join(save_path, "means.tex"), "w") as f:
        f.write(strng)
    print(strng)

    strng = "A & B & p \\\\ \n"
    for k, v in p_values.items():
        strng += f"{k[0]} & {k[1]} & {v:.3f} \\\\ \n"

    with open(os.path.join(save_path, "p_values.tex"), "w") as f:
        f.write(strng)

    print(strng)
    return means, stds, cis, p_values, confidence
