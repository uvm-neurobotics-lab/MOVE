from analysis_util import *

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
    
    ci_1 = mean_confidence_interval(batch1_losses)
    ci_2 = mean_confidence_interval(batch2_losses)
    
    print('95% CI ', batch1, batch1_losses.mean() - ci_1, '-', batch1_losses.mean() + ci_1, f'${batch1_losses.mean()} \pm ' + str(ci_1) + '$')
    print('95% CI ', batch2, batch2_losses.mean() - ci_2, '-', batch2_losses.mean() + ci_2, f'${batch2_losses.mean()} \pm ' + str(ci_2) + '$')
    
    return wilcoxon(batch1_losses, batch2_losses)


def get_stats_by_condition(df, condition_col_name, var):
    from scipy.stats import wilcoxon

 

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
            print(len(conditionA_vals), len(conditionB_vals))
            if len(conditionA_vals) != len(conditionB_vals):
                print("Warning: different number of samples for conditions", conditionA, conditionB)
                if len(conditionA_vals) < len(conditionB_vals):
                    conditionB_vals = conditionB_vals[:len(conditionA_vals)]
                else:
                    conditionA_vals = conditionA_vals[:len(conditionB_vals)]
            p_values[(conditionA, conditionB)] = wilcoxon(conditionA_vals, conditionB_vals).pvalue

    return means, stds, cis, p_values, confidence



    
