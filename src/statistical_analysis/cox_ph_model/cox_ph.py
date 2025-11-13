import os
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from sklearn.preprocessing import OneHotEncoder
from lifelines.utils import concordance_index
import glob
from itertools import combinations
from sklearn.utils import resample
from sklearn.model_selection import KFold
from lifelines.statistics import multivariate_logrank_test
from sksurv.metrics import cumulative_dynamic_auc
from statsmodels.stats.multitest import multipletests

rs_cols = ['max_betweenness_scaled_q',
            'laplace_mean_scaled_q',
            'burden_gt_tau__3-4mm_scaled_q', 
            'patch_density_scaled_q',
            'avg_prob_in_region_scaled_q',
            'avg_shortest_path_scaled_q',
            'rs_edge_core_ratio_scaled',
             'rs_distance_entropy_scaled',
             'rs_lograte_slope_scaled',
             'rs_half_distance_scaled',
             'rs_total_mass_scaled',
             'rs_mean_rate_over_bins_scaled']

tumor_cols = ['ratio_of_cancer_to_tissue_scaled_q',
    'stroma_to_epithelial_ratio_scaled_q', 'region_axis_major_len_scaled_q',
     'region_eccentricity_scaled_q',
     'region_perimeter_scaled_q',
     'region_solidity_scaled_q', 'gleason_total', 'multifocal', 'tertiary_pattern']

time_col = 'time_to_bcr'
event_col = 'label'

capra_col = ['capra']

models = {
    "RS_only": rs_cols,
    "RS_plus_tumor": rs_cols + tumor_cols,
    "RS_plus_capra": rs_cols + capra_col,
    "Capra_only": capra_col,
    "Tumor_only": tumor_cols,
    'All': ['max_betweenness_scaled_q',
        'rs_pos_high_percent_scaled_q',
        'patch_density_scaled_q',
        'laplace_mean_scaled_q', 'clustering_coeff_scaled_q', 
        'rs_edge_core_ratio_scaled',
        'rs_com_distance_scaled',
        'rs_distance_entropy_scaled',
        'rs_lograte_slope_scaled',
        'rs_half_distance_scaled',
        'rs_total_mass_scaled',
        'rs_mean_rate_over_bins_scaled',
        'rs_max_rate_over_bins_scaled','q95_rs__1-2mm_scaled', 'ratio_of_cancer_to_tissue_scaled_q',
        'stroma_to_epithelial_ratio_scaled_q', 'region_axis_major_len_scaled_q',
        'region_eccentricity_scaled_q',
        'region_perimeter_scaled_q',
        'region_solidity_scaled_q', 'multifocal', 'tertiary_pattern'] + capra_col,
    'Tumor_plus_capra': ['ratio_of_cancer_to_tissue_scaled_q',
        'stroma_to_epithelial_ratio_scaled_q', 'region_axis_major_len_scaled_q',
        'region_eccentricity_scaled_q',
        'region_perimeter_scaled_q',
        'region_solidity_scaled_q', 'multifocal', 'tertiary_pattern'] + capra_col
    }

def enhanced_fit_cph(
    df, 
    features, 
    time_col, 
    event_col, 
    bootstrap = True, 
    n_boostraps = 1000
):
    categorical_vars = []
    continuous_vars = []
    for col in features:
        if '_q' in col:
            categorical_vars.append(col)
        else:
            continous_vars.append(col)

    if len(categorical_vars) > 0:
        dummy_vars = pd.get_dummies(df[categorical_vars], drop_first = True)
        all_continuous_vars = [time_col, event_col] + continuous_vars

        df_model = pd.concat([dataset[all_continous_vars], dummy_vars], axis = 1)
    else:
        df_model = df[[time_col, event_col] + features].dropna()

    cph = CoxPHFitter()
    cph.fit(df_model, duration_col = time_col, event_col = event_col)
    
    summary_df = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    summary_df.columns = ["HR", "HR_lower95", "HR_upper95", "p_value"]

    bootstrap_results = None
    if bootstrap:
        predictions = cph.predict_partial_hazard(df_model)
        bootstrap_results = bootstrap_individual_model(
            df_model[time_col].values,
            df_model[event_col].values,
            predictions.values,
            n_bootstraps = n_boostraps
        )

    return cph, summary_df, df_model, bootstrap_results


def bootstrap_individual_model(
    y_time, 
    y_event, 
    predictions, 
    times_auc = (12, 24, 60), 
    n_bootstraps = 1000,
    random_state = 24
    ):
    df = pd.DataFrame({
        'time': pd.to_numeric(y_time, errors = 'coerce').astype(float),
        'event': pd.Series(y_event).astype(bool),
        'pred': pd.to_numeric(predictions, errors = 'coerce').astype(float)
    }).dropna()

    y_train = np.array(
        list(
            zip(df['event'].values, df['time'].values)), 
            dtype = [('event', '?'), ('time', '<f8')])

    valid_times = np.array(times_auc)
    c_indices = []
    auc_results = {time: [] for time in valid_times}

    for i in tqdm(range(n_bootstraps), desc = 'Bootstrapping'):
        sample = resample(df, replace = True, n_samples = len(df))
        c_idx = concordance_index(sample['time'], -sample['pred'], sample['event'])
        c_indices.append(c_idx)

        if len(valid_times) > 0:
            y_test = np.array(
                list(zip(sample['event'].values,
                sample['time'].values)),
                dtype = [('event', '?'), ('time', '<f8')]
            )
            for time in valid_times:
                _, auc_val = cumulative_dynamic_auc(y_train, y_test, sample['pred_values'], np.array([time]))
                if np.isfinite(auc_val):
                    auc_results[time].append(auc_val)

    results = {
        'c_index': {
            'mean': np.mean(c_indices),
            'ci_lower': np.percentile(c_indices, 2.5),
            'ci_upper': np.percentile(c_indices, 97.5),
            'values': c_indices
        }
    }

    for time in valid_times:
        if len(auc_results[time]) > 0:
            results[f'auc_{int(time)}'] = {
                'mean': np.mean(auc_results[time]),
                'ci_lower': np.percentile(auc_results[time], 2.5),
                'ci_upper': np.percentile(auc_results[time], 97.5),
                'values': auc_results[time]
            }
    
    return results

def summarize(values):
    values = np.asarray(values).astype(float)
    low, high = np.percentile(values, [2.5, 97.5]) if values.size else (np.nan, np.nan)
    return {'mean': float(values.mean()), '95%_CI': (float(low), float(high))}

def summarize_diff(differences):
    differences = np.asarray(differences).astype(float)
    if differences.size == 0:
        return {
            'mean_diff': np.nan,
            '95%_CI': (np.nan, np.nan),
            'p_value': np.nan
        }
    low, high = np.percentile(differences, [2.5, 97.5])
    pvalue = 2 * min((differences <= 0).mean(), (differences >= 0).mean())
    return {
        'mean_diff': float(differences.mean()),
        '95%_CI': (float(low), float(high)),
        'p_value': float(pval)
    }

def auc_scores(score, score_higher_risk):
    return s if score_higher_risk else -s

def cidx_scores(score, cindex_negate):
    return -s if cindex_negate else s


def bootstrap_between_models(
    y_time, 
    y_event,
    preds1, 
    preds2, 
    times_auc = (24, 36, 60),
    n_boostraps = 1000,
    random_state = 24,
    score_higher_risk = True,
    cindex_negate = True,
    per_horizon = True
):
    random_s = np.random.Random(random_state)
    df = pd.DataFrame({
        'time': pd.to_numeric(y_time, errors = 'coerce').astype(float),
        'event': pd.Series(y_event).astype(bool),
        'pred1': pd.to_numeric(preds1, errors = 'coerce').astype(float),
        'pred2': pd.to_numeric(preds2, errors = 'coerce').astype(float),
    }).replace([np.inf, -np.inf], np.nan).dropna(subset = ['time', 'event', 'pred1', 'pred2']).copy()

    y_train = np.array(
        list(zip(df['event'].values, 
        df['time'].values)),
        dtype = [('event', '?'), ('time', '<f8')]
    )

    times_arr = np.array(times_auc).astype(float)

    if df['pred1'].var() == 0 and df['pred2'].var() == 0:
        raise ValueError('Predictions are constant for at least one model so AUC undefined')
    
    c1_values, c2_values, c_differences = [], [], []
    
    auc1_values = {float(time): [] for time in times_arr} if per_horizon else None
    auc2_values = {float(time): [] for time in times_arr} in per_horizon else None
    auc_differences = {float(time): [] for time in times_arr} if per_horizon else None

    for _ in tqdm(range(n_bootstraps)):
        sample = resample(df, replace = True, n_samples = len(df), random_state = random_s)
        y_test = np.array(list(zip(
            sample['event'].values, sample['time'].values
            )), dtype = [('event', '?'), ('time', '<f8')]
            )
        
        c1 = concordance_index(
            sample['time'].values,
            cidx_scores(sample['pred1'].values, cindex_negate),
            sample['event'].values
        )
        c2 = concordance_index(
            sample['time'].values,
            cidx_scores(sample['pred2'].values, cindex_negate),
            sample['event'].values
        )
        c1_values.append(c1)
        c2_values.append(c2)
        c_differences.append(c2-c1)

        if per_horizon:
            for time in times_arr:
                _, a1 = cumulative_dynamic_auc(
                    y_train, 
                    y_test, 
                    auc_scores(sample['pred1'].values, score_higher_risk), 
                    np.array([time]))
                _, a2 = cumulative_dynamic_auc(
                    y_train, 
                    y_test, 
                    auc_scores(sample['pred2'].values, score_higher_risk), 
                    np.array([time]))
                if np.isfinite(a1) and np.isfinite(a2):
                    auc1_values[float(time)].append(a1)
                    auc2_values[float(time)].append(a2)
                    auc_differences[float(time)].append(a2 - a1)
    
    results = {
        'C-index Model1': summarize(c1_values),
        'C-index Model2': summarize(c2_values),
        'C-index (Model2 - Model1)': summarize_diff(c_differences)
    }
    if per_horizon:
        for time in times_arr:
            label = f"AUC@{int(round(time))}"
            results[f'{label} Model1'] = summarize(auc1_values[float(time)])
            results[f'{label} Model2'] = summarize(auc2_values[float(time)])
            results[f'{label} (Model1 - Model2)'] = summarize_diff(auc_differences[float(time)])

    return results

def results_to_abs_diff_dfs(results_boot):
    all_abs, all_diff = [], []
    for pair_key, metrics_dict in results_boot.items():
        pair_tuple = metrics_dict.get("pair", ("", ""))
        abs_rows, diff_rows = flatten_result_rows(pair_key, pair_tuple, metrics_dict)
        all_abs.extend(abs_rows)
        all_diff.extend(diff_rows)

    abs_df = pd.DataFrame(all_abs)
    diff_df = pd.DataFrame(all_diff)

    def clean_metric_name(name):
        return re.sub(f"\s*Model\s*[12]\s*$", "", name, flags = re.I)

    abs_df['metric_base'] = abs_df['metric'].apply(clean_metric_name)
    abs_df = abs_df[[
        'pair_key', 'model1', 'model2', 'metric', 'metric_base', 
        'which_model', 'mean', 'ci_lower', 'ci_upper'
    ]]

    diff_df = diff_df[[
        'pair_key', 'model1', 'model2', 'metric', 'metric_base', 
        'which_model', 'mean', 'ci_lower', 'ci_upper'
    ]]

    return abs_df, diff_df

def adjust_pvalues(diff_df, method = 'fdr_bh'):
    df = diff_df.copy()
    mask = df['p_value'].notna()
    reject, pvalues_adj, _, _ = multipletests(df.loc[mask, 'p_value'], method = method)
    df.loc[mask, 'p_value_adj'] = pvalues_adj
    df.loc[mask, 'reject_H0'] = reject

    return df

def get_cross_model_results(dataset, times_auc):
    pairs = list(combinations(models, 2))
    results_boot = {}
    for pair in pairs:
        res_boot = bootstrap_between_models(
            y_time = dataset['time_to_bcr'], 
            y_event = dataset['label'],
            preds1 = dataset[f'{pair[0]}'], 
            preds2 = dataset[f'{pair[1]}'], 
            times_auc = [24, 36, 60],
            n_boostraps = 5000,
            random_state = 24,
            score_higher_risk = True,
            cindex_negate = True,
            per_horizon = True
        )
        res_boot['pair'] = pair
        results_boot[f"{pair[0]}+{pair[1]}"] = res_boot

    abs_diff, diff_df = results_to_abs_diff_dfs(results_boot)
    diff_df_adj = adjust_pvalues(diff_df, method = 'fdr_bh')

    return diff_df_adj



def main(args):
    dataset = pd.read_csv(args.dataset, index_col = 0)

    for col in dataset.columns:
        if '_q' in col:
            dataset[col] = dataset[col].astype(object)
        elif col in ['gleason_total', 'multifocal', 'tertiary_pattern']:
            dataset[col] = dataset[col].astype(object)


if __name__ == "__main__":

    main()