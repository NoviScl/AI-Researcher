import json 
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns


with open("../reviews_ideation/data_points_all_anonymized.json", "r") as f:
    data_study1 = json.load(f)

with open('data_points_all_execution.json', 'r') as f:
    data_study2 = json.load(f)

## whether to exclude the 6 ideas the involved human eval
exclude_idea_ids = ["Uncertainty_2_AI", "Multilingual_5_AI", "Bias_4_AI", "Factuality_8_AI", "Uncertainty_6_AI", "Multilingual_8_AI"]
exclude_auto_eval = False

df_study1 = pd.DataFrame(data_study1)
df_study2 = pd.DataFrame(data_study2)

if exclude_auto_eval:
    df_study1 = df_study1[~df_study1['idea_id'].isin(exclude_idea_ids)]
    df_study2 = df_study2[~df_study2['idea_id'].isin(exclude_idea_ids)]

# Get the set of idea_ids that appear in both datasets
common_idea_ids = set(df_study1['idea_id']).intersection(set(df_study2['idea_id']))

# Print the number of unique idea_ids that appear in both datasets
print(f"Number of unique idea_ids in both datasets: {len(common_idea_ids)}")

# Filter both dataframes to only include data points with idea_ids that appear in both datasets
df_study2 = df_study2[df_study2['idea_id'].isin(common_idea_ids)]
df_study1 = df_study1[df_study1['idea_id'].isin(common_idea_ids)]

# Print the number of data points in each df
print(f"Number of data points in df_study1: {len(df_study1)}")
print(f"Number of data points in df_study2: {len(df_study2)}")

metrics = ['novelty_score', 'excitement_score', 'effectiveness_score', 'overall_score']

# ---- 1. Collapse reviewer scores to per‑response means *within each study* ----
study1_mean = (df_study1
            .groupby(['idea_id', 'condition'])[metrics]
            .mean()
            .add_suffix('_s1'))           # columns like novelty_score_s1 …

study2_mean = (df_study2
            .groupby(['idea_id', 'condition'])[metrics]
            .mean()
            .add_suffix('_s2'))

# Merge on (response_id, condition) so rows stay aligned
merged = study1_mean.join(study2_mean, how='inner')

# ---- 2. Compute change scores d_r = μ_r,2 − μ_r,1 for every metric ------------
for m in metrics:
    merged[f'{m}_change'] = merged[f'{m}_s2'] - merged[f'{m}_s1']

# ---- 3. Welch t‑tests on change scores, Human vs AI --------------------------
results = []

for m in metrics:
    # Explicitly compute s2 - s1 scores for each condition
    human_s1 = merged.loc[merged.index.get_level_values('condition') == 'Human', f'{m}_s1']
    human_s2 = merged.loc[merged.index.get_level_values('condition') == 'Human', f'{m}_s2']
    d_h = human_s2 - human_s1  # Human change scores (s2 - s1)
    
    ai_s1 = merged.loc[merged.index.get_level_values('condition') == 'AI', f'{m}_s1']
    ai_s2 = merged.loc[merged.index.get_level_values('condition') == 'AI', f'{m}_s2']
    d_a = ai_s2 - ai_s1  # AI change scores (s2 - s1)

    t, p_two = stats.ttest_ind(d_h, d_a, equal_var=False, nan_policy='omit')
    p_one = p_two/2 if t > 0 else 1-p_two/2      # one‑tailed

    results.append({'metric'        : m,
                    'human_mean'    : d_h.mean(),
                    'human_std'     : d_h.std(),
                    'human_n'       : len(d_h),
                    'ai_mean'       : d_a.mean(),
                    'ai_std'        : d_a.std(),
                    'ai_n'          : len(d_a),
                    'Δd'            : d_h.mean() - d_a.mean(),
                    't_stat'        : t,
                    'p_uncorrected' : p_one})


pvals = [r['p_uncorrected'] for r in results]
reject, p_corr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

for r, p_adj, rej in zip(results, p_corr, reject):
    r['p_FDR'] = p_adj
    r['sig']   = 'yes' if rej else 'no'

summary = (pd.DataFrame(results)
           .sort_values('p_uncorrected')
           .reset_index(drop=True))

print("\n===== Change‑score Welch t‑tests (Human vs AI) =====")
print(summary)

# Print detailed statistics for each metric
print("\n===== Detailed Change Score Statistics by Metric =====")
for m in metrics:
    metric_data = next(r for r in results if r['metric'] == m)
    print(f"\n{m}:")
    print(f"  Human: mean = {metric_data['human_mean']:.4f}, std = {metric_data['human_std']:.4f}, n = {metric_data['human_n']}")
    print(f"  AI: mean = {metric_data['ai_mean']:.4f}, std = {metric_data['ai_std']:.4f}, n = {metric_data['ai_n']}")
    print(f"  Difference (Human - AI): {metric_data['Δd']:.4f}")
    print(f"  p-value (uncorrected): {metric_data['p_uncorrected']:.4f}")
    print(f"  p-value (FDR corrected): {metric_data['p_FDR']:.4f}")
    print(f"  Significant: {metric_data['sig']}")