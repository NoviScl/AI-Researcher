import json 
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# Load data from a JSON file
with open('data_points_all_execution.json', 'r') as f:
    data = json.load(f)

# Convert JSON data to a pandas DataFrame
df = pd.DataFrame(data)

# Separate DataFrames based on the condition
df_ai = df[df['condition'] == 'AI']
df_human = df[df['condition'] == 'Human']

# Define the metrics to test
metrics = ['novelty_score', 'excitement_score', 'soundness_score', 'effectiveness_score', 'overall_score']


# Initialize a dictionary to store results
results = {}

# Perform independent t-tests for each metric
for metric in metrics:
    t_stat_ai, p_val_ai = stats.ttest_ind(df_ai[metric], df_human[metric], equal_var=False, alternative='less')
    results[metric] = {
        'AI vs Human': (t_stat_ai, p_val_ai),
    }

p_vals_ai = [results[metric]['AI vs Human'][1] for metric in metrics]

# Adjust p-values
adjusted_p_vals_ai = multipletests(p_vals_ai, method='fdr_bh')[1]

# Store both unadjusted and adjusted p-values
for i, metric in enumerate(metrics):
    results[metric]['AI vs Human'] = (results[metric]['AI vs Human'][0], p_vals_ai[i], adjusted_p_vals_ai[i])

# When printing results:
for k,v in results.items():
    print(k)
    human_mean = np.mean(df_human[k])
    human_std = np.std(df_human[k])
    human_n = len(df_human[k])
    human_ci_95 = stats.t.ppf(0.975, human_n-1) * (human_std / np.sqrt(human_n))
    
    ai_mean = np.mean(df_ai[k])
    ai_std = np.std(df_ai[k])
    ai_n = len(df_ai[k])
    ai_ci_95 = stats.t.ppf(0.975, ai_n-1) * (ai_std / np.sqrt(ai_n))
    
    print("Human: N=", human_n, " Mean=", round(human_mean, 2), " Median=", round(np.median(df_human[k]), 2), 
          " SD=", round(human_std, 2), " 95% CI=[", round(human_mean-human_ci_95, 2), ",", round(human_mean+human_ci_95, 2), "]",
          " Min=", round(np.min(df_human[k]), 2), " Max=", round(np.max(df_human[k]), 2))
    print("AI: N=", ai_n, " Mean=", round(ai_mean, 2), " Median=", round(np.median(df_ai[k]), 2), 
          " SD=", round(ai_std, 2), " 95% CI=[", round(ai_mean-ai_ci_95, 2), ",", round(ai_mean+ai_ci_95, 2), "]",
          " Min=", round(np.min(df_ai[k]), 2), " Max=", round(np.max(df_ai[k]), 2))
    
    for k2, v2 in v.items():
        print(k2, "t-statistic:", round(v2[0], 2), "p-value (unadjusted):", round(v2[1], 2), "p-value (adjusted):", round(v2[2], 2))
    print()
