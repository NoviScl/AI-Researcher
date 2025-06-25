import json 
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# Load data from a JSON file
with open('data_points_all_anonymized.json', 'r') as f:
    data = json.load(f)

# Convert JSON data to a pandas DataFrame
df = pd.DataFrame(data)

# Separate DataFrames based on the condition
df_ai = df[df['condition'] == 'AI']
df_human = df[df['condition'] == 'Human']
df_human_ai = df[df['condition'] == 'AI_Rerank']

# Define the metrics to test
metrics = ['overall_score', 'novelty_score', 'feasibility_score', 'effectiveness_score', 'excitement_score']

# Initialize a dictionary to store results
results = {}

# Perform independent t-tests for each metric
for metric in metrics:
    t_stat_ai, p_val_ai = stats.ttest_ind(df_ai[metric], df_human[metric], equal_var=False)
    t_stat_human_ai, p_val_human_ai = stats.ttest_ind(df_human_ai[metric], df_human[metric], equal_var=False)
    results[metric] = {
        'AI vs Human': (t_stat_ai, p_val_ai),
        'Human+AI vs Human': (t_stat_human_ai, p_val_human_ai)
    }

# Apply Bonferroni correction or another method for multiple comparisons
p_vals_ai = [results[metric]['AI vs Human'][1] for metric in metrics]
p_vals_human_ai = [results[metric]['Human+AI vs Human'][1] for metric in metrics]

# Adjust p-values
adjusted_p_vals_ai = multipletests(p_vals_ai, method='bonferroni')[1]
adjusted_p_vals_human_ai = multipletests(p_vals_human_ai, method='bonferroni')[1]

# Store both unadjusted and adjusted p-values
for i, metric in enumerate(metrics):
    results[metric]['AI vs Human'] = (results[metric]['AI vs Human'][0], p_vals_ai[i], adjusted_p_vals_ai[i])
    results[metric]['Human+AI vs Human'] = (results[metric]['Human+AI vs Human'][0], p_vals_human_ai[i], adjusted_p_vals_human_ai[i])

# When printing results:
for k,v in results.items():
    print(k)
    print("Human: N=", len(df_human[k]), " Mean=", np.mean(df_human[k]), " Median=", np.median(df_human[k]), " SD=", np.std(df_human[k]), " SE=", np.std(df_human[k] / np.sqrt(len(df_human[k]))), " Min=", np.min(df_human[k]), " Max=", np.max(df_human[k]))
    print("AI: N=", len(df_ai[k]), " Mean=", np.mean(df_ai[k]), " Median=", np.median(df_ai[k]), " SD=", np.std(df_ai[k]), " SE=", np.std(df_ai[k] / np.sqrt(len(df_ai[k]))), " Min=", np.min(df_ai[k]), " Max=", np.max(df_ai[k]))
    print("AI + Human: N=", len(df_human_ai[k]), " Mean=", np.mean(df_human_ai[k]), " Median=", np.median(df_human_ai[k]), " SD=", np.std(df_human_ai[k]), " SE=", np.std(df_human_ai[k] / np.sqrt(len(df_human_ai[k]))), " Min=", np.min(df_human_ai[k]), " Max=", np.max(df_human_ai[k]))
    for k2, v2 in v.items():
        print(k2, "t-statistic:", v2[0], "p-value (unadjusted):", v2[1], "p-value (adjusted):", v2[2])
    print()

