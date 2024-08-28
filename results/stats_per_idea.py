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

# Define the metrics to test
metrics = ['overall_score', 'novelty_score', 'feasibility_score', 'effectiveness_score', 'excitement_score']

# Select only the columns relevant for averaging (i.e., metrics and grouping columns)
columns_to_average = ['idea_id', 'condition'] + metrics  # Include 'idea_id', 'condition', and the metrics
df_numeric = df[columns_to_average]

# Group by idea and condition, and calculate the mean for each metric
df_grouped = df_numeric.groupby(['idea_id', 'condition']).mean().reset_index()

# Separate DataFrames based on the condition
df_ai = df_grouped[df_grouped['condition'] == 'AI']
df_human = df_grouped[df_grouped['condition'] == 'Human']
df_human_ai = df_grouped[df_grouped['condition'] == 'AI_Rerank']

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
for metric in metrics:
    print(metric)
    print("Human: N=", len(df_human[metric]), " Mean=", np.mean(df_human[metric]), " Median=", np.median(df_human[metric]),  " SD=", np.std(df_human[metric]), " SE=", np.std(df_human[metric] / np.sqrt(len(df_human[metric]))), " Min=", np.min(df_human[metric]), " Max=", np.max(df_human[metric]))
    print("AI: N=", len(df_ai[metric]), " Mean=", np.mean(df_ai[metric]), " Median=", np.median(df_ai[metric]), " SD=", np.std(df_ai[metric]), " SE=", np.std(df_ai[metric] / np.sqrt(len(df_ai[metric]))), " Min=", np.min(df_ai[metric]), " Max=", np.max(df_ai[metric]))
    print("AI + Human: N=", len(df_human_ai[metric]), " Mean=", np.mean(df_human_ai[metric]), " Median=", np.median(df_human_ai[metric]), " SD=", np.std(df_human_ai[metric]), " SE=", np.std(df_human_ai[metric] / np.sqrt(len(df_human_ai[metric]))), " Min=", np.min(df_human_ai[metric]), " Max=", np.max(df_human_ai[metric]))
    for condition, values in results[metric].items():
        print(condition, "t-statistic:", values[0], "p-value (unadjusted):", values[1], "p-value (adjusted):", values[2])
    print()
