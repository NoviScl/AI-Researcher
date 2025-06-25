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

# List of conditions for comparison
comparisons = [('AI', 'Human'), ('AI_Rerank', 'Human')]

# Iterate over each metric
for metric in metrics:
    metric_results = {}
    
    # Iterate over each comparison
    for cond1, cond2 in comparisons:
        # Initialize a list to store differences for each reviewer
        differences = []
        
        # Get unique names (reviewers)
        names = df['name'].unique()
        
        # Calculate differences for each name
        for name in names:
            # Filter data for the current name and conditions
            score_cond1 = df[(df['name'] == name) & (df['condition'] == cond1)][metric].mean()
            score_cond2 = df[(df['name'] == name) & (df['condition'] == cond2)][metric].mean()
            
            # Check if the reviewer has scores for both conditions
            if not np.isnan(score_cond1) and not np.isnan(score_cond2):
                # Calculate the difference
                difference = score_cond1 - score_cond2
                differences.append(difference)
        
        # Perform a one-sample t-test to check if the differences are significantly different from 0
        if differences:  # Ensure there are differences to test
            t_stat, p_value = stats.ttest_1samp(differences, 0)
        else:
            t_stat, p_value = np.nan, np.nan
        
        # Store the results
        metric_results[f'{cond1} vs {cond2}'] = {
            'mean_difference': np.mean(differences) if differences else np.nan,
            't_stat': t_stat,
            'p_value': p_value,
            'differences': differences  # Store the differences if you need them later
        }
    
    # Add results to the main dictionary
    results[metric] = metric_results

# Convert results to a DataFrame for easier viewing
results_df = pd.DataFrame.from_dict({(i,j): results[i][j] 
                                   for i in results.keys() 
                                   for j in results[i].keys()},
                                   orient='index')

# # Display the results
# for k,v in results.items():
#     print (k)
#     for k2, v2 in v.items():
#         print (k2)
#         print ("Mean Difference:", v2['mean_difference'])
#         print ("t-statistic:", v2['t_stat'])
#         print ("p-value:", v2['p_value'])
#         print ("Number of differences:", len(v2['differences']))
#         print()


# Collect all p-values from the tests
p_values = []

# Iterate over each metric and comparison to collect p-values
for metric in metrics:
    for comparison in comparisons:
        p_values.append(results[metric][f'{comparison[0]} vs {comparison[1]}']['p_value'])

# Apply multiple testing correction (e.g., FDR)
adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]

# Assign the corrected p-values back to the results
index = 0
for metric in metrics:
    for comparison in comparisons:
        results[metric][f'{comparison[0]} vs {comparison[1]}']['adjusted_p_value'] = adjusted_p_values[index]
        index += 1

# Convert results to a DataFrame for easier viewing
results_df = pd.DataFrame.from_dict({(i,j): results[i][j] 
                                   for i in results.keys() 
                                   for j in results[i].keys()},
                                   orient='index')

# Display the results with adjusted p-values
for k,v in results.items():
    print (k)
    for k2, v2 in v.items():
        print (k2)
        print ("Mean Difference:", v2['mean_difference'])
        print ("t-statistic:", v2['t_stat'])
        print ("p-value:", v2['p_value'])
        print ("Adjusted p-value:", v2['adjusted_p_value'])
        print ("Number of differences:", len(v2['differences']))
        print()