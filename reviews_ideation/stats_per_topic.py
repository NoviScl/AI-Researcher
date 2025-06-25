import pandas as pd
import json
from scipy.stats import ttest_ind

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
metrics = ['overall_score', 'novelty_score', 'excitement_score', 'feasibility_score', 'effectiveness_score']

# Function to calculate mean, std, and p-value by topic and condition
def calculate_stats(df, topic, metrics):
    results = {}
    for metric in metrics:
        mean_val = df[df['topic'] == topic][metric].mean()
        std_val = df[df['topic'] == topic][metric].std()
        results[metric] = {'mean': mean_val, 'std': std_val}
    return results

# Function to calculate p-value using Welch's t-test
def calculate_p_value(group1, group2, metric):
    t_stat, p_value = ttest_ind(group1[metric], group2[metric], equal_var=False)
    return p_value

# Topics available in the DataFrame
topics = df['topic'].unique()

# Dictionary to store all results
all_results = {}

# Number of comparisons for Bonferroni correction
num_comparisons = len(metrics) * 2

# Loop through each topic and calculate the mean, std, and p-values for each condition
for topic in topics:
    all_results[topic] = {
        'Human': calculate_stats(df_human, topic, metrics),
        'AI': calculate_stats(df_ai, topic, metrics),
        'AI_Rerank': calculate_stats(df_human_ai, topic, metrics),
    }

    for metric in metrics:
        # Calculate p-values
        p_value_ai = calculate_p_value(df_human[df_human['topic'] == topic], df_ai[df_ai['topic'] == topic], metric)
        p_value_ai_rerank = calculate_p_value(df_human[df_human['topic'] == topic], df_human_ai[df_human_ai['topic'] == topic], metric)
        
        # Apply Bonferroni correction
        p_value_ai_corrected = min(p_value_ai * num_comparisons, 1.0)
        p_value_ai_rerank_corrected = min(p_value_ai_rerank * num_comparisons, 1.0)
        
        all_results[topic]['AI'][metric]['p_value'] = p_value_ai_corrected
        all_results[topic]['AI_Rerank'][metric]['p_value'] = p_value_ai_rerank_corrected

# Print results in the requested format
for topic, conditions in all_results.items():
    print(f"Topic: {topic}")
    for metric in metrics:
        print(f"  {metric}:")
        for condition in ['Human', 'AI', 'AI_Rerank']:
            stats = conditions[condition][metric]
            p_value_str = f", p-value = {stats.get('p_value', 'N/A')}" if condition != 'Human' else ""
            print(f"    {condition}: Mean = {stats['mean']}, Std = {stats['std']}{p_value_str}")
