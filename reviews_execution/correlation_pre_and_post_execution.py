import json
import numpy as np
import pandas as pd
from scipy import stats

# Load data from both studies
with open('data_points_all_execution.json', 'r') as f:
    data_study2 = json.load(f)

with open("../reviews_ideation/data_points_all_anonymized.json", "r") as f:
    data_study1 = json.load(f)

# Convert JSON data to pandas DataFrames
df_study2 = pd.DataFrame(data_study2)
df_study1 = pd.DataFrame(data_study1)

# Get the set of idea_ids that appear in both datasets
common_idea_ids = set(df_study2['idea_id']).intersection(set(df_study1['idea_id']))
print(f"Number of unique idea_ids in both datasets: {len(common_idea_ids)}")

# Filter both dataframes to only include data points with idea_ids that appear in both datasets
df_study2 = df_study2[df_study2['idea_id'].isin(common_idea_ids)]
df_study1 = df_study1[df_study1['idea_id'].isin(common_idea_ids)]

# Define metrics to analyze
metrics = ['novelty_score', 'excitement_score', 'effectiveness_score', 'overall_score']

# Compute mean scores per idea for each study
means_study1 = df_study1.groupby('idea_id')[metrics].mean().sort_index()
means_study2 = df_study2.groupby('idea_id')[metrics].mean().sort_index()

# Ensure both have the same order of idea_ids
means_study1 = means_study1.loc[sorted(common_idea_ids)]
means_study2 = means_study2.loc[sorted(common_idea_ids)]

# Get condition information to separate AI and human ideas
idea_conditions = df_study1.groupby('idea_id')['condition'].first()

# Separate AI and human ideas based on condition
ai_idea_ids = idea_conditions[idea_conditions == 'AI'].index
human_idea_ids = idea_conditions[idea_conditions == 'Human'].index

print(f"\nNumber of AI ideas: {len(ai_idea_ids)}")
print(f"Number of human ideas: {len(human_idea_ids)}")

print("\n=== OVERALL CORRELATION ===")
print("Correlation between Study 1 and Study 2 average scores per idea:")
for metric in metrics:
    x = means_study1[metric].values
    y = means_study2[metric].values

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    print(f"\nMetric: {metric}")
    print(f"  Pearson correlation: r = {pearson_r:.3f}, p = {pearson_p:.4g}")
    print(f"  Spearman correlation: r = {spearman_r:.3f}, p = {spearman_p:.4g}")

print("\n=== AI IDEAS CORRELATION ===")
print("Correlation between Study 1 and Study 2 average scores per AI idea:")
for metric in metrics:
    x = means_study1.loc[ai_idea_ids, metric].values
    y = means_study2.loc[ai_idea_ids, metric].values

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    print(f"\nMetric: {metric}")
    print(f"  Pearson correlation: r = {pearson_r:.3f}, p = {pearson_p:.4g}")
    print(f"  Spearman correlation: r = {spearman_r:.3f}, p = {spearman_p:.4g}")

print("\n=== HUMAN IDEAS CORRELATION ===")
print("Correlation between Study 1 and Study 2 average scores per human idea:")
for metric in metrics:
    x = means_study1.loc[human_idea_ids, metric].values
    y = means_study2.loc[human_idea_ids, metric].values

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    print(f"\nMetric: {metric}")
    print(f"  Pearson correlation: r = {pearson_r:.3f}, p = {pearson_p:.4g}")
    print(f"  Spearman correlation: r = {spearman_r:.3f}, p = {spearman_p:.4g}")
