import json 
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns



# Load data from a JSON file
with open('data_points_all_execution.json', 'r') as f:
    data = json.load(f)

with open("../reviews_ideation/data_points_all_anonymized.json", "r") as f:
    data_study1 = json.load(f)

# Convert JSON data to a pandas DataFrame
df = pd.DataFrame(data)

# Convert study1 data to DataFrame
df_study1 = pd.DataFrame(data_study1)

# Get the set of idea_ids that appear in both datasets
common_idea_ids = set(df['idea_id']).intersection(set(df_study1['idea_id']))
# Print the number of unique idea_ids that appear in both datasets
print(f"Number of unique idea_ids in both datasets: {len(common_idea_ids)}")

# Filter both dataframes to only include data points with idea_ids that appear in both datasets
df = df[df['idea_id'].isin(common_idea_ids)]
df_study1 = df_study1[df_study1['idea_id'].isin(common_idea_ids)]

show_study1 = False
if show_study1:
    # Use the data from study1 for the common idea ids
    df = df_study1
    print ("Using results from study 1.")
    metrics = ['novelty_score', 'excitement_score', 'effectiveness_score', 'overall_score']
else:
    print ("Using results from study 2.")
    metrics = ['novelty_score', 'excitement_score', 'effectiveness_score', 'overall_score']


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
    # t_stat_human_ai, p_val_human_ai = stats.ttest_ind(df_human_ai[metric], df_human[metric], equal_var=False)
    results[metric] = {
        'AI vs Human': (t_stat_ai, p_val_ai),
        # 'Human+AI vs Human': (t_stat_human_ai, p_val_human_ai)
    }

# Apply Bonferroni correction or another method for multiple comparisons
p_vals_ai = [results[metric]['AI vs Human'][1] for metric in metrics]
# p_vals_human_ai = [results[metric]['Human+AI vs Human'][1] for metric in metrics]

# Adjust p-values
adjusted_p_vals_ai = multipletests(p_vals_ai, method='fdr_bh')[1]
# adjusted_p_vals_human_ai = multipletests(p_vals_human_ai, method='bonferroni')[1]
# Store both unadjusted and adjusted p-values
for i, metric in enumerate(metrics):
    results[metric]['AI vs Human'] = (results[metric]['AI vs Human'][0], p_vals_ai[i], adjusted_p_vals_ai[i])
    # results[metric]['Human+AI vs Human'] = (results[metric]['Human+AI vs Human'][0], p_vals_human_ai[i], adjusted_p_vals_human_ai[i])

# When printing results:
for metric in metrics:
    mean_human = np.mean(df_human[metric])
    mean_ai = np.mean(df_ai[metric])
    mean_diff = mean_human - mean_ai
    
    print(metric)
    print("Human: N=", len(df_human[metric]), " Mean=", mean_human, " Median=", np.median(df_human[metric]),  " SD=", np.std(df_human[metric]), " SE=", np.std(df_human[metric] / np.sqrt(len(df_human[metric]))), " Min=", np.min(df_human[metric]), " Max=", np.max(df_human[metric]))
    print("AI: N=", len(df_ai[metric]), " Mean=", mean_ai, " Median=", np.median(df_ai[metric]), " SD=", np.std(df_ai[metric]), " SE=", np.std(df_ai[metric] / np.sqrt(len(df_ai[metric]))), " Min=", np.min(df_ai[metric]), " Max=", np.max(df_ai[metric]))
    print("Mean Difference (Human - AI):", mean_diff)
    # print("AI + Human: N=", len(df_human_ai[metric]), " Mean=", np.mean(df_human_ai[metric]), " Median=", np.median(df_human_ai[metric]), " SD=", np.std(df_human_ai[metric]), " SE=", np.std(df_human_ai[metric] / np.sqrt(len(df_human_ai[metric]))), " Min=", np.min(df_human_ai[metric]), " Max=", np.max(df_human_ai[metric]))
    for condition, values in results[metric].items():
        print(condition, "t-statistic:", values[0], "p-value (unadjusted):", values[1], "p-value (adjusted):", values[2])
    print()


# # Filter the grouped DataFrame to only include the 'Human' condition
# df_human_sorted = df_grouped[df_grouped['condition'] == 'Human']

# # Sort the DataFrame by 'overall_score' in descending order and select the top 5
# top_5_human_ideas = df_human_sorted.sort_values(by='overall_score', ascending=False).head(8)

# # Print the idea_id of the top 5 ideas
# print("Top 5 Human Ideas based on average overall_score:")
# for i, row in top_5_human_ideas.iterrows():
#     print(f"Idea ID: {row['idea_id']}, Overall Score: {row['overall_score']:.2f}")

# print ("\n\n")

# # Function to compute top-K ideas based on 'overall_score' (excluding AI_Rerank) and print their name, condition, and overall score
# def print_top_k_ideas(df_grouped, K):
#     # # Exclude AI_Rerank condition from the DataFrame
#     # df_filtered = df_grouped[df_grouped['condition'] != 'AI_Rerank']
#     df_filtered = df_grouped
    
#     # Sort the filtered DataFrame by 'overall_score' in descending order
#     df_sorted = df_filtered.sort_values(by='novelty_score', ascending=False)
    
#     # Select the top-K ideas
#     top_k_ideas = df_sorted.head(K)

#     # Calculate the percentages of Human and AI ideas in the top-K
#     human_count = (top_k_ideas['condition'] == 'Human').sum()
#     ai_count = (top_k_ideas['condition'] == 'AI').sum()
    
#     # Print the top-K ideas with their 'idea_id', 'condition', and 'overall_score'
#     print(f"Top {K} Ideas based on average overall_score (excluding AI_Rerank):")
#     for i, row in top_k_ideas.iterrows():
#         print(f"Idea ID: {row['idea_id']}, Condition: {row['condition']}, Overall Score: {row['overall_score']:.2f}")
#     print()

#     # Calculate and print percentages
#     print(f"Percentage of Human ideas in Top-{K}: {100 * human_count / K:.2f}%")
#     print(f"Percentage of AI ideas in Top-{K}: {100 * ai_count / K:.2f}%")
#     print()

# # Now let's vary K and print the top-K ideas excluding AI_Rerank
# for K in [10, 50]:
#     print_top_k_ideas(df_grouped, K)




# # Create a boxplot with all data points shown
# plt.figure(figsize=(10, 6))

# # Create the boxplot for each condition with data points overlaid
# sns.boxplot(data=df_grouped, x='condition', y='feasibility_score', whis=[5, 95], showfliers=False)  # Boxplot for each condition

# # Overlay the individual data points as a scatter plot
# sns.stripplot(data=df_grouped, x='condition', y='feasibility_score', color='black', alpha=0.6, jitter=True)  # Add data points

# # Customize the plot
# plt.title('Distribution of Average Feasibility Scores by Condition')
# plt.xlabel('Condition')
# plt.ylabel('Average Feasibility Score')
# plt.ylim(0, 10)  # Assuming scores are between 0 and 10
# plt.grid(True)

# # Display the plot
# plt.show()



