import json 
import os 
from utils import avg_score
import numpy as np
import matplotlib.pyplot as plt

def plot_score_buckets(scores):
    # Create bins for score ranges
    bins = np.arange(1, 11, 1)  # 1-2, 2-3, ..., 9-10
    hist, bin_edges = np.histogram(scores, bins=bins)

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist, width=0.9, edgecolor='black', align='edge')
    plt.xlabel('Score Range')
    plt.ylabel('Number of Papers')
    plt.title('Distribution of Average Paper Scores')
    plt.xticks(bin_edges)
    plt.grid(axis='y')
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    cache_name = "openreview_benchmark"
    filenames = os.listdir("../{}".format(cache_name))
    filenames = [f for f in filenames if f.endswith(".json") and '5' in f]
    all_scores = []

    for filename in filenames:
        with open("../{}/{}".format(cache_name, filename), "r") as f:
            paper = json.load(f)
        
        scores = paper["scores"]
        score = avg_score(scores)
        all_scores.append(score)
    
    print (np.mean(all_scores), np.std(all_scores))
    print (max(all_scores), min(all_scores))

    plot_score_buckets(all_scores)

