import json 
import os 
from utils import avg_score, min_score, max_score
import numpy as np
import matplotlib.pyplot as plt
import random 
random.seed(2024)

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
    filenames = [f for f in filenames if f.endswith(".json")]
    # filenames = [f for f in filenames]
    all_scores = []
    all_papers = []
    pos_papers = []
    neg_papers = []

    for filename in filenames:
        with open("../{}/{}".format(cache_name, filename), "r") as f:
            paper = json.load(f)
        
        scores = paper["scores"]
        mean_score = avg_score(scores)
        all_scores.append(mean_score)
        all_papers.append(paper)

        if mean_score > 6 and min_score(scores) >= 6:
            pos_papers.append(paper)
        if mean_score < 5 and max_score(scores) <= 5:
            neg_papers.append(paper)

    print (np.mean(all_scores), np.std(all_scores))
    print (max(all_scores), min(all_scores))
    # print (pos, neg)

    # plot_score_buckets(all_scores)
    
    '''
    pos_papers = [paper for paper in pos_papers if "structured_summary" in paper and isinstance(paper["structured_summary"], dict) and "scores" in paper]
    neg_papers = [paper for paper in neg_papers if "structured_summary" in paper and isinstance(paper["structured_summary"], dict) and "scores" in paper]
    random.shuffle(pos_papers)
    random.shuffle(neg_papers)

    print (len(pos_papers), len(neg_papers))

    with open("../ORB_full/pos_papers.json", "w") as f:
        json.dump(pos_papers, f, indent=4)
    with open("../ORB_full/neg_papers.json", "w") as f:
        json.dump(neg_papers, f, indent=4)
    '''

    sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i])
    lowest_indices = sorted_indices[:50]
    highest_indices = sorted_indices[-50:]

    neg_papers = [all_papers[i] for i in lowest_indices]
    pos_papers = [all_papers[i] for i in highest_indices]

    neg_scores = [all_scores[i] for i in lowest_indices]
    pos_scores = [all_scores[i] for i in highest_indices]

    print (np.mean(neg_scores), np.std(neg_scores))
    print (np.mean(pos_scores), np.std(pos_scores))

    random.shuffle(pos_papers)
    random.shuffle(neg_papers)

    with open("../ORB_easy/pos_papers.json", "w") as f:
        json.dump(pos_papers, f, indent=4)
    with open("../ORB_easy/neg_papers.json", "w") as f:
        json.dump(neg_papers, f, indent=4)