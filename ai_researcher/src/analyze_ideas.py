import nltk
from nltk.corpus import stopwords
import string
import json 
from tqdm import tqdm 
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

def plot_string_occurrences(strings_list):
    # Count occurrences of each string
    occurrences = Counter(strings_list)
    
    # Count how many strings have each occurrence count
    count_of_occurrences = Counter(occurrences.values())
    
    # Extracting the data for plotting
    x = sorted(count_of_occurrences.keys())
    y = [count_of_occurrences[occ] for occ in x]
    
    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color='skyblue')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Number of Strings')
    plt.title('Frequency of String Occurrences')
    plt.xticks(x)
    plt.grid(axis='y')
    plt.show()

def process_text(input_text, tokenize=False):
    # Define the list of stopwords
    stop_words = set(stopwords.words('english'))
    
    # Lowercase the input text
    lowercased_text = input_text.lower()
    
    # Remove punctuation from the text
    no_punctuation_text = lowercased_text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize the text into words
    words = no_punctuation_text.split()
    
    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join the filtered words back into a single string
    processed_text = ' '.join(filtered_words)

    if tokenize:
        return set(filtered_words)
    else:
        return process_text

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def find_representative_paper(cluster, similarity_matrix, labels):
    cluster_indices = [i for i, label in enumerate(labels) if label == cluster]
    cluster_sims = similarity_matrix[cluster_indices][:, cluster_indices]
    avg_sims = cluster_sims.mean(axis=1)
    representative_index = cluster_indices[avg_sims.argmax()]
    return representative_index

def find_top_n_papers(representative_index, similarity_matrix, n=5):
    sims = similarity_matrix[representative_index]
    closest_indices = np.argsort(-sims)[:n]  # Sort in descending order and get top-n
    return closest_indices


if __name__ == "__main__":
    track = "uncertainty"
    with open("../cache_results_claude_may/ideas_1k_dedup/{}_prompting_method_prompting.json".format(track), "r") as f:
        ideas_json = json.load(f)

    # topic = ideas_json["topic_description"]
    # ideas_lst = ideas_json["ideas"]
    # print ("Original #ideas: ", len(ideas_lst) * 5)

    # dedup_dict = {}
    # dedup_dict["topic_description"] = topic
    # dedup_dict["ideas"] = {}

    # for ideas_dict in tqdm(ideas_lst):
    #     for idea_k, idea_v in ideas_dict.items():
    #         title = process_text(idea_k)
    #         if title not in dedup_dict["ideas"]:
    #             dedup_dict["ideas"][idea_k] = idea_v
    
    # print ("Dedup'ed #ideas: ", len(dedup_dict["ideas"]))

    # with open("../cache_results_claude_may/ideas_1k_dedup/{}_prompting_method_prompting.json".format(track), "w") as f:
    #     json.dump(dedup_dict, f, indent=4)

    idea_names = list(ideas_json["ideas"].keys())
    abstracts = []
    for idea_name in idea_names:
        content = ideas_json["ideas"][idea_name]
        abstracts.append(" ".join([content["Problem"], content["Existing Methods"], content["Motivation"], content["Proposed Method"], content["Experiment Plan"]]))  
    tokenized_summaries = [process_text(abstract, tokenize=True) for abstract in abstracts]

    num_summaries = len(tokenized_summaries)
    similarity_matrix = np.zeros((num_summaries, num_summaries))

    for i in tqdm(range(num_summaries)):
        for j in range(num_summaries):
            similarity_matrix[i][j] = jaccard_similarity(tokenized_summaries[i], tokenized_summaries[j])

    # Apply Agglomerative Clustering
    num_clusters = 6
    clustering = AgglomerativeClustering(n_clusters=num_clusters)
    clustering.fit(1 - similarity_matrix)  # 1 - similarity_matrix to convert similarity to distance

    representative_indices = [find_representative_paper(cluster, similarity_matrix, clustering.labels_) for cluster in range(num_clusters)]
    top_n_papers_per_cluster = [find_top_n_papers(rep_idx, similarity_matrix) for rep_idx in representative_indices]
    # Print the top papers in each cluster
    for cluster, top_indices in enumerate(top_n_papers_per_cluster):
        print(f"Cluster {cluster + 1}:")
        for idx in top_indices:
            print(idea_names[idx])
        # print()


    # # bucketing
    # clusters = {i: [] for i in range(num_clusters)}
    # for i, paper in enumerate(idea_names):
    #     cluster_label = clustering.labels_[i]
    #     clusters[cluster_label].append(paper)
    
    # for cluster_label, papers in clusters.items():
    #     print ("Cluster #{}: ".format(cluster_label))
    #     print ("Top-5: ", papers[:5])

    # # Add cluster labels to the original papers
    # for i, paper in enumerate(idea_names):
    #     paper['cluster'] = clustering.labels_[i]

    # # Convert to DataFrame for easier manipulation
    # df_papers = pd.DataFrame(papers)
    # print(df_papers)



