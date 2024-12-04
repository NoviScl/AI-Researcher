import nltk
from nltk.corpus import stopwords
import string
import json 
from tqdm import tqdm 
from collections import Counter
import numpy as np
import pandas as pd
import argparse
import os
from sentence_transformers import SentenceTransformer

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

def concatenate_idea(idea_k, idea_v):
    output = ""
    output += idea_k + "\n"
    
    if isinstance(idea_v, dict):
        output += "Problem: " + idea_v["Problem"] + "\n"
        output += "Existing Methods: " + idea_v["Existing Methods"] + "\n"
        output += "Motivation: " + idea_v["Motivation"] + "\n"
        output += "Proposed Method: " + idea_v["Proposed Method"] + "\n"
        output += "Experiment Plan: " + idea_v["Experiment Plan"] + "\n"
    else:
        output += str(idea_v) + "\n"

    return output


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default="bias", help='cache file name')
    parser.add_argument('--cache_name', type=str, default="bias", help='cache file name')
    parser.add_argument('--similarity_threshold', type=float, default=0.8, help='NN Similarity Threshold')
    parser.add_argument('--dedup_cache_dir', type=str, default="bias", help='cache file name')
    parser.add_argument("--num_ideas", type=int, default=1000, help="top n ideas to consider")
    args = parser.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print ("Similarity Threshold: ", args.similarity_threshold)

    all_ideas = []
    all_idea_ks = []
    all_idea_vs = []
    topic = ""
    with open(os.path.join(args.cache_dir, args.cache_name + ".json"), "r") as f:
        ideas_json = json.load(f)
        topic = ideas_json["topic_description"]
        for ideas_dict in ideas_json["ideas"]:
            for idea_k, idea_v in ideas_dict.items():
                try:
                    all_ideas.append(concatenate_idea(idea_k, idea_v))
                    all_idea_ks.append(idea_k)
                    all_idea_vs.append(idea_v)
                except:
                    continue
    
    # all_ideas = all_ideas[:40]
    print ("#original ideas: ", len(all_ideas))
    all_ideas = all_ideas[:args.num_ideas]

    similarity_matrix = np.load(os.path.join(args.cache_dir, args.cache_name + "_similarity_matrix.npy"))
    if len(similarity_matrix) != len(all_ideas):
        print ("Error: similarity matrix size mismatch")
        exit(0)

    final_ideas = {}
    filter_idx = [] ## ideas that should be filtered
    for i in tqdm(range(len(all_ideas))):
        if i not in filter_idx:
            ## add current idea to filtered_ideas
            final_ideas[all_idea_ks[i]] = all_idea_vs[i]

            ## filter out similar ideas
            for j in range(i+1, len(all_ideas)):
                if j not in filter_idx and similarity_matrix[i][j] > args.similarity_threshold or all_idea_ks[j] == all_idea_ks[i]:
                    filter_idx.append(j)
    
    print ("#final ideas: ", len(final_ideas))

    final_json = {}
    final_json["topic_description"] = topic 
    final_json["ideas"] = final_ideas 
    if not os.path.exists(args.dedup_cache_dir):
        os.makedirs(args.dedup_cache_dir)
    with open(os.path.join(args.dedup_cache_dir, args.cache_name + ".json"), "w") as f:
        json.dump(final_json, f, indent=4)
