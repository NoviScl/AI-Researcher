import nltk
from nltk.corpus import stopwords
import string
import json 
from tqdm import tqdm 
from collections import Counter
import numpy as np
import pandas as pd
import argparse
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
    output += "Problem: " + idea_v["Problem"] + "\n"
    output += "Existing Methods: " + idea_v["Existing Methods"] + "\n"
    output += "Motivation: " + idea_v["Motivation"] + "\n"
    output += "Proposed Method: " + idea_v["Proposed Method"] + "\n"
    output += "Experiment Plan: " + idea_v["Experiment Plan"] + "\n"

    return output


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_name', type=str, default="bias", help='cache file name')
    args = parser.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_ideas = []
    with open("../cache_results_claude_may/ideas_1k_claude3-5/{}_prompting_RAG.json".format(args.cache_name), "r") as f:
        ideas_json = json.load(f)
        for ideas_dict in ideas_json["ideas"]:
            for idea_k, idea_v in ideas_dict.items():
                all_ideas.append(concatenate_idea(idea_k, idea_v))
    
    # all_ideas = all_ideas[:40]
    print ("#ideas: ", len(all_ideas))

    embeddings = model.encode(all_ideas)
    similarity_matrix = model.similarity(embeddings, embeddings)
    similarity_matrix = similarity_matrix.numpy()
    ## setting the diagonal to 0
    np.fill_diagonal(similarity_matrix, 0)

    # print (similarity_matrix)

    nn_similarity = []
    nn_similarity_idx = []
    avg_similarity = []
    for i in range(len(all_ideas)):
        nn_similarity.append(np.max(similarity_matrix[i]))
        nn_similarity_idx.append(np.argmax(similarity_matrix[i]))
        avg_similarity.append(np.sum(similarity_matrix[i]) / (len(all_ideas) - 1))

    highest_nn_similarity = np.argmax(nn_similarity)
    print ("Idea with Highest NN Similarity:\n", all_ideas[highest_nn_similarity])
    print ("\nMost Similar Idea:\n", all_ideas[nn_similarity_idx[highest_nn_similarity]])
    print ("\nSimilarity: ", nn_similarity[highest_nn_similarity])

    lowest_nn_similarity = np.argmin(nn_similarity)
    print ("\nIdea with Lowest NN Similarity:\n", all_ideas[lowest_nn_similarity])
    print ("\nMost Similar Idea:\n", all_ideas[nn_similarity_idx[lowest_nn_similarity]])
    print ("\nSimilarity: ", nn_similarity[lowest_nn_similarity])

    highest_avg_similarity = np.argmax(avg_similarity)
    print ("\nIdea with Highest Avg Similarity:\n", all_ideas[highest_avg_similarity])
    print ("\nMost Similar Idea:\n", all_ideas[nn_similarity_idx[highest_avg_similarity]])
    print ("\nSimilarity: ", avg_similarity[highest_avg_similarity])

    lowest_avg_similarity = np.argmin(avg_similarity)
    print ("\nIdea with Lowest Avg Similarity:\n", all_ideas[lowest_avg_similarity])
    print ("\nMost Similar Idea:\n", all_ideas[nn_similarity_idx[lowest_avg_similarity]])
    print ("\nSimilarity: ", avg_similarity[lowest_avg_similarity])

    print ("\n\nCorpus level metrics:")
    print ("Avg NN Similarity: ", np.mean(nn_similarity))
    print ("Avg Avg Similarity: ", np.mean(avg_similarity))