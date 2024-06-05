import nltk
from nltk.corpus import stopwords
import string
import json 
from tqdm import tqdm 
import matplotlib.pyplot as plt
from collections import Counter

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

def process_text(input_text):
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
    
    return processed_text

if __name__ == "__main__":
    track = "uncertainty"
    with open("../cache_results_claude_may/ideas_1k/{}_prompting_method_prompting.json".format(track), "r") as f:
        ideas_json = json.load(f)

    topic = ideas_json["topic_description"]
    ideas_lst = ideas_json["ideas"]
    print ("Original #ideas: ", len(ideas_lst) * 5)

    dedup_dict = {}
    dedup_dict["topic_description"] = topic
    dedup_dict["ideas"] = {}

    for ideas_dict in tqdm(ideas_lst):
        for idea_k, idea_v in ideas_dict.items():
            title = process_text(idea_k)
            if title not in dedup_dict["ideas"]:
                dedup_dict["ideas"][idea_k] = idea_v
    
    print ("Dedup'ed #ideas: ", len(dedup_dict["ideas"]))

    with open("../cache_results_claude_may/ideas_1k_dedup/{}_prompting_method_prompting.json".format(track), "w") as f:
        json.dump(dedup_dict, f, indent=4)



