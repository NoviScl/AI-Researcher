import nltk
from nltk.corpus import stopwords
import string
import json 

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
    with open("../cache_results_claude_may/ideas_1k/uncertainty_prompting_method_prompting.json", "r") as f:
        ideas_json = json.load(f)
    ideas_lst = [process_text(idea) for lst in ideas_json["ideas"] for idea in lst]
    print (len(ideas_lst))

