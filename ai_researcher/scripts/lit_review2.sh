#!/bin/bash

# # Define the array of topics
# topics=("Computational Social Science and Cultural Analytics" "Dialogue and Interactive Systems" "Discourse and Pragmatics" "Low-resource Methods for NLP" "Ethics, Bias, and Fairness" "Generation" "Information Extraction" "Information Retrieval and Text Mining" "Interpretability and Analysis of Models for NLP" "Linguistic theories, Cognitive Modeling and Psycholinguistics" "Machine Learning for NLP")

# # Define the corresponding shortnames
# shortnames=("CSS" "Dialogue" "Discourse_Pragmatics" "Low_resource" "Ethics_Bias_Fairness" "Generation" "Information_Extraction" "Information_Retrieval_Text_Mining" "Interpretability_Analysis" "Linguistic_theories_Cognitive_Modeling_Psycholinguistics" "Machine_Learning")

# Define the array of topics
topics=("Interpretability and Analysis of Models for NLP")

# Define the corresponding shortnames
shortnames=("Interpretability_Analysis")


# Path to the lit_review.py script
lit_review_script="src/lit_review.py"

# Engine to use
engine="claude-3-5-sonnet-20240620"

# Cache directory
cache_dir="../cache_results_claude_july/lit_review_emnlp"

# Maximum paper bank size
max_paper_bank_size=120

# Log directory
log_dir="logs/lit_review_emnlp/"

# Create log directory if it doesn't exist
mkdir -p $log_dir

# Loop through each topic and run the corresponding job
for i in "${!topics[@]}"; do
    topic="${topics[i]}"
    shortname="${shortnames[i]}"
    
    # Create the cache file path
    cache_path="${cache_dir}/${shortname}.json"
    
    # Create the log file path
    log_path="${log_dir}/${shortname}.log"

    # Echo the current topic being processed
    echo "Processing topic: $topic"
    
    # Run the lit_review.py script for the current topic
    python3 $lit_review_script \
        --engine "$engine" \
        --mode "topic" \
        --topic_description "novel prompting methods for $topic" \
        --cache_name "$cache_path" \
        --max_paper_bank_size $max_paper_bank_size \
        --print_all > "$log_path" 2>&1
done

echo "All jobs have been executed."
