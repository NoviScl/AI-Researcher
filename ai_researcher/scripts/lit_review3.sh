#!/bin/bash

# Define the array of topics
topics=("Machine Translation" "Multilinguality and Language Diversity" "Multimodality and Language Grounding to Vision, Robotics and Beyond" "Phonology, Morphology and Word Segmentation" "Question Answering" "Semantics: Lexical, Sentence-level Semantics, Textual Inference and Other areas" "Sentiment Analysis, Stylistic Analysis, and Argument Mining" "Speech processing and spoken language understanding" "Summarization" "Syntax: Tagging, Chunking and Parsing" "NLP Applications" "Efficiency in Model Algorithms, Training, and Inference" "Machine learning for sciences (e.g. climate, health, life sciences, physics, social sciences)")

# Define the corresponding shortnames
shortnames=("Machine_Translation" "Multilinguality_Language_Diversity" "Multimodality_Language_Grounding" "Phonology_Morphology_Word_Segmentation" "Question_Answering" "Semantics" "Sentiment_Stylistic_Argument" "Speech_processing" "Summarization" "Syntax" "NLP_Applications" "Efficiency_Model_Algorithms" "Machine_learning_sciences")

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
