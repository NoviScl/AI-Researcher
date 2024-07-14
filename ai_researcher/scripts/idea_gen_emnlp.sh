topic_names=("Interpretability and Analysis of Models for NLP" "Computational Social Science and Cultural Analytics" "Information Retrieval and Text Mining" "Linguistic theories, Cognitive Modeling and Psycholinguistics" "Multimodality and Language Grounding to Vision, Robotics and Beyond" "Efficiency in Model Algorithms, Training, and Inference" "Summarization" "Question Answering")
ideas_n=5

mkdir -p logs/idea_gen_emnlp

# Iterate over each seed from 1 to 1000
for seed in {1..3}; do
    # Iterate over each cache name 
    for topic in "${topic_names[@]}"; do
        # Iterate over each method 
        echo "Running idea_gen_emnlp.py on: $topic with seed $seed"
        python3 src/idea_gen_emnlp.py \
            --engine "claude-3-5-sonnet-20240620" \
            --idea_cache "../cache_results_claude_may/idea_gen_emnlp/$topic.json" \
            --topic_description "$topic" \
            --ideas_n $ideas_n \
            --seed $seed >> "logs/idea_gen_emnlp/$topic.log" 2>&1
    done
done


## tmux 8