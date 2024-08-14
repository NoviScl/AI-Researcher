cache_dir="../cache_results_claude_may/ideas_5k/"
cache_names=("coding_prompting_method_merged" "factuality_prompting_method_merged" "math_prompting_method_merged"  "multilingual_prompting_method_merged" "safety_prompting_method_merged"  "uncertainty_prompting_method_merged")

for cache_name in "${cache_names[@]}"; do
    echo "Running dedup_ideas.py with cache_name: $cache_name"
    python3 src/dedup_ideas.py \
    --cache_dir "$cache_dir" \
    --cache_name "$cache_name" \
    --dedup_cache_dir "../cache_results_claude_may/ideas_5k_dedup" \
    --similarity_threshold 0.85 > logs/dedup_"$cache_name".log 2>&1
done


# cache_dir="../cache_results_claude_july/ideas_emnlp/"
# cache_names=("bias_prompting_method_RAG" "bias_prompting_method" "coding_prompting_method_RAG" "coding_prompting_method" "factuality_prompting_method_RAG" "factuality_prompting_method" "math_prompting_method_RAG" "math_prompting_method" "multilingual_prompting_method_RAG" "multilingual_prompting_method" "safety_prompting_method_RAG" "safety_prompting_method" "uncertainty_prompting_method_RAG" "uncertainty_prompting_method")

# for cache_name in "${cache_names[@]}"; do
#     echo "Running dedup_ideas.py with cache_name: $cache_name"
#     python3 src/dedup_ideas.py \
#     --cache_dir "$cache_dir" \
#     --cache_name "$cache_name" \
#     --dedup_cache_dir "../cache_results_claude_may/ideas_5k_dedup" \
#     --similarity_threshold 0.8 > logs/dedup_"$cache_name".log 2>&1
# done



# cache_dir="../cache_results_claude_july/ideas_emnlp/"
# cache_names=("CSS" "Dialogue" "Discourse_Pragmatics" "Low_resource" "Ethics_Bias_Fairness" "Information_Extraction" "Information_Retrieval_Text_Mining" "Interpretability_Analysis" "Machine_Translation" "Multilinguality_Language_Diversity" "Multimodality_Language_Grounding" "Question_Answering" "Semantics" "Sentiment_Stylistic_Argument" "Speech_processing" "Summarization" "Efficiency_Model_Algorithms")

# for cache_name in "${cache_names[@]}"; do
#     echo "Running dedup_ideas.py with cache_name: $cache_name"
#     python3 src/dedup_ideas.py \
#     --cache_dir "$cache_dir" \
#     --cache_name "$cache_name" \
#     --dedup_cache_dir "../cache_results_claude_july/ideas_emnlp_dedup" \
#     --similarity_threshold 0.85 > logs/dedup_emnlp/dedup_"$cache_name".log 2>&1
# done



