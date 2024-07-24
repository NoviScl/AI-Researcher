# cache_dir="../cache_results_claude_may/ideas_5k/"
# cache_names=("bias_prompting_method_RAG" "bias_prompting_method" "coding_prompting_method_RAG" "coding_prompting_method" "factuality_prompting_method_RAG" "factuality_prompting_method" "math_prompting_method_RAG" "math_prompting_method" "multilingual_prompting_method_RAG" "multilingual_prompting_method" "safety_prompting_method_RAG" "safety_prompting_method" "uncertainty_prompting_method_RAG" "uncertainty_prompting_method")

# for cache_name in "${cache_names[@]}"; do
#     echo "Running analyze_ideas_semantic_similarity.py with cache_name: $cache_name"
#     python3 src/analyze_ideas_semantic_similarity.py \
#     --cache_dir "$cache_dir" \
#     --cache_name "$cache_name" \
#     --save_similarity_matrix > logs/semantic_similarity_"$cache_name".log 2>&1
# done


# cache_dir="../cache_results_claude_may/experiment_plans_5k_dedup/"
# cache_names=("bias_prompting_method_merged" "coding_prompting_method_merged" "factuality_prompting_method_merged" "math_prompting_method_merged" "multilingual_prompting_method_merged" "safety_prompting_method_merged" "uncertainty_prompting_method_merged")

# mkdir -p logs/semantic_similarity_experiment_plans

# for cache_name in "${cache_names[@]}"; do
#     echo "Running analyze_experiment_plans_semantic_similarity.py with cache_name: $cache_name"
#     python3 src/analyze_experiment_plans_semantic_similarity.py \
#     --cache_dir "$cache_dir" \
#     --cache_name "$cache_name" \
#     --save_similarity_matrix > logs/semantic_similarity_experiment_plans/"$cache_name".log 2>&1
# done


cache_dir="../cache_results_claude_july/ideas_emnlp/"
cache_names=("CSS" "Dialogue" "Discourse_Pragmatics" "Low_resource" "Ethics_Bias_Fairness" "Information_Extraction" "Information_Retrieval_Text_Mining" "Interpretability_Analysis" "Machine_Translation" "Multilinguality_Language_Diversity" "Multimodality_Language_Grounding" "Question_Answering" "Semantics" "Sentiment_Stylistic_Argument" "Speech_processing" "Summarization" "Efficiency_Model_Algorithms")

mkdir -p logs/dedup_emnlp

for cache_name in "${cache_names[@]}"; do
    echo "Running analyze_ideas_semantic_similarity.py with cache_name: $cache_name"
    python3 src/analyze_ideas_semantic_similarity.py \
    --cache_dir "$cache_dir" \
    --cache_name "$cache_name" \
    --save_similarity_matrix > logs/dedup_emnlp/semantic_similarity_"$cache_name".log 2>&1
done

