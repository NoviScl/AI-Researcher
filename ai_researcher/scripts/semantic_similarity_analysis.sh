# cache_names=("bias" "coding" "factuality" "math" "multilingual" "safety" "uncertainty")

# for cache_name in "${cache_names[@]}"; do
#     echo "Running analyze_ideas2.py with cache_name: $cache_name"
#     python3 src/analyze_ideas2.py --cache_name "$cache_name" > logs/semantic_similarity_"$cache_name".log 2>&1
# done


cache_dir="../cache_results_claude_may/ideas_5k/"
cache_names=("bias_prompting_method_RAG" "bias_prompting_method" "coding_prompting_method_RAG" "coding_prompting_method" "factuality_prompting_method_RAG" "factuality_prompting_method" "math_prompting_method_RAG" "math_prompting_method" "multilingual_prompting_method_RAG" "multilingual_prompting_method" "safety_prompting_method_RAG" "safety_prompting_method" "uncertainty_prompting_method_RAG" "uncertainty_prompting_method")

for cache_name in "${cache_names[@]}"; do
    echo "Running analyze_ideas_semantic_similarity.py with cache_name: $cache_name"
    python3 src/analyze_ideas_semantic_similarity.py \
    --cache_dir "$cache_dir" \
    --cache_name "$cache_name" \
    --save_similarity_matrix > logs/semantic_similarity_"$cache_name".log 2>&1
done

