cache_dir="../cache_results_claude_may/ideas_5k/"
cache_names=("bias_prompting_method_RAG" "bias_prompting_method" "coding_prompting_method_RAG" "coding_prompting_method" "factuality_prompting_method_RAG" "factuality_prompting_method" "math_prompting_method_RAG" "math_prompting_method" "multilingual_prompting_method_RAG" "multilingual_prompting_method" "safety_prompting_method_RAG" "safety_prompting_method" "uncertainty_prompting_method_RAG" "uncertainty_prompting_method")

for cache_name in "${cache_names[@]}"; do
    echo "Running dedup_ideas.py with cache_name: $cache_name"
    python3 src/dedup_ideas.py \
    --cache_dir "$cache_dir" \
    --cache_name "$cache_name" \
    --dedup_cache_dir "../cache_results_claude_may/ideas_5k_dedup" \
    --similarity_threshold 0.8 > logs/dedup_"$cache_name".log 2>&1
done