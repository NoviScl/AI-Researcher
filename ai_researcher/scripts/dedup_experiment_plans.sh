cache_dir="../cache_results_claude_may/experiment_plans_5k_dedup/"
cache_names=("bias_prompting_method_merged" "coding_prompting_method_merged" "factuality_prompting_method_merged" "math_prompting_method_merged" "multilingual_prompting_method_merged" "safety_prompting_method_merged" "uncertainty_prompting_method_merged")

for cache_name in "${cache_names[@]}"; do
    echo "Running dedup_experiment_plans.py with cache_name: $cache_name"
    python3 src/dedup_experiment_plans.py \
    --cache_dir "$cache_dir" \
    --cache_name "$cache_name" \
    --dedup_cache_dir "../cache_results_claude_may/experiment_plans_5k_merged_dedup" \
    --similarity_threshold 0.85 > logs/dedup_"$cache_name".log 2>&1
done