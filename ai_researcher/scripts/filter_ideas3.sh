cache_dir="../cache_results_claude_may/experiment_plans_5k_dedup"
passed_cache_dir="../cache_results_claude_may/experiment_plans_5k_dedup_passed"
cache_names=("multilingual_prompting_method" "multilingual_prompting_method_RAG" "safety_prompting_method" "safety_prompting_method_RAG" "uncertainty_prompting_method" "uncertainty_prompting_method_RAG")
# cache_names=("safety_prompting_method_RAG")

# Seed value
seed=2024

# Iterate over each cache name and run the Python script
for cache_name in "${cache_names[@]}"; do
    echo "Running filter_ideas.py with cache_name: $cache_name"
    python3 src/filter_ideas.py \
    --engine "claude-3-5-sonnet-20240620" \
    --cache_dir "$cache_dir" \
    --cache_name "$cache_name" \
    --passed_cache_dir "$passed_cache_dir" \
    --score_file "logs/ranking_score_predictions/$cache_name/round_5.json" > logs/filter_ideas/$cache_name.log 2>&1
done

## tmux 10