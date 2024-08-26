## example usage
cache_dir="../cache_results_test/project_proposals/"
passed_cache_dir="../cache_results_test/project_proposals_passed/"
cache_names=("factuality_prompting_method")
seed=2024

# Iterate over each cache name and run the Python script
for cache_name in "${cache_names[@]}"; do
    echo "Running filter_ideas.py with cache_name: $cache_name"
    python3 src/filter_ideas.py \
    --engine "claude-3-5-sonnet-20240620" \
    --cache_dir "$cache_dir" \
    --cache_name "$cache_name" \
    --passed_cache_dir "$passed_cache_dir" \
    --score_file "../cache_results_test/ranking/$cache_name/round_5.json" 
done
