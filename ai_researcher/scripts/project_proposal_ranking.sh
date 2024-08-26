## example usage
experiment_plan_cache_dir="../cache_results_test/project_proposals/"
ranking_score_dir="../cache_results_test/ranking/"
cache_names=("factuality_prompting_method")
seed=2024

for cache_name in "${cache_names[@]}"; do
    echo "Running tournament_ranking.py with cache_name: $cache_name"
    python3 src/tournament_ranking.py \
    --engine claude-3-5-sonnet-20240620 \
    --experiment_plan_cache_dir "$experiment_plan_cache_dir" \
    --cache_name "$cache_name" \
    --ranking_score_dir "$ranking_score_dir" \
    --max_round 5 
done
