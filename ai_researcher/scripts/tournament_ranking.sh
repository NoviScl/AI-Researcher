# experiment_plan_cache_dir="../cache_results_claude_may/experiment_plans_5k_merged_dedup/"
# cache_names=("math_prompting_method_merged" "multilingual_prompting_method_merged" "safety_prompting_method_merged" "uncertainty_prompting_method_merged")

# # Seed value
# seed=2024

# # Iterate over each cache name and run the Python script
# for cache_name in "${cache_names[@]}"; do
#     echo "Running tournament_ranking.py with cache_name: $cache_name"
#     python3 src/tournament_ranking.py \
#     --engine claude-3-5-sonnet-20240620 \
#     --experiment_plan_cache_dir "$experiment_plan_cache_dir" \
#     --cache_name "$cache_name" \
#     --max_round 6 > logs/tournament_ranking_"$cache_name".log 2>&1
# done


experiment_plan_cache_dir="../cache_results_claude_july/experiment_plans_emnlp_dedup/"
cache_names=("Dialogue" "Discourse_Pragmatics" "Low_resource" "Ethics_Bias_Fairness" "Information_Extraction" "Information_Retrieval_Text_Mining" "Interpretability_Analysis" "Machine_Translation")

# Seed value
seed=2024

mkdir -p logs/ranking_emnlp

# Iterate over each cache name and run the Python script
for cache_name in "${cache_names[@]}"; do
    echo "Running tournament_ranking.py with cache_name: $cache_name"
    python3 src/tournament_ranking.py \
    --engine claude-3-5-sonnet-20240620 \
    --experiment_plan_cache_dir "$experiment_plan_cache_dir" \
    --cache_name "$cache_name" \
    --max_round 5 > logs/ranking_emnlp/tournament_ranking_"$cache_name".log 2>&1
done



