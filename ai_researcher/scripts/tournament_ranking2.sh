# experiment_plan_cache_dir="../cache_results_claude_may/experiment_plans_5k_merged_dedup/"
# cache_names=("bias_prompting_method_merged" "coding_prompting_method_merged" "factuality_prompting_method_merged")

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
cache_names=("Multilinguality_Language_Diversity" "Multimodality_Language_Grounding" "Question_Answering" "Semantics" "Sentiment_Stylistic_Argument" "Speech_processing" "Summarization" "Efficiency_Model_Algorithms")

# Seed value
seed=2024

# Iterate over each cache name and run the Python script
for cache_name in "${cache_names[@]}"; do
    echo "Running tournament_ranking.py with cache_name: $cache_name"
    python3 src/tournament_ranking.py \
    --engine claude-3-5-sonnet-20240620 \
    --experiment_plan_cache_dir "$experiment_plan_cache_dir" \
    --cache_name "$cache_name" \
    --max_round 5 > logs/ranking_emnlp/tournament_ranking_"$cache_name".log 2>&1
done

