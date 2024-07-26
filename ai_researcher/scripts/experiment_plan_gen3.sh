# idea_cache_dir="../cache_results_claude_may/ideas_5k_dedup/"
# experiment_plan_cache_dir="../cache_results_claude_may/experiment_plans_5k_dedup/"
# cache_names=("multilingual_prompting_method_RAG" "factuality_prompting_method_RAG" "factuality_prompting_method" "math_prompting_method_RAG" "math_prompting_method")

# # Seed value
# seed=2024

# # Iterate over each cache name and run the Python script
# for cache_name in "${cache_names[@]}"; do
#     echo "Running experiment_plan_gen.py with cache_name: $cache_name"
#     python3 src/experiment_plan_gen.py \
#     --engine "claude-3-5-sonnet-20240620" \
#     --idea_cache_dir "$idea_cache_dir" \
#     --cache_name "$cache_name" \
#     --experiment_plan_cache_dir "$experiment_plan_cache_dir" \
#     --idea_name "all" \
#     --seed $seed \
#     --method "prompting" > logs/experiment_plan_gen_"$cache_name".log 2>&1
# done


idea_cache_dir="../cache_results_claude_july/ideas_emnlp_dedup/"
experiment_plan_cache_dir="../cache_results_claude_july/experiment_plans_emnlp_dedup/"
cache_names=("Question_Answering" "Semantics" "Sentiment_Stylistic_Argument" "Speech_processing" "Summarization" "Efficiency_Model_Algorithms")

# Seed value
seed=2024

mkdir -p logs/experiment_plan_gen_emnlp

# Iterate over each cache name and run the Python script
for cache_name in "${cache_names[@]}"; do
    echo "Running experiment_plan_gen.py with cache_name: $cache_name"
    python3 src/experiment_plan_gen.py \
    --engine "claude-3-5-sonnet-20240620" \
    --idea_cache_dir "$idea_cache_dir" \
    --cache_name "$cache_name" \
    --experiment_plan_cache_dir "$experiment_plan_cache_dir" \
    --idea_name "all" \
    --seed $seed \
    --method "prompting" > logs/experiment_plan_gen_emnlp/experiment_plan_gen_"$cache_name".log 2>&1
done

# tmux 3