# Define an array of cache names
cache_names=("bias" "code_prompting" "factuality" "in_context_learning" "multi_step_prompting" "multimodal_bias" "multimodal_probing" "uncertainty")

# Number of ideas to generate
ideas_n=25

# Seed values
seeds=(12 2024)

#  Grounded Idea Gen
for seed in "${seeds[@]}"; do
    # Then iterate over each cache name
    for cache_name in "${cache_names[@]}"; do
        echo "Running grounded_idea_gen.py with cache_name: $cache_name and seed: $seed"
        python3 grounded_idea_gen.py --cache_name "$cache_name" --ideas_n $ideas_n --seed $seed
    done
done


# #  Experiment Plan Gen
# Iterate over each cache name and run the Python script
for cache_name in "${cache_names[@]}"; do
    echo "Running experiment_plan_gen.py with cache_name: $cache_name"
    python3 experiment_plan_gen.py --cache_name "$cache_name" --idea_name "all" --seed $seed
done


# #  Self Critique
# cache_names=("bias" "code_prompting" "factuality" "in_context_learning" "multi_step_prompting" "multimodal_bias" "multimodal_probing" "uncertainty")
# Iterate over each cache name
for cache_name in "${cache_names[@]}"; do
    echo "Running self_critique.py with cache_name: $cache_name and idea_name: all"
    python3 self_critique.py --cache_name "$cache_name" --idea_name "all"
done


# # novelty check retrieval and self-improvement
# # Iterate over each cache name
for cache_name in "${cache_names[@]}"; do
    echo "Running self_improvement.py with cache_name: $cache_name and idea_name: all"
    python3 self_improvement.py --cache_name "$cache_name" --idea_name "all"
done


# # Iterate over each cache name
for cache_name in "${cache_names[@]}"; do
    echo "Running novelty_check.py with cache_name: $cache_name and idea_name: all"
    python3 novelty_check.py --cache_name "$cache_name" --idea_name "all"
done