# python3 experiment_plan_gen.py \
#  --cache_name "bias" \
#  --idea_name all \
#  --seed 2024

#!/bin/bash

# Define an array of cache names
cache_names=("uncertainty_prompting_RAG")

# Seed value
seed=2024

# Iterate over each cache name and run the Python script
for cache_name in "${cache_names[@]}"; do
    echo "Running experiment_plan_gen.py with cache_name: $cache_name"
    python3 src/experiment_plan_gen.py --engine "claude-3-5-sonnet-20240620" --cache_name "$cache_name" --idea_name "all" --seed $seed --method "prompting" > logs/experiment_plan_gen_"$cache_name".log 2>&1
done


