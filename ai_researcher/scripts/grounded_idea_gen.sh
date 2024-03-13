# python3 grounded_idea_gen.py \
#  --cache_name "bias" \
#  --ideas_n 20 \
#  --seed 2024



#!/bin/bash

# Define an array of cache names
# cache_names=("bias_analysis" "adversarial_attack_method" "factuality_prompting_method" "uncertainty_method")
cache_names=("uncertainty_method")

# Number of ideas to generate
ideas_n=10

# Seed value
seeds=(2023)

for seed in "${seeds[@]}"; do
    # Iterate over each cache name and run the Python script
    for cache_name in "${cache_names[@]}"; do
        echo "Running grounded_idea_gen.py with cache_name: $cache_name"
        python3 src/grounded_idea_gen.py --cache_name "$cache_name" --ideas_n $ideas_n --seed $seed
    done
done 
