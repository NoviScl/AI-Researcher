# python3 grounded_idea_gen.py \
#  --cache_name "bias" \
#  --ideas_n 20 \
#  --seed 2024



#!/bin/bash

# Define an array of cache names
cache_names=("bias" "code_prompting" "factuality" "in_context_learning" "multi_step_prompting" "multimodal_bias" "multimodal_probing" "uncertainty")

# Number of ideas to generate
ideas_n=25

# Seed value
seed=2024

# Iterate over each cache name and run the Python script
for cache_name in "${cache_names[@]}"; do
    echo "Running grounded_idea_gen.py with cache_name: $cache_name"
    python3 grounded_idea_gen.py --cache_name "$cache_name" --ideas_n $ideas_n --seed $seed
done
