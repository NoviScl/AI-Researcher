# python3 grounded_idea_gen.py \
#  --cache_name "bias" \
#  --ideas_n 20 \
#  --seed 2024



#!/bin/bash

# Define an array of cache names
# cache_names=("attack_prompting_method" "bias_prompting_method" "defense_prompting_method" "factuality_prompting_method" "multilingual_prompting_method" "multimodal_prompting_method" "reasoning_prompting_method" "uncertainty_prompting_method")
cache_names=("factuality_method")

# Number of ideas to generate
ideas_n=15

# Seed value
seeds=(2023)

for seed in "${seeds[@]}"; do
    # Iterate over each cache name and run the Python script
    for cache_name in "${cache_names[@]}"; do
        echo "Running grounded_idea_gen.py with cache_name: $cache_name"
        python3 src/grounded_idea_gen.py --engine "gpt-4-1106-preview" --cache_name "$cache_name" --grounding_k 10 --method "finetuning" --ideas_n $ideas_n --seed $seed
    done
done 
