#!/bin/bash

# Define an array of cache names
cache_names=("factuality_prompting_new_method_prompting")

# Seed value
seed=2024

# Iterate over each cache name and run the Python script
for cache_name in "${cache_names[@]}"; do
    echo "Running execution_code_gen.py with cache_name: $cache_name"
    python3 src/execution_code_gen.py --engine "claude-3-opus-20240229" --cache_name "$cache_name" --idea_name "attribute_grounding" --seed $seed 
done

