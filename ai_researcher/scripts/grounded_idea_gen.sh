# python3 grounded_idea_gen.py \
#  --cache_name "bias" \
#  --ideas_n 20 \
#  --seed 2024



#!/bin/bash

# Define an array of cache names
# cache_names=("attack_prompting_method" "bias_prompting_method" "defense_prompting_method" "factuality_prompting_method" "multilingual_prompting_method" "multimodal_prompting_method" "reasoning_prompting_method" "uncertainty_prompting_method")
# cache_names=("attack_method" "bias_method" "defense_method" "factuality_method" "multilingual_method" "multimodal_method" "reasoning_method" "uncertainty_method")

# # Number of ideas to generate
# ideas_n=10

# # Seed value
# seeds=(12 42 2023 2024)
# methods=("prompting" "finetuning")

# # Iterate over each seed
# for seed in "${seeds[@]}"; do
#     # Iterate over each cache name 
#     for cache_name in "${cache_names[@]}"; do
#         # Iterate over each method 
#         for method in "${methods[@]}"; do
#             echo "Running grounded_idea_gen.py with cache_name: $cache_name"
#             python3 src/grounded_idea_gen.py --engine "gpt-4-1106-preview" --cache_name "$cache_name" --grounding_k 10 --method "$method" --ideas_n $ideas_n --seed $seed
#         done
#     done
# done 


cache_names=("bias_method" "defense_method" "factuality_method" "multilingual_method" "multimodal_method" "reasoning_method" "uncertainty_method")

# Number of ideas to generate
ideas_n=10

# Seed value
seeds=(42)
methods=("prompting" "finetuning")

# Iterate over each seed
for seed in "${seeds[@]}"; do
    # Iterate over each cache name 
    for cache_name in "${cache_names[@]}"; do
        # Iterate over each method 
        for method in "${methods[@]}"; do
            echo "Running grounded_idea_gen.py with cache_name: $cache_name"
            python3 src/grounded_idea_gen.py --engine "gpt-4-1106-preview" --cache_name "$cache_name" --grounding_k 10 --method "$method" --ideas_n $ideas_n --seed $seed
        done
    done
done 



cache_names=("attack_method" "bias_method" "defense_method" "factuality_method" "multilingual_method" "multimodal_method" "reasoning_method" "uncertainty_method")

# Number of ideas to generate
ideas_n=10

# Seed value
seeds=(2023 2024)
methods=("prompting" "finetuning")

# Iterate over each seed
for seed in "${seeds[@]}"; do
    # Iterate over each cache name 
    for cache_name in "${cache_names[@]}"; do
        # Iterate over each method 
        for method in "${methods[@]}"; do
            echo "Running grounded_idea_gen.py with cache_name: $cache_name"
            python3 src/grounded_idea_gen.py --engine "gpt-4-1106-preview" --cache_name "$cache_name" --grounding_k 10 --method "$method" --ideas_n $ideas_n --seed $seed
        done
    done
done 