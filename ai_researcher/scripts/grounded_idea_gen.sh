# cache_names=("attack_method" "bias_method" "defense_method" "factuality_method" "multilingual_method" "multimodal_method" "reasoning_method" "uncertainty_method")

# # Number of ideas to generate
# ideas_n=10

# # Seed value
# seeds=(2023 2024)
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


cache_names=("bias_method" "reasoning_method")

# Number of ideas to generate
ideas_n=5

# Seed value
seeds=(2024)
methods=("prompting")

# Iterate over each seed
for seed in "${seeds[@]}"; do
    # Iterate over each cache name 
    for cache_name in "${cache_names[@]}"; do
        # Iterate over each method 
        for method in "${methods[@]}"; do
            echo "Running grounded_idea_gen.py with cache_name: $cache_name"
            python3 src/grounded_idea_gen.py --engine "claude-3-opus-20240229" --cache_name "$cache_name" --grounding_k 10 --method "$method" --ideas_n $ideas_n --seed $seed
        done
    done
done 


