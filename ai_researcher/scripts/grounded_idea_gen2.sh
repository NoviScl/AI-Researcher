# cache_names=("bias_prompting_method" "coding_prompting_method" "factuality_prompting_method" "math_prompting_method" "multilingual_prompting_method" "safety_prompting_method" "uncertainty_prompting_method")
cache_names=("uncertainty")
ideas_n=5
methods=("prompting")

# Iterate over each seed from 1 to 1000
for seed in {1..440}; do
    # Iterate over each cache name 
    for cache_name in "${cache_names[@]}"; do
        # Iterate over each method 
        for method in "${methods[@]}"; do
            echo "Running grounded_idea_gen.py with cache_name: $cache_name"
            python3 src/grounded_idea_gen.py --engine "claude-3-5-sonnet-20240620" --cache_name "$cache_name" --grounding_k 10 --method "$method" --ideas_n $ideas_n --seed $seed --RAG False
        done
    done
done