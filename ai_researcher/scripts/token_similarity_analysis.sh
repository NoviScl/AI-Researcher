cache_names=("bias" "coding" "factuality" "math" "multilingual" "safety" "uncertainty")

for cache_name in "${cache_names[@]}"; do
    echo "Running analyze_ideas.py with cache_name: $cache_name"
    python3 src/analyze_ideas.py --cache_name "$cache_name" > logs/token_similarity_"$cache_name".log 2>&1
done
