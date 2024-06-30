# cache_names=("bias" "coding" "factuality" "math" "multilingual" "safety" "uncertainty")

# for cache_name in "${cache_names[@]}"; do
#     echo "Running analyze_ideas2.py with cache_name: $cache_name"
#     python3 src/analyze_ideas2.py --cache_name "$cache_name" > logs/semantic_similarity_"$cache_name".log 2>&1
# done


cache_names=("uncertainty")

for cache_name in "${cache_names[@]}"; do
    echo "Running analyze_ideas2.py with cache_name: $cache_name"
    python3 src/analyze_ideas_semantic_similarity.py --cache_name "$cache_name" > logs/semantic_similarity_"$cache_name".log 2>&1
done

