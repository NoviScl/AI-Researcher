cache_names=("uncertainty_prompting")

for cache_name in "${cache_names[@]}"; do
    echo "Running dedup with cache_name: $cache_name"
    python3 src/dedup_ideas.py --cache_name "$cache_name" > logs/dedup_ideas_"$cache_name".log 2>&1
done


cache_names=("uncertainty_prompting_RAG")

for cache_name in "${cache_names[@]}"; do
    echo "Running dedup with cache_name: $cache_name"
    python3 src/dedup_ideas.py --cache_name "$cache_name" > logs/dedup_ideas_"$cache_name".log 2>&1
done
