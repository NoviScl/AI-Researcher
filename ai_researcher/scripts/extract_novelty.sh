cache_names=("openreview_benchmark")

# Iterate over each cache name
for cache_name in "${cache_names[@]}"; do
    echo "Running extract_novelty.py with cache_name: $cache_name"
    python3 src/extract_novelty.py --engine "gpt-4o" --cache_name "$cache_name"
done
