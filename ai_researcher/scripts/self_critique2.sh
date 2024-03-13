cache_names=("factuality_prompting_method")
# Iterate over each cache name
for cache_name in "${cache_names[@]}"; do
    echo "Running self_critique.py with cache_name: $cache_name and idea_name: all"
    python3 src/self_critique.py --cache_name "$cache_name" --idea_name "all"
done