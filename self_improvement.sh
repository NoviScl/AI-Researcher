# python3 self_improvement.py --cache_name "bias" --idea_name "Bias by Association"

cache_names=("bias")
# Iterate over each cache name
for cache_name in "${cache_names[@]}"; do
    echo "Running self_improvement.py with cache_name: $cache_name and idea_name: all"
    python3 self_improvement.py --cache_name "$cache_name" --idea_name "all"
done

