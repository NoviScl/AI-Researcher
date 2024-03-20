# python3 self_improvement.py --cache_name "bias" --idea_name "Bias by Association"

idea_name="Conceptual Meshing Prompting"
cache_names=("factuality_method_prompting")
# Iterate over each cache name
for cache_name in "${cache_names[@]}"; do
    echo "Running self_improvement.py with cache_name: $cache_name and idea name: $idea_name"
    python3 src/self_improvement.py --cache_name "$cache_name" --idea_name "$idea_name" 
done

