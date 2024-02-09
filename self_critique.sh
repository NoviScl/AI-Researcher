# python3 self_critique.py --cache_name "bias" --idea_name "all"

# python3 self_critique.py --cache_name "code_prompting" --idea_name "all"

# python3 self_critique.py --cache_name "uncertainty" --idea_name "all"


cache_names=("factuality")
# Iterate over each cache name
for cache_name in "${cache_names[@]}"; do
    echo "Running self_critique.py with cache_name: $cache_name and idea_name: all"
    python3 self_critique.py --cache_name "$cache_name" --idea_name "all"
done


