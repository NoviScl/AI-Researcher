cache_names=("factuality_prompting_method_prompting")

for cache_name in "${cache_names[@]}"; do
    echo "Running novelty_check.py with cache_name: $cache_name"
    python3 src/novelty_check.py --engine "claude-3-opus-20240229" --cache_name "$cache_name" --idea_name "all" --retrieve --novelty --check_n 10
done



