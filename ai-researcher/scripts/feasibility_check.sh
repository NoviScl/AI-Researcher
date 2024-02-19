cache_names=("code_prompting")
# Iterate over each cache name
for cache_name in "${cache_names[@]}"; do
    echo "Running feasibility_check.py with cache_name: $cache_name and idea_name: all"
    python3 feasibility_check.py --cache_name "$cache_name" --idea_name "all" --novelty_only true
done

# python3 feasibility_check.py --cache_name "code_prompting" --idea_name "Bilingual Code Prompting"

