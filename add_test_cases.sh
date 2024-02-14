cache_names=("code_prompting")
# Iterate over each cache name
for cache_name in "${cache_names[@]}"; do
    echo "Running add_test_cases.py with cache_name: $cache_name and idea_name: all"
    python3 add_test_cases.py --cache_name "$cache_name" --idea_name "all" --novelty_only true
done

# python3 add_test_cases.py --cache_name "code_prompting" --idea_name "Bilingual Code Prompting"

