# cache_names=("bias")
# # Iterate over each cache name
# for cache_name in "${cache_names[@]}"; do
#     echo "Running self_improvement.py with cache_name: $cache_name and idea_name: all"
#     python3 self_improvement.py --cache_name "$cache_name" --idea_name "all"
# done

python3 add_test_cases.py --cache_name "code_prompting" --idea_name "Bilingual Code Prompting"

