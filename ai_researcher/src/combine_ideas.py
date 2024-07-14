## script to combine RAG and no RAG experiment plans
import os
import json
import shutil

experiment_plan_dir = "../cache_results_claude_may/experiment_plans_5k_dedup"
cache_names = ["bias", "coding", "factuality", "math", "multilingual", "safety", "uncertainty"]

for cache_name in cache_names:
    merged_dir = os.path.join(experiment_plan_dir, cache_name + "_prompting_method_merged")
    os.makedirs(merged_dir, exist_ok=True)

    source_dirs = [
        os.path.join(experiment_plan_dir, cache_name + "_prompting_method"),
        os.path.join(experiment_plan_dir, cache_name + "_prompting_method_RAG")
    ]

    for source_dir in source_dirs:
        for file_name in os.listdir(source_dir):
            if file_name.endswith('.json'):
                source_file = os.path.join(source_dir, file_name)
                target_file = os.path.join(merged_dir, file_name)
                shutil.copy2(source_file, target_file)

    total_files = len([name for name in os.listdir(merged_dir) if os.path.isfile(os.path.join(merged_dir, name))])
    print ("cache_name: ", cache_name, ";  total_files: ", total_files)
