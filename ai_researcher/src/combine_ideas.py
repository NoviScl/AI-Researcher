## script to combine RAG and no RAG experiment plans
import os
import json
import shutil

# experiment_plan_dir = "../cache_results_claude_may/experiment_plans_5k_dedup"
# cache_names = ["bias", "coding", "factuality", "math", "multilingual", "safety", "uncertainty"]

# for cache_name in cache_names:
#     merged_dir = os.path.join(experiment_plan_dir, cache_name + "_prompting_method_merged")
#     os.makedirs(merged_dir, exist_ok=True)

#     source_dirs = [
#         os.path.join(experiment_plan_dir, cache_name + "_prompting_method"),
#         os.path.join(experiment_plan_dir, cache_name + "_prompting_method_RAG")
#     ]

#     for source_dir in source_dirs:
#         for file_name in os.listdir(source_dir):
#             if file_name.endswith('.json'):
#                 source_file = os.path.join(source_dir, file_name)
#                 target_file = os.path.join(merged_dir, file_name)
#                 shutil.copy2(source_file, target_file)

#     total_files = len([name for name in os.listdir(merged_dir) if os.path.isfile(os.path.join(merged_dir, name))])
#     print ("cache_name: ", cache_name, ";  total_files: ", total_files)



ideas_dir = "../cache_results_claude_may/ideas_5k"
cache_names = ["bias", "coding", "factuality", "math", "multilingual", "safety", "uncertainty"]

for cache_name in cache_names:
    print (cache_name)
    with open(os.path.join(ideas_dir, cache_name + "_prompting_method.json"), "r") as f:
        no_rag_ideas = json.load(f)
        print ("#ideas: ", len(no_rag_ideas["ideas"]) * 5)
    with open(os.path.join(ideas_dir, cache_name + "_prompting_method_RAG.json"), "r") as f:
        rag_ideas = json.load(f)
        print ("#ideas: ", len(rag_ideas["ideas"]) * 5)
    
    all_ideas = {}
    all_ideas["topic_description"] = rag_ideas["topic_description"]
    all_ideas["ideas"] = no_rag_ideas["ideas"] + rag_ideas["ideas"]
    with open(os.path.join(ideas_dir, cache_name + "_prompting_method_merged.json"), "w") as f:
        json.dump(all_ideas, f, indent=4)

