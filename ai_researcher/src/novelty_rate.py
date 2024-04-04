import json 
import os

cache_dir = "../../cache_results_claude/experiment_plans/factuality_prompting_new_method_prompting"

novel_idea = 0
for filename in os.listdir(cache_dir):
    with open(os.path.join(cache_dir, filename), "r") as f:
        ideas = json.load(f)
    if ideas["novelty"] == "yes":
        novel_idea += 1
print ("novelty rate: {} / {} = {}%".format(novel_idea, len(os.listdir(cache_dir)), novel_idea / len(os.listdir(cache_dir)) * 100))
