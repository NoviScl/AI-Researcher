import json
import os

if __name__ == "__main__":
    cache_names = ["bias", "code_prompting", "factuality", "in_context_learning", "multi_step_prompting", "multimodal_bias", "multimodal_probing", "uncertainty"]
    for cache_name in cache_names:
        print (cache_name)
        counter = 0
        filenames = os.listdir("cache_results/experiment_plans/"+cache_name)
        for filename in filenames:
            cache_file = os.path.join("cache_results/experiment_plans/"+cache_name, filename)
            with open(cache_file, "r") as f:
                ideas = json.load(f)
            if ideas["novelty"] == "yes":
                counter += 1
        print (counter)