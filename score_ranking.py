import json 
import os

def rank_dict_by_score(input_dict):
    # Sorting the dictionary by "score" in descending order
    ranked_dict = sorted(input_dict.items(), key=lambda x: x[1]["excitement_score"], reverse=True)
    # Convert the sorted items back to a dictionary
    return dict(ranked_dict)

if __name__ == "__main__":
    cache_name = "code_prompting"
    filenames = os.listdir("cache_results/experiment_plans/" + cache_name)
    
    ## maintain a dict to track all files that pass the novelty check 
    passed_files = {}
    for filename in filenames:
        ## load the idea
        cache_file = os.path.join("cache_results/experiment_plans/" + cache_name, filename)
        with open(cache_file, "r") as f:
            ideas = json.load(f)
        if ideas["novelty"] == "yes":
            passed_files[filename] = ideas
    
    ## rank the ideas by excitement score
    ranked_files = rank_dict_by_score(passed_files)
    for k,v in ranked_files.items():
        print (v["idea_name"], v["excitement_score"])
    
