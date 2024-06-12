import json 

with open("/nlp/scr/clsi/AI-Researcher/cache_results_claude_may/lit_review_new/uncertainty_prompting_method.json", "r") as f:
    data = json.load(f) 

paper_bank = data["paper_bank"]
for i in range(20):
    print ("Uncertainty Paper #" + str(i+1))
    print ("Title: " + paper_bank[i]["title"])
    print ("Abstract: " + paper_bank[i]["abstract"])
    print ("\n\n")