from openai import OpenAI
import anthropic
from utils import call_api, shuffle_dict_and_convert_to_string
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing
from utils import cache_output, concat_reviews, avg_score
import random 
import retry
from tqdm import tqdm 

@retry.retry(tries=3, delay=2)
def extract_excitement(reviews, openai_client, model, seed):
    prompt = "I have received reviews on a paper and I want you to help me decide whether the reviewers think the paper is novel.\n" 
    prompt += "The full reviews are:\n"
    prompt += reviews + "\n"
    prompt += "The paper should be considered novel (for which you should return yes) if at least one review explicitly mentioned that the idea is novel and none of the other reviewers thinks that it is not novel.\n"
    prompt += "The paper should be considered not novel (for which you should return no) if at least one review explicitly mentioned that the idea is not novel and none of the other reviewers thinks that it is novel.\n"
    prompt += "If there is no mention about novelty at all, or if there is disagreement, you should return neutral.\n"
    prompt += "Return a single answer (yes / no / neutral) on whether the paper is novel with no other explanation.\n"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=2, seed=seed, json_output=False)
    return prompt, response, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-opus-20240229', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default=None, required=True, help='cache file name for the retrieved papers')
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()

    with open("../keys.json", "r") as f:
        keys = json.load(f)
    random.seed(args.seed)

    ANTH_KEY = keys["anthropic_key"]
    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    
    if "claude" in args.engine:
        client = anthropic.Anthropic(
            api_key=ANTH_KEY,
        )
    else:
        client = OpenAI(
            organization=ORG_ID,
            api_key=OAI_KEY
        )

    with open("prompts/paper_summary_demos.txt", "r") as f:
        demos = f.read()
    
    filenames = os.listdir("../{}".format(args.cache_name))

    ## sample a subset 
    filenames = [f for f in filenames if f.endswith(".json") and '5' in f]

    novelty_yes = 0 
    novelty_no = 0
    for filename in tqdm(filenames):
        with open("../{}/{}".format(args.cache_name, filename), "r") as f:
            paper = json.load(f)
        
        # if paper["novelty_label"] == "no" and "reject" in paper["decision"].lower():
        #     print (filename, avg_score(paper["scores"]), paper["title"])

        try:
            reviews = concat_reviews(paper)
            prompt, response, cost = extract_novelty(reviews, client, args.engine, args.seed)
            # print (prompt)
            # print ("\n\n")
            # print (response)
            # print (cost)
            
            novelty = ""
            if response.strip().lower() == "yes":
                novelty_yes += 1
                novelty = "yes"
            elif response.strip().lower() == "no":
                novelty_no += 1
                novelty = "no"
            else:
                novelty = "neutral"

            # response = json.loads(response)
            paper["novelty_label"] = novelty

            with open("../{}/{}".format(args.cache_name, filename), "w") as f:
                json.dump(paper, f, indent=4)
            
            # print (filename, cost)
            
        except: 
            continue 
    
    print ("Novelty yes: ", novelty_yes)
    print ("Novelty no: ", novelty_no)

