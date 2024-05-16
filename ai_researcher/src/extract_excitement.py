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
    prompt = "I have received reviews on a paper and I want you to help me decide whether the reviewers think the paper is exciting and impactful.\n" 
    prompt += "The full reviews are:\n"
    prompt += reviews + "\n"
    prompt += "The paper should be considered exciting (for which you should return yes) if at least one review explicitly mentioned that the idea is exciting / impactful (or similar sentiment) and none of the other reviewers disagrees.\n"
    prompt += "The paper should be considered not exciting (for which you should return no) if at least one review explicitly mentioned that the idea is not exciting enough and none of the other reviewers disagrees.\n"
    prompt += "If there is no mention about excitement or potential impact, or if there is disagreement, you should return neutral.\n"
    prompt += "Return a single answer (yes / no / neutral) on whether the paper is exciting with no other explanation.\n"

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

    excitement_yes = 0 
    excitement_no = 0
    for filename in tqdm(filenames):
        with open("../{}/{}".format(args.cache_name, filename), "r") as f:
            paper = json.load(f)
        
        # if paper["novelty_label"] in ["yes", "no"] and paper["excitement_label"] in ["yes", "no"]:
        #     print (filename, avg_score(paper["scores"]))
        #     print (paper["title"])
        #     print ("novelty: ", paper["novelty_label"])
        #     print ("excitement: ", paper["excitement_label"])
        #     print ("\n\n")

        try:
            reviews = concat_reviews(paper)
            prompt, response, cost = extract_excitement(reviews, client, args.engine, args.seed)
            # print (prompt)
            # print ("\n\n")
            # print (response)
            # print (cost)
            
            excitement = ""
            if response.strip().lower() == "yes":
                excitement_yes += 1
                excitement = "yes"
            elif response.strip().lower() == "no":
                excitement_no += 1
                excitement = "no"
            else:
                excitement = "neutral"

            # response = json.loads(response)
            paper["excitement_label"] = excitement

            with open("../{}/{}".format(args.cache_name, filename), "w") as f:
                json.dump(paper, f, indent=4)
            
            # print (filename, cost)
            
        except: 
            continue 
    
    print ("excitement yes: ", excitement_yes)
    print ("excitement no: ", excitement_no)

