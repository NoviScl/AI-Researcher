from openai import OpenAI
import anthropic
from utils import call_api, shuffle_dict_and_convert_to_string
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing
from utils import cache_output
import random 
import retry
from tqdm import tqdm 

@retry.retry(tries=3, delay=2)
def extract_structure(title, abstract, full_text, demos, openai_client, model, seed):
    prompt = "Help me summarize a full paper into a structured format. The full paper is: \n" 
    prompt += "Title: " + title + "\n"
    prompt += "Abstract: " + abstract + "\n"
    prompt += "Full paper: " + full_text + "\n\n"
    prompt += "I want a summary in the following format:\n" 
    prompt += "1. Title: Title of the paper.\n"
    prompt += "2. Problem Statement: Clearly define the problem the paper intends to address. Explain clearly why this problem is interesting and important.\n"
    prompt += "3. Motivation: Explain why existing methods are not good enough to solve the problem, and explain the inspiration behind the new proposed method. You should also mention why the proposed method would work better than existing baselines on the problem.\n"
    prompt += "4. Proposed Method: Explain how the proposed method works, describe all the essential steps.\n"
    prompt += "5. Step-by-Step Experiment Plan: Break down every single step of the experiments, make sure every step is executable. Cover all essential details such as the datasets, models, and metrics to be used. If the project involves prompting, give some example prompts for each step. If it's not a prompting paper, skip the \"Construct Prompts\" step and instead you should describe other details such as the model architecture, training objectives, or data construction pipeline based on the core contribution of the paper (feel free to add new sections as you deem appropriate).\n"
    prompt += "\n" + "Below is a few examples of how the structured summary should look like:\n\n" + demos + "\n\n"
    prompt += "Now please write down the structured summary in JSON format (keys should be the section names, just like the above examples). Make sure to be as detailed as possible."

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4096, seed=seed, json_output=True)
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
    filenames = [f for f in filenames if f.endswith(".json")]

    for filename in tqdm(filenames):
        with open("../{}/{}".format(args.cache_name, filename), "r") as f:
            paper = json.load(f)
        
        if "structured_summary" in paper:
            continue
        
        try:
            title = paper["title"].strip()
            abstract = paper["abstract"].strip()
            full_text = paper["full_text"].strip()
            prompt, response, cost = extract_structure(title, abstract, full_text, demos, client, args.engine, args.seed)
            # print (prompt)
            # print ("\n\n")
            # print (response)
            # print (cost)

            response = json.loads(response)
            paper["structured_summary"] = response

            with open("../{}/{}".format(args.cache_name, filename), "w") as f:
                json.dump(paper, f, indent=4)
            
            print (filename, cost)
            
        except: 
            continue 

