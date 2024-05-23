from openai import OpenAI
import anthropic
from utils import call_api
import argparse
import json
import os
from utils import cache_output, format_plan_json, avg_score
import random 
from tqdm import tqdm
import retry
random.seed(2024)

@retry.retry(tries=3, delay=2)
def overall_score(experiment_plan, criteria, openai_client, model, seed):
    prompt = "You are a professor specialized in Natural Language Processing and Large Language Models. You are given a project proposal and you need to score it.\n"
    prompt += "The project proposal is:\n\n" 
    prompt += format_plan_json(experiment_plan)
    prompt += "\n\nYour should follow the scoring rubrics:\n" + criteria + "\n"
    prompt += "Now directly provide me with a final score between 1 and 10, no other explanation needed.\n"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=2, seed=seed, json_output=False)
    return prompt, response, cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default="openreview_benchmark", help='cache file name for the retrieved papers')
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

    with open("prompts/review_rubrics.txt", "r") as f:
        criteria = f.read().strip()

    filenames = os.listdir("../{}".format(args.cache_name))
    filenames = [f for f in filenames if f.endswith(".json") and '5' in f]

    score_predictions = {}

    for filename in tqdm(filenames):
        with open("../{}/{}".format(args.cache_name, filename), "r") as f:
            paper = json.load(f)
    
        # print (paper["structured_summary"])
        # print (format_plan_json(paper["structured_summary"]))

        try:
            experiment_plan = paper["structured_summary"]
            prompt, response, cost = overall_score(experiment_plan, criteria, client, args.engine, args.seed)
            
            # print (prompt)
            score_predictions[filename] = {"predicted_score": int(response.strip()), "actual_score": avg_score(paper["scores"])}
            print ("predicted score: ", response)
            print ("actual score: ", avg_score(paper["scores"]))
            print (cost)
        
        except: 
            continue 
    
    with open("score_predictions.json", "w") as f:
        json.dump(score_predictions, f, indent=4)
    
