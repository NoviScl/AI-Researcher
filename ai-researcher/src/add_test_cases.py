from openai import OpenAI
from utils import call_api
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing
from utils import cache_output, format_plan_json
from tqdm import tqdm
import random 
import retry
random.seed(2024)


@retry.retry(tries=3, delay=2)
def generate_test_cases(experiment_plan, test_case_demos, openai_client, model, seed):
    ## use gpt4 to generate test cases
    prompt = "You are a researcher specialized in Natural Language Processing. I will give you a research project proposal to work on, and your task is to come up with a few concrete test cases based on the project description.\n"
    prompt += "The project proposal is:\n" + experiment_plan.strip() + "\n"
    prompt += "You should genearte 3 - 5 test examples for this project. I will give two examples to illustrate what the test examples should look like:\n" + test_case_demos + "\n"
    prompt += "Now directly genearte the test cases following the above format."
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4000, seed=seed, json_output=False)
    return prompt, response, cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default=None, required=True, help='cache file name for the retrieved papers')
    parser.add_argument('--idea_name', type=str, default=None, required=True, help='the specific idea to be formulated into an experiment plan')
    parser.add_argument('--load_papers_from_cache', type=bool, default=False, help='whether to load the retrieved papers from cache')
    parser.add_argument('--novelty_only', type=bool, default=False, help='whether to only process papers that passed the novelty check')
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()

    with open("keys.json", "r") as f:
        keys = json.load(f)

    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    S2_KEY = keys["s2_key"]
    openai_client = OpenAI(
        organization=ORG_ID,
        api_key=OAI_KEY
    )

    with open("test_cases.txt", "r") as f:
        test_case_demos = f.read().strip()

    if args.idea_name == "all":
        filenames = os.listdir("cache_results/experiment_plans/"+args.cache_name)
    else:
        filenames = ["_".join(args.idea_name.lower().split())+".json"]

    for filename in tqdm(filenames):
        print ("working on: ", filename)
        ## load the idea
        cache_file = os.path.join("cache_results/experiment_plans/"+args.cache_name, filename)
        with open(cache_file, "r") as f:
            ideas = json.load(f)

        topic_description = ideas["topic_description"]
        idea = ideas["final_plan_json"]
        idea = format_plan_json(idea)
        idea_name = ideas["idea_name"]

        if args.novelty_only and ideas["novelty"] == "no":
            print ("skipping this idea because it's not novel")
            continue 
        
        ## generate test cases
        prompt, response, cost = generate_test_cases(idea, test_case_demos, openai_client, args.engine, args.seed)
        print (prompt + "\n")
        print (response + "\n")
        print (cost)

        ideas["final_plan_json"]["Test Cases"] = response.strip()
        cache_output(ideas, cache_file)

    

    
    
