from openai import OpenAI
from utils import call_api
import argparse
import json
import os
from lit_review_tools import parse_and_execute, format_papers_for_printing, print_top_papers_from_paper_bank, dedup_paper_bank
from utils import cache_output, format_plan_json
import random 
from tqdm import tqdm
import retry
random.seed(2024)

@retry.retry(tries=3, delay=2)
def feasibility_check(experiment_plan, criteria, openai_client, model, seed):
    prompt = "You are a professor specialized in Natural Language Processing. A student has submitted a research project proposal to you and your job is to decide whether this project is feasible.\n"
    prompt += "The project proposal is:\n" + experiment_plan.strip() + "\n\n"
    prompt += "Here is a list of criteria that would make a project infeasible:\n" + criteria.strip() + "\n\n"
    prompt += "If the project violates any of the above criteria, it's counted as infeasible, and you should respond with a short explanation of which criteria it violates and why. If the project is feasible, you should respond with a short explanation of why it is feasible. Give the short explanation, then change to a new line, respond with the final judgment by saying either \"yes\" or \"no\"."
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return prompt, response, cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default=None, required=True, help='cache file name for the retrieved papers')
    parser.add_argument('--idea_name', type=str, default=None, required=True, help='the specific idea to be formulated into an experiment plan')
    parser.add_argument('--check_n', type=int, default=10, help="number of top papers to check for novelty")
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

    with open("feasibility_check.txt", "r") as f:
        criteria = f.read().strip()

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
        related_papers = ideas["novelty_check_papers"]

        prompt, response, cost = feasibility_check(idea, criteria, openai_client, args.engine, args.seed)
        print (prompt + "\n")
        print (response + "\n")
        print (cost)
    
        # cache_output(ideas, cache_file)

    
