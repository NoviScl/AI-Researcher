from openai import OpenAI
from utils import call_api
import argparse
import json
import os
from utils import cache_output, format_plan_json
import random 
from tqdm import tqdm
import retry
random.seed(2024)

@retry.retry(tries=3, delay=2)
def excitement_score(experiment_plan_lst, criteria, openai_client, model, seed):
    prompt = "You are a professor specialized in Natural Language Processing. You received several project proposals from your students and your job is to give a score to each project proposal to judge whether it is good enough to be accepted by the ACL conference.\n"
    prompt += "The project proposals are:\n" 
    for i in range(1, len(experiment_plan_lst)+1):
        prompt += str(i) + "\n"
        prompt += format_plan_json(experiment_plan_lst[i-1]).strip() + "\n\n"
    prompt += "Your should follow these scoring rubrics:\n" + criteria.strip() + "\n"
    prompt += "Assign a score of 1 to 5 (integer score only) to each project proposal. Note that the conference has an acceptance rate of 20%, meaning that only the best project out of the five candidates can get a score above 3. So you should be critical about the scoring.\n"
    prompt += "Return the scoring in json format. The key is the index of the project proposal, and the value is the scoring, which should be a short explanation followed by the numeric score. For example, if you think the first project proposal is good enough to be accepted, you should return {\"1\": \"This project is a well-motivated study on the comparison of search engines and language models for fact-checking with rigorous experiment design. It is very timely to fill in an important gap in current large language model research.\nscore: 4\"}.\n"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4000, seed=seed, json_output=True)
    return prompt, response, cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default=None, required=True, help='cache file name for the retrieved papers')
    parser.add_argument('--idea_name', type=str, default=None, required=True, help='the specific idea to be formulated into an experiment plan')
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

    with open("excitement_ranking.txt", "r") as f:
        criteria = f.read().strip()

    if args.idea_name == "all":
        filenames = os.listdir("cache_results/experiment_plans/"+args.cache_name)
    else:
        filenames = ["_".join(args.idea_name.lower().split())+".json"]
    
    ## maintain a dict to track all files that pass the novelty check 
    passed_files = {}
    for filename in filenames:
        ## load the idea
        cache_file = os.path.join("cache_results/experiment_plans/"+args.cache_name, filename)
        with open(cache_file, "r") as f:
            ideas = json.load(f)
        if ideas["novelty"] == "yes":
            passed_files[filename] = ideas
    filenames = list(passed_files.keys())

    ## process in batch of 5
    batch_size = 5
    for i in tqdm(range(0, len(filenames), batch_size)):
        batch_files = filenames[i:i+batch_size]
        batch_ideas = [passed_files[filename] for filename in batch_files]
        experiment_plan_lst = [idea["final_plan_json"] for idea in batch_ideas]
        prompt, response, cost = excitement_score(experiment_plan_lst, criteria, openai_client, args.engine, args.seed)
        print (prompt)
        print (response)
        print (cost)

        response = json.loads(response)
        for idx, filename in enumerate(batch_files):
            passed_files[filename]["excitement_rationale"] = response[str(idx+1)]
            passed_files[filename]["excitement_score"] = int(response[str(idx+1)].split()[-1])
            cache_output(passed_files[filename], os.path.join("cache_results/experiment_plans/"+args.cache_name, filename))


    
