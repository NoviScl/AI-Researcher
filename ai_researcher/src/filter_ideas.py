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
from collections import defaultdict
random.seed(2024)

@retry.retry(tries=3, delay=2)
def novelty(experiment_plan, openai_client, model, seed):
    prompt = "You are a professor specialized in Natural Language Processing and Large Language Models. You are given a project proposal and you need to decide whether it is novel enough.\n"
    prompt += "The project proposal is:\n\n" 
    prompt += format_plan_json(experiment_plan)
    prompt += "\nReturn yes if the project is significantly different from existing work (both classic ones and recent ones), otherwise return no. Give a short rationale and then change to a new line to return either yes or no and then end the response.\n"
    prompt += "In the rationale, reference the most similar works and explain how the proposed project is similar to or different from them. You should return no if the proposed project is only a minor modification or combination of the existing ideas.\n"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=3000, seed=seed, json_output=False)
    return prompt, response, cost


@retry.retry(tries=3, delay=2)
def consistency(experiment_plan, openai_client, model, seed):
    prompt = "You are a professor specialized in Natural Language Processing and Large Language Models. You are given a project proposal and you need to decide whether it is consistent in the methodology and experiment design.\n"
    prompt += "The project proposal is:\n\n" 
    prompt += format_plan_json(experiment_plan)
    prompt += "\nYou should return no if the proposed method is based on prompting and assumes black-box model access but proposes any step that involves white-box model access, such as: getting input embedding, training the model, applying any loss functions, constrastive learning, performing dropout on the model, extracting or modifying internal models activations, computing influence functions, and any other operations that require model weight or data access. The only exception is that finetuning a small open-source model is fine, but the proposal should explicitly mention that the model being trained is open.\n"
    prompt += "Only return yes if the proposed method is either: 1) based only on open-source models; or 2) based on prompting black-box models (GPT, Claude, Gemini, etc.) and all steps do not require white-box model or data access as explained above. Give a short explanation first and then change to a new line to return either yes or no and then end the response.\n"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=3000, seed=seed, json_output=False)
    return prompt, response, cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-5-sonnet-20240620', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default="uncertainty_prompting_method_prompting", help='cache file name for the retrieved papers')
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

    # filenames = os.listdir("../cache_results_claude_may/experiment_plans_1k/{}".format(args.cache_name))
    # filenames = [f for f in filenames if f.endswith(".json")]
    
    with open("logs/uncertainty_score_predictions_swiss_round_5.json", "r") as f:
        scores = json.load(f)
    
    top_ideas = [filename for filename in scores if scores[filename] == 5]
    print ("#top ideas: ", len(top_ideas))
    
    for filename in tqdm(top_ideas):
        with open("../cache_results_claude_may/experiment_plans_1k/{}/{}".format(args.cache_name, filename), "r") as f:
            idea = json.load(f)
            experiment_plan = idea["full_experiment_plan"]
        
        prompt, response, cost = novelty(experiment_plan, client, args.engine, args.seed)
        print (prompt)
        print (response)
        print ("Cost: ", cost)
        print ("-"*50)
        print ("\n\n")