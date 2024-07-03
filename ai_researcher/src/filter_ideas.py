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
from lit_review import collect_papers
from lit_review_tools import format_papers_for_printing
random.seed(2024)

@retry.retry(tries=3, delay=2)
def self_novelty_score(experiment_plan, openai_client, model, seed):
    prompt = "You are a professor specialized in Natural Language Processing and Large Language Models. You are given a project proposal and you need to decide whether it is novel enough.\n"
    prompt += "The project proposal is:\n\n" 
    prompt += format_plan_json(experiment_plan)
    prompt += "\nReturn yes if the project is significantly different from existing work (both classic ones and recent ones), otherwise return no. Give a short rationale and then change to a new line to return either yes or no and then end the response.\n"
    prompt += "In the rationale, reference the most similar works and explain how the proposed project is similar to or different from them. You should return no if the proposed project is only a minor modification or combination of the existing ideas.\n"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=3000, seed=seed, json_output=False)
    return prompt, response, cost


@retry.retry(tries=3, delay=2)
def consistency_score(experiment_plan, openai_client, model, seed):
    prompt = "You are a professor specialized in Natural Language Processing and Large Language Models. You are given a project proposal and you need to decide whether it is consistent in the methodology and experiment design.\n"
    prompt += "The project proposal is:\n\n" 
    prompt += format_plan_json(experiment_plan)
    prompt += "\nYou should return no if the proposed method is based on prompting and assumes black-box model access but proposes any step that involves white-box model access, such as: getting input embedding, training the model, applying any loss functions, constrastive learning, performing dropout on the model, extracting or modifying internal models activations, computing influence functions, and any other operations that require model weight or data access. The only exception is that finetuning a small open-source model is fine, but the proposal should explicitly mention that the model being trained is open.\n"
    prompt += "Only return yes if the proposed method is either: 1) based only on open-source models; or 2) based on prompting black-box models (GPT, Claude, Gemini, etc.) and all steps do not require white-box model or data access as explained above. Give a short explanation first and then change to a new line to return either yes or no and then end the response.\n"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=3000, seed=seed, json_output=False)
    return prompt, response, cost

@retry.retry(tries=3, delay=2)
def retrieve_novelty_score(experiment_plan, related_paper, openai_client, model, seed):
    ## use gpt4 to give novelty judgment wrt one individual paper 
    prompt = "You are a professor specialized in Natural Language Processing. You have a project proposal and want to decide whether it is novel or has been done before.\n\n"
    prompt += "The project proposal is:\n" + format_plan_json(experiment_plan) + ".\n\n"
    prompt += "We have found a related paper:\n" + format_papers_for_printing([related_paper], include_score=False) + "\n\n"
    prompt += "The project proposal and paper abstract are considered a match if both the research problem and the approach are the same. For example, if they are both trying to improve code generation accuracy and both propose to use retrieval augmentation. You should answer yes if the proposed project is exploring essentially the same idea as the given related paper, and answer no otherwise.\n"
    prompt += "You should first specify what is the proposed research problem and approach. If answering yes, your explanation should be the one-sentence summary of both the abstract and the proposal and their similarity (e.g., they are both about probing biases of language models via fictional characters). If answering no, give the short summaries of the abstract and proposal separately, then highlight their differences. Then end your response with a binary judgment, saying either \"Yes\" or \"No\". Change to a new line after your explanation and just say Yes or No with no punctuation in the end.\n"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=3000, seed=seed, json_output=False)
    return prompt, response, cost

@retry.retry(tries=3, delay=2)
def all_checks(topic_description, experiment_plan, client, model, seed):
    ## perform all the checks
    consistency_prompt, consistency_response, consistency_cost = consistency_score(experiment_plan, client, model, seed)
    print ("Performing Consistency Check")
    if consistency_response.lower().split()[-1].strip() != "yes":
        print ("Failed Consistency Check!")
        print (consistency_prompt)
        print (consistency_response)
        return False
    
    print ("Performing Self-Novelty Check")
    self_novelty_prompt, self_novelty_response, self_novelty_cost = self_novelty_score(experiment_plan, client, model, seed)
    if self_novelty_response.lower().split()[-1].strip() != "yes":
        print ("Failed Self-Novelty Check!")
        print (self_novelty_prompt)
        print (self_novelty_response)
        return False
    
    print ("Performing Retrieval-Novelty Check")
    paper_bank, total_cost, all_queries = collect_papers(topic_description, client, model, seed, grounding_k=10, max_papers=100, print_all=True, mode="idea", idea=experiment_plan)
    ## check through the top-10 papers
    for related_paper in paper_bank[ : 5]:
        retrieve_novelty_prompt, retrieve_novelty_response, retrieve_novelty_cost = retrieve_novelty_score(experiment_plan, related_paper, client, model, seed)
        if retrieve_novelty_response.lower().split()[-1].strip() != "no":
            print ("Failed Related Paper Check!")
            print (retrieve_novelty_prompt)
            print (retrieve_novelty_response)
            return False

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-5-sonnet-20240620', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default="uncertainty_prompting_method_prompting", help='cache file name for the retrieved papers')
    parser.add_argument('--score_file', type=str, default="uncertainty_score_predictions_swiss_round_5", help='score file for reranking ideas')
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
    
    with open("logs/{}.json".format(args.score_file), "r") as f:
        scores = json.load(f)
    
    top_ideas = sorted(scores, key=scores.get, reverse=True)
    print ("#top ideas: ", len(top_ideas))
    
    passed_ideas = []
    for filename in tqdm(top_ideas):
        with open("../{}/{}".format(args.cache_name, filename), "r") as f:
            idea = json.load(f)
            experiment_plan = idea["full_experiment_plan"]
            topic_description = idea["topic_description"]
        
        check = all_checks(topic_description, experiment_plan, client, args.engine, args.seed)
        if check:
            print ("Idea Passed: ", filename)
            passed_ideas.append(experiment_plan)
        print ("#passed ideas: ", len(passed_ideas))
        if len(passed_ideas) >= 10:
            break
    
    print ("All Passed Ideas: ")
    for idea in passed_ideas:
        print (format_plan_json(idea))
        print ("-"*50 + "\n")