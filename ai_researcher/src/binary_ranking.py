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
def better_idea(idea_1, idea_2, method, openai_client, model, seed, few_shot_demos=None, temperature=0.):
    prompt = "You are a reviewer specialized in Natural Language Processing and Large Language Models. You are given two project summaries. One of them is accepted by a top AI conference (like ICLR or ACL) and the other one is rejected. Your task is to identify the one that has been accepted.\n"
    
    ## zero-shot methods
    if "zero_shot" in method:
        prompt += "The two project proposals are:\n\n" 
        prompt += "paper 1:\n" + format_plan_json(idea_1) + "\n\n"
        prompt += "paper 2:\n" + format_plan_json(idea_2) + "\n\n"
        # prompt += "\nYou can consider factors like novelty, soundness, excitement, and potential impact.\n"
    
        if method == "zero_shot":
            prompt += "Now decide which one is the accepted idea. Directly return a number 1 or 2 and nothing else.\n"
        elif method == "zero_shot_cot":
            prompt += "Now decide which one is the accepted idea. Think step by step by writing a meta-review to compare the strengths and weaknesses of both ideas and explain why one idea is better than the other. After the meta-review, start a new line and directly return a number 1 or 2 to indicate the accepted idea and end the response.\n"
    
    ## few-shot methods
    elif "few_shot" in method:
        prompt += "Here are some examples:\n" + few_shot_demos
        prompt += "\n\nThe two project summaries given to you are:\n\n" 
        prompt += "paper 1:\n" + format_plan_json(idea_1) + "\n\n"
        prompt += "paper 2:\n" + format_plan_json(idea_2) + "\n\n"
        # prompt += "\nYou should consider factors like novelty, soundness, excitement,and potential impact.\n"
        
        if method == "few_shot":
            prompt += "Now decide which one is the accepted idea. Follow the above examples: return a number 1 or 2 and nothing else.\n"
        elif method == "few_shot_cot":
            prompt += "Now decide which one is the accepted idea. Follow the above examples: give a meta-review and score to each paper, and then start a new line and directly return a number 1 or 2 to indicate the accepted idea and end the response.\n"

    print (prompt)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=temperature, max_tokens=3000, seed=seed, json_output=False)
    return prompt, response, cost


def self_consistency(idea_1, idea_2, method, openai_client, model, seed, sc_n=10, few_shot_demos=None):
    predictions = []
    costs = 0
    for i in range(sc_n):
        new_method = method.replace("_sc", "")
        prompt, response, cost = better_idea(idea_1, idea_2, new_method, openai_client, model, seed, few_shot_demos, temperature=0.7)
        predictions.append(response.strip().split()[-1])
        costs += cost
    
    print ("all predictions: ", predictions)

    ## take majority vote
    response = max(set(predictions), key=predictions.count)
    
    return prompt, response, costs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-opus-20240229', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default="ORB_full", help='cache file name for the retrieved papers')
    parser.add_argument('--method', type=str, default="few_shot_cot", help='methods: {zero_shot, zero_shot_cot, few_shot, few_shot_cot} x {normal, sc}')
    parser.add_argument('--sc_n', type=int, default=10, help="number of sampling for self-consistency")
    parser.add_argument('--seed', type=int, default=2024, help="seed")
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

    with open("../{}/pos_papers.json".format(args.cache_name), "r") as f:
        pos_papers = json.load(f)
    with open("../{}/neg_papers.json".format(args.cache_name), "r") as f:
        neg_papers = json.load(f)
    
    if "cot" in args.method:
        with open("prompts/binary_ranking_cot_examples.txt", "r") as f:
            few_shot_demos = f.read()
    else:
        with open("prompts/binary_ranking_examples.txt", "r") as f:
            few_shot_demos = f.read()

    n_pos = len(pos_papers)
    n_neg = len(neg_papers)
    N = min(n_pos, n_neg)
    print ("#pos: ", n_pos, "#neg: ", n_neg, "N: ", N)
    correct = 0

    for i in tqdm(range(N)):
        pos_idea = pos_papers[i]
        neg_idea = neg_papers[i]
        ## randomly sample 1 or 2
        label = random.randint(1, 2)

        if "_sc" not in args.method:
            if label == 1:
                prompt, response, cost = better_idea(pos_idea["structured_summary"], neg_idea["structured_summary"], args.method, client, args.engine, args.seed, few_shot_demos)
            else:
                prompt, response, cost = better_idea(neg_idea["structured_summary"], pos_idea["structured_summary"], args.method, client, args.engine, args.seed, few_shot_demos)
        else:
            if label == 1:
                prompt, response, cost = self_consistency(pos_idea["structured_summary"], neg_idea["structured_summary"], args.method, client, args.engine, args.seed, args.sc_n, few_shot_demos)
            else:
                prompt, response, cost = self_consistency(neg_idea["structured_summary"], pos_idea["structured_summary"], args.method, client, args.engine, args.seed, args.sc_n, few_shot_demos)

        print ("full response: ", response)
        print ("predicted: ", response.strip().split()[-1])
        print ("label: ", label)
        print ("cost: ", cost)

        if response.strip().split()[-1] == str(label):
            correct += 1
    
    print ("Accuracy: {} / {} = {}%".format(correct, N, correct / N * 100))
