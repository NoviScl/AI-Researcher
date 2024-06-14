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
def overall_score(experiment_plan, criteria, openai_client, model, seed):
    prompt = "You are a professor specialized in Natural Language Processing and Large Language Models. You are given a project proposal and you need to score it.\n"
    prompt += "The project proposal is:\n\n" 
    prompt += format_plan_json(experiment_plan)
    prompt += "\n\nYour should follow the scoring rubrics:\n" + criteria + "\n"
    prompt += "Now directly provide me with a final score between 1 and 10, no other explanation needed.\n"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=2, seed=seed, json_output=False)
    return prompt, response, cost

def better_idea(idea_1, idea_2, openai_client, model, seed):
    prompt = "You are a reviewer specialized in Natural Language Processing and Large Language Models. You are given two project summaries. One of them is accepted by the top AI conference and the other one is rejected. Your task is to identify the one that has been accepted.\n"
    prompt += "The two project proposals are:\n\n" 
    prompt += "1.\n" + format_plan_json(idea_1) + "\n\n"
    prompt += "2.\n" + format_plan_json(idea_2) + "\n\n"
    prompt += "\nYou should consider factors like novelty, soundness, and potential impact.\n"
    # prompt += "Now decide which one is the accepted idea. Directly return a number 1 or 2 and nothing else.\n"
    # prompt += "Now decide which one is the accepted idea. First imagine you are a reviewer and give a review to both ideas analyzing their strengths and weaknesses. Then start a new line and directly return a number 1 or 2 to indicate the accpted idea and nothing else.\n"
    prompt += "Now decide which one is the accepted idea. Write a meta-review to explain why one idea is better than the other first. Then start a new line and directly return a number 1 or 2 to indicate the accpted idea to end the response."

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return prompt, response, cost


def tournament_ranking(idea_lst, openai_client, model, seed):
    # Initialize scores for each idea using the first 200 characters as keys
    scores = defaultdict(int)
    decision_correct = 0
    decision_all = 0
    
    # Helper function to conduct a single round of the tournament
    def single_round(ideas, decision_correct=0, decision_all=0):
        winners = []
        for i in tqdm(range(0, len(ideas), 2)):
            if i + 1 < len(ideas):
                prompt, result, cost = better_idea(ideas[i], ideas[i+1], openai_client, model, seed)
                # print (prompt, result, cost)
                if result.strip() == '1':
                    winners.append(ideas[i])
                    scores[format_plan_json(ideas[i])[:200]] += 1
                    if ideas[i]["score"] >= ideas[i+1]["score"]:
                        decision_correct += 1
                else:
                    winners.append(ideas[i+1])
                    scores[format_plan_json(ideas[i+1])[:200]] += 1

                    if ideas[i]["score"] <= ideas[i+1]["score"]:
                        decision_correct += 1

                decision_all += 1
            else:
                # If there is an odd number of ideas, the last one automatically wins this round
                winners.append(ideas[i])
                scores[format_plan_json(ideas[i])[:200]] += 1
        return winners, decision_correct, decision_all
    
    # Conduct the tournament rounds until only one idea remains
    current_round = idea_lst[:]
    while len(current_round) > 1:
        print ("Current round #ideas: ", len(current_round))
        current_round, decision_correct, decision_all = single_round(current_round, decision_correct, decision_all)
        print ("Currect decision accuracy: {} / {} = {}".format(decision_correct, decision_all, decision_correct / decision_all))
    
    # The final winner
    final_winner = current_round[0]
    
    # Convert scores to a list matching the order of the original idea list
    final_scores = [scores[format_plan_json(idea)[:200]] for idea in idea_lst]
    
    return final_winner, final_scores, decision_correct, decision_all



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-opus-20240229', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default="openreview_binary", help='cache file name for the retrieved papers')
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

    with open("../{}/pos_papers.json".format(args.cache_name), "r") as f:
        pos_papers = json.load(f)
    with open("../{}/neg_papers.json".format(args.cache_name), "r") as f:
        neg_papers = json.load(f)

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

        if label == 1:
            prompt, response, cost = better_idea(pos_idea["structured_summary"], neg_idea["structured_summary"], client, args.engine, args.seed)
        else:
            prompt, response, cost = better_idea(neg_idea["structured_summary"], pos_idea["structured_summary"], client, args.engine, args.seed)
        
        # print (prompt)
        print ("predicted: ", response)
        print ("label: ", label)
        # print (cost)

        if response.strip().split()[-1] == str(label):
            correct += 1
    
    print ("Accuracy: {} / {} = {}%".format(correct, N, correct / N * 100))
