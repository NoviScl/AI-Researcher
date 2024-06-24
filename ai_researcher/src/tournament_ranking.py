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
    prompt = "You are a reviewer specialized in Natural Language Processing and Large Language Models. You are given two project proposals but you can only choose one of them to execute.\n"
    prompt += "The two project proposals are:\n\n" 
    prompt += "1.\n" + format_plan_json(idea_1) + "\n\n"
    prompt += "2.\n" + format_plan_json(idea_2) + "\n\n"
    prompt += "\nYou should consider multiple factors including novelty, technical soundsness, experiment design, potential impact, feasibility to be executed, and likelihood that this idea would work well empirically."
    prompt += "Now decide which one is the better project that you will work on. Directly return a number 1 or 2 and nothing else.\n"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=2, seed=seed, json_output=False)
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

    # with open("prompts/review_rubrics.txt", "r") as f:
    #     criteria = f.read().strip()

    filenames = os.listdir("../{}".format(args.cache_name))
    filenames = [f for f in filenames if f.endswith(".json") and '5' in f]

    score_predictions = {}
    filename_lst = []
    idea_lst = []

    for filename in filenames:
        with open("../{}/{}".format(args.cache_name, filename), "r") as f:
            paper = json.load(f)
        if "structured_summary" in paper and isinstance(paper["structured_summary"], dict) and "scores" in paper:
            summary = paper["structured_summary"]
            summary["score"] = avg_score(paper["scores"])
            idea_lst.append(summary)
            filename_lst.append(filename)

    idea_lst = idea_lst[:50] + idea_lst[-50:]
    filename_lst = filename_lst[:50] + filename_lst[-50:]
    print ("total #ideas: ", len(idea_lst))
    final_winner, final_scores, decision_correct, decision_all = tournament_ranking(idea_lst, client, args.engine, args.seed)
    print ("Final winner: ", final_winner)
    print ("Decision accuracy: {} / {} = {}".format(decision_correct, decision_all, decision_correct / decision_all))
    
    for i in range(len(filename_lst)):
        score_predictions[filename_lst[i]] = final_scores[i]
    
        # print (paper["structured_summary"])
        # print (format_plan_json(paper["structured_summary"]))

    #     try:
    #         experiment_plan = paper["structured_summary"]
    #         prompt, response, cost = overall_score(experiment_plan, criteria, client, args.engine, args.seed)
            
    #         # print (prompt)
    #         score_predictions[filename] = {"predicted_score": int(response.strip()), "actual_score": avg_score(paper["scores"])}
    #         print ("predicted score: ", response)
    #         print ("actual score: ", avg_score(paper["scores"]))
    #         print (cost)
        
    #     except: 
    #         continue 
    
    with open("score_predictions_tournament.json", "w") as f:
        json.dump(score_predictions, f, indent=4)
    
