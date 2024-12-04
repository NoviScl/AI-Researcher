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

    # print (prompt)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=temperature, max_tokens=3000, seed=seed, json_output=False)
    return prompt, response, cost


def tournament_ranking(idea_lst, filename_lst, openai_client, model, seed, cache_name, ranking_score_dir, max_round=5, format="json"):
    # Initialize scores for each idea using the first 200 characters as keys
    # initial score is 1
    scores = defaultdict(lambda: 1)
    all_costs = 0
    # decision_correct = 0
    # decision_all = 0
    
    # Helper function to conduct a single round of the tournament
    def single_round(ideas, current_round=0, decision_correct=0, decision_all=0, all_costs=0):
        ## shuffle ideas in the first round
        if current_round == 0:
            random.shuffle(ideas)
        
        match_pairs = []
        # Sort ideas based on current scores
        sorted_ideas = sorted(ideas, key=lambda idea: scores[format_plan_json(idea)[:200]], reverse=True)

        for i in range(0, len(sorted_ideas), 2):
            if i + 1 < len(sorted_ideas):
                match_pairs.append((sorted_ideas[i], sorted_ideas[i+1]))
            else:
                # If there is an odd number of ideas, the last one automatically wins this round
                scores[format_plan_json(sorted_ideas[i])[:200]] += 1

        for idea1, idea2 in tqdm(match_pairs):
            prompt, result, cost = better_idea(idea1, idea2, "zero_shot", openai_client, model, seed)
            if result.strip() == '1':
                scores[format_plan_json(idea1)[:200]] += 1
                # if idea1["score"] >= idea2["score"]:
                #     decision_correct += 1
            else:
                scores[format_plan_json(idea2)[:200]] += 1
                # if idea1["score"] <= idea2["score"]:
                #     decision_correct += 1

            # decision_all += 1
            all_costs += cost
        
        return all_costs
    
    # Conduct the tournament rounds until only one idea remains
    current_round = 0
    score_predictions = {}
    while current_round < max_round:
        print ("Current round: ", current_round + 1)
        all_costs = single_round(idea_lst[:], current_round=current_round, all_costs=all_costs)
        # print ("Currect decision accuracy: {} / {} = {}".format(decision_correct, decision_all, decision_correct / decision_all))
        current_round += 1

        # Convert scores to a list matching the order of the original idea list
        final_scores = [scores[format_plan_json(idea)[:200]] for idea in idea_lst]

        for i in range(len(filename_lst)):
            score_predictions[filename_lst[i]] = final_scores[i]
        
        # Save all scores
        cache_file = os.path.join(ranking_score_dir, "{}/round_{}.json".format(cache_name, current_round))
        if not os.path.exists(os.path.dirname(cache_file)):
            os.makedirs(os.path.dirname(cache_file))
        with open(cache_file, "w") as f:
            json.dump(score_predictions, f, indent=4)
            
        # Save top 10 ideas with full content
        top_ideas = {}
        sorted_ideas_with_idx = sorted(enumerate(zip(filename_lst, final_scores)), key=lambda x: x[1][1], reverse=True)[:10]
        for idx, (idea_name, score) in sorted_ideas_with_idx:
            top_ideas[idea_name] = {
                "idea": idea_lst[idx],
                "ai_ranking_score": score,
                "human_novelty_score": 0,
                "human_excitement_score": 0,
                "human_feasibility_score": 0,
                "human_effectiveness_score": 0,
                "human_overall_score": 0
            }
            
        top_ideas_file = os.path.join(ranking_score_dir, "{}/top_ideas.json".format(cache_name))
        with open(top_ideas_file, "w") as f:
            json.dump(top_ideas, f, indent=4)
    
    return final_scores, all_costs




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--experiment_plan_cache_dir', type=str, default="openreview_benchmark", help='cache file name for the experiment plans')
    parser.add_argument('--rank_seed_ideas', action='store_true', help='whether to rank seed ideas (as opposed to full experiment plans)')
    parser.add_argument('--cache_name', type=str, default="openreview_benchmark", help='name of the specific cache dir')
    parser.add_argument('--ranking_score_dir', type=str, default="ranking_score_dir", help='dir to store the ranking scores')
    parser.add_argument('--max_round', type=int, default=5, help="seed for GPT-4 generation")
    parser.add_argument('--format', type=str, default="json", help='format of the input file, either json or txt')
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


    if args.rank_seed_ideas:
        idea_lst = []
        filename_lst = []
        filename = os.path.join(args.experiment_plan_cache_dir, args.cache_name + ".json")
        with open(filename, "r") as f:
            data = json.load(f)
        for k,v in data["ideas"].items():
            idea_lst.append(v)
            filename_lst.append(k)
    else:
        filenames = os.listdir(os.path.join(args.experiment_plan_cache_dir, args.cache_name))
        if args.format == "json":
            filenames = [f for f in filenames if f.endswith(".json")]
        else:
            filenames = [f for f in filenames if f.endswith(".txt")]

        score_predictions = {}
        filename_lst = []
        idea_lst = []

        for filename in filenames:
            if args.format == "json":
                with open(os.path.join(args.experiment_plan_cache_dir, args.cache_name, filename), "r") as f:
                    paper = json.load(f)
                if "full_experiment_plan" in paper and isinstance(paper["full_experiment_plan"], dict):
                    summary = paper["full_experiment_plan"]
                    idea_lst.append(summary)
                    filename_lst.append(filename)
            else:
                with open(os.path.join(args.experiment_plan_cache_dir, args.cache_name, filename), "r") as f:
                    summary = f.read()
                idea_lst.append(summary)
                filename_lst.append(filename)

    print ("total #ideas: ", len(idea_lst))
    final_scores, all_costs = tournament_ranking(idea_lst, filename_lst, client, args.engine, args.seed, args.cache_name, args.ranking_score_dir, args.max_round, format=args.format)
    print ("all costs: ", all_costs)

