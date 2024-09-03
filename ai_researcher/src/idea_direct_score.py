from openai import OpenAI
import anthropic
from utils import call_api
import argparse
import json
import os
from utils import cache_output
from tqdm import tqdm
import random 
random.seed(2024)

def overall_score(idea_proposal, openai_client, model):
    prompt = "You are a professor in Natural Language Processing. Your task is to review a project proposal.\n\n"

    prompt += "The project proposal is:\n" + idea_proposal + "\n\n"
    prompt += "You should give an overall score for the idea on a scale of 1 - 10 as defined below (Major AI conferences in the descriptions below refer to top-tier NLP/AI conferences such as *ACL, COLM, NeurIPS, ICLR, and ICML.):\n"
    prompt += "1 (Critically flawed, trivial, or wrong, would be a waste of students’ time to work on it)\n"
    prompt += "2 (Strong rejection for major AI conferences)\n"
    prompt += "3 (Clear rejection for major AI conferences)\n"
    prompt += "4 (Ok but not good enough, rejection for major AI conferences)\n"
    prompt += "5 (Decent idea but has some weaknesses or not exciting enough, marginally below the acceptance threshold of major AI conferences)\n"
    prompt += "6 (Marginally above the acceptance threshold of major AI conferences)\n"
    prompt += "7 (Good idea, would be accepted by major AI conferences)\n"
    prompt += "8 (Top 50% of all published ideas on this topic at major AI conferences, clear accept)\n"
    prompt += "9 (Top 15% of all published ideas on this topic at major AI conferences, strong accept)\n"
    prompt += "10 (Top 5% of all published ideas on this topic at major AI conferences, will be a seminal paper)\n"
    prompt += "Please directly provide a score between 1 and 10 for the project proposal (just a number, nothing else).\n"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=2, json_output=False)
    return prompt, response, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-5-sonnet-20240620', help='api engine; https://openai.com/api/')
    args = parser.parse_args()

    with open("../keys.json", "r") as f:
        keys = json.load(f)

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
    
    overall_scores = {}
    for filename in tqdm(os.listdir("../all_ideas/all_ideas")):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join("../all_ideas/all_ideas", filename), "r") as f:
            proposal = f.read()
        prompt, response, cost = overall_score(proposal, client, args.engine)
        overall_scores[filename] = int(response.strip())
        print (filename, response, cost)

    with open("../all_ideas/overall_scores_claude_direct.json", "w") as f:
        json.dump(overall_scores, f, indent=4)