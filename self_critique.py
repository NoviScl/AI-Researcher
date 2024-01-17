from openai import OpenAI
from utils import call_api
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing
from utils import cache_output
import random 
random.seed(2024)

def critique(idea_proposal, topic_description, openai_client, model):
    prompt = "You are a reviewer with expertise in Natural Language Processing. You need to criticize project proposals on the topic of: " + topic_description + ".\n\n"

    prompt += "The project proposal is:\n" + idea_proposal + "\n\n"
    prompt += "Please come up with a list questions and critical comments targeting the weaknesses of proposal.\n"
    prompt += "You should focus solely on missing details in the proposal, mainly:\n"
    prompt += "1. did the proposal list the datasets to be used?\n"
    prompt += "2. did the proposal describe every single step of the proposed method or experiment?\n"
    prompt += "3. did the proposal describe the evaluation metrics?\n"
    prompt += "You do not have to worry about subjective matters like broader impact of ethical concerns. Your questions and critic:"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=2400, json_output=False)
    return prompt, response, cost

def more_lit_review():
    return 

def improve_idea():
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--idea_file', type=str, default="human_idea_0.txt", help='a txt file containing the idea proposal')
    args = parser.parse_args()

    with open("keys.json", "r") as f:
        keys = json.load(f)

    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    S2_KEY = keys["s2_key"]
    MODEL = args.engine
    openai_client = OpenAI(
        organization=ORG_ID,
        api_key=OAI_KEY
    )
    
    with open(os.path.join("cache_results", args.idea_file), "r") as f:
        idea_proposal = f.read()

    topic_description = "better prompting strategies for large language models to improve mathematical problem solving abilities"
    prompt, response, cost = critique(idea_proposal, topic_description, openai_client, MODEL)
    
    # topic_description = "better prompting strategies for large language models to improve multi-step problem solving abilities"
    # prompt, response, cost = critique(idea_proposal, topic_description, openai_client, MODEL)

    print (response)
    print ("Total cost: ", cost)

    cache_output(response, "critique_" + args.idea_file)
