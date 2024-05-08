from openai import OpenAI
import anthropic
from utils import call_api, shuffle_dict_and_convert_to_string
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing
from utils import cache_output
import random 
import retry

@retry.retry(tries=3, delay=2)
def style_transfer(model_idea, human_idea, openai_client, model, seed):
    prompt = "You are a writing assistant specialized in editing academic writing.\n"
    prompt += "I will give you a human-written research idea and a machine-written research idea. Your task is to edit the human-written idea to make the writing very similar to the style of the machine-written idea.\n"
    prompt += "Machine-written idea:\n" + model_idea + "\n\n"
    prompt += "Human-written idea:\n" + human_idea + "\n\n"
    prompt += "Make sure that you only edit the writing style, including things like punctuation, capitalization, linebreaks, and bullet points. Also make sure to edit the wording and phrasing to use vocabulary that sounds like machine-written.\n"
    prompt += "Apart from the writing, do not change the content of the idea. You must preserve the exact meaning of the original idea, do not change, remove or add any technical details.\n"
    prompt += "Now directly generate the edited human-written idea to match the style of the machine-written idea, in the exact same format as the machine-written idea:\n"
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4096, seed=seed, json_output=False)
    return prompt, response, cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-opus-20240229', help='api engine; https://openai.com/api/')
    parser.add_argument('--ideas_n', type=int, default=5, help="how many ideas to generate")
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
    
    with open("prompts/machine_idea.txt", "r") as f:
        machine_ideas = f.read() 
    with open("prompts/human_idea.txt", "r") as f:
        human_ideas = f.read()

    prompt, response, cost = style_transfer(model_idea, human_idea, openai_client, args.engine, args.seed)
    print (response)

