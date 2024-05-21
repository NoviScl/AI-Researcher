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
    prompt += "I will give you a human-written research idea and a machine-written research idea. Your task is to edit the human-written idea to make the writing style very similar to the style of the machine-written idea.\n"
    prompt += "Machine-written idea:\n" + model_idea + "\n\n"
    prompt += "Human-written idea:\n" + human_idea + "\n\n"
    prompt += "Make sure that you only edit the writing style, including things like punctuation, capitalization, linebreaks, and bullet points. Also make sure to edit the wording and phrasing to use vocabulary that sounds like machine-written.\n"
    prompt += "The main sections should be indexed clearly without indentation at the beginning. These sections include title, problem statement, motivation, proposed method, step-by-step experiment plan, test case examples, and fallback plan. Each section can then have sub-bullets for sub-sections if applicable.\n"
    prompt += "You should use tab as indentation, and make sure to use appropriate nested indentation for sub-bullets. All bullets should have a clear hierarchy so people can easily differetiate the sub-bullets. Only leave empty lines between sections and remove any extra linebreaks. If many bullets points are clustered together in a paragraph, separate them clearly with indentation and appropriate bullet point markers. Change to a new line for each new bullet point.\n"
    prompt += "For the fallback plan, do not list a bunch of bullet points. Instead, condense them into one coherent paragraph. Apart from these, you also make necessary formatting changes to make the whole idea easier to read.\n"
    prompt += "For linebreaks, avoid Raw String Literals or Double Backslashes when using \"\\n\", change them to spaces or tabs.\n"
    prompt += "Apart from the writing and formatting, do not change the content of the idea. You must preserve the exact meaning of the original idea, do not change, remove or add any technical details.\n"
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
        machine_idea = f.read() 
    with open("prompts/human_idea.txt", "r") as f:
        human_idea = f.read()

    prompt, response, cost = style_transfer(machine_idea, human_idea, client, args.engine, args.seed)
    print (response)

