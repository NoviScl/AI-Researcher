from openai import OpenAI
from utils import call_api
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing
from utils import cache_output
import random 
random.seed(2024)

def idea_generation(paper_bank, grounding_k, topic_description, openai_client, model):
    ## retrieve top papers 
    grounding_papers = paper_bank[ : grounding_k]
    random.shuffle(grounding_papers)

    prompt = "You are an expert researcher in Natural Language Processing. Now I want you to help me brainstorm some new research project ideas on the topic of: " + topic_description + ".\n\n"
    prompt += "Here are some relevant papers that I have found for you:\n" + format_papers_for_printing(grounding_papers) + "\n"
    prompt += "You should generate three ideas that are within the same scope, but should be novel and different from the papers above. You can use the papers above as inspiration, but you should not copy them directly. You should also make sure that your ideas are not too broad or complicated. Try to include all the necessary methodology details and experiment setups.\n"
    prompt += "Please write down your three ideas (each idea should be described as one paragraph; index the ideas and separate them with new lines):"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0.7, max_tokens=2400, json_output=False)
    return prompt, response, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--papers', type=str, default="paper_bank_math_reasoning_max60.json", help='the json file containing retrieved papers')
    parser.add_argument('--grounding_k', type=int, default=10, help='how many papers to use for grounding')
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

    with open(os.path.join("cache_results", args.papers), "r") as f:
        paper_bank = json.load(f)
    
    topic_description = "better prompting strategies for large language models to improve mathematical problem solving abilities"
    prompt, response, cost = idea_generation(paper_bank, args.grounding_k, topic_description, openai_client, MODEL)

    print (response)
    print ("Total cost: ", cost)

    cache_output(response, "ideas.txt")
