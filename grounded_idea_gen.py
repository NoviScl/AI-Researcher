from openai import OpenAI
from utils import call_api
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing
from utils import cache_output
import random 
from lit_review import collect_papers

def idea_generation(paper_bank, grounding_k, ideas_n, topic_description, openai_client, model, seed):
    ## retrieve top papers 
    grounding_papers = paper_bank[ : grounding_k]
    random.shuffle(grounding_papers)

    prompt = "You are an expert researcher in Natural Language Processing. Now I want you to help me brainstorm some new research project ideas on the topic of: " + topic_description + ".\n\n"
    prompt += "Here are some relevant papers that I have found for you:\n" + format_papers_for_printing(grounding_papers) + "\n"
    prompt += "You should generate {} ideas that are within the same scope, but should be novel and different from the papers above. Try to be creative and diverse in the idea generation, and do not repeat any similar ideas multiple times. You can use the papers above as inspiration, but you should not copy them. You should also make sure that your ideas are not too broad or complicated. Include all the necessary methodology details and experiment setups.\n".format(str(ideas_n))
    # prompt += "Before generating the actual idea description, first specify the intended type of contribution, which should be either: 1) analysis; or 2) method. Analysis papers should propose a novel and interesting behavior worth studying, and then describe the methods to gather relevant data and the experiment steps for conducting the evaluation. Method papers should propose a novel method to improve model performance on a speific tasks, as measured on corresponding benchmarks. The method should be described in detail, and the experiment steps should be clearly specified.\n"
    # prompt += "You do not necessarily need to balance the number of analysis and method paper ideas, if certain topics are more suitable for analysis projects, feel free to prioritize those, and vice versa.\n"
    prompt += "To make the projects as feasible as possible (preferably something that can be executed by a student within two weeks), please avoid projects involveing large-scale model training, or human studies; insead, please prefer ideas that mostly only involve prompting.\n"
    prompt += "Please write down your ideas (each idea should be described as one paragraph. Output the ideas in json format as a dictionary, where you should generate a short idea name (e.g., \"Non-Linear Story Understanding\", or \"Multi-Agent Negotiation\") as the key and the actual idea description as the value."

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0.9, max_tokens=4000, seed=seed, json_output=True)
    return prompt, response, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default=None, required=True, help='cache file name for the retrieved papers')
    parser.add_argument('--grounding_k', type=int, default=10, help='how many papers to use for grounding')
    parser.add_argument('--ideas_n', type=int, default=5, help="how many ideas to generate")
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()

    with open("keys.json", "r") as f:
        keys = json.load(f)
    random.seed(args.seed)

    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    openai_client = OpenAI(
        organization=ORG_ID,
        api_key=OAI_KEY
    )

    with open(os.path.join("cache_results/lit_review", args.cache_name+".json"), "r") as f:
        lit_review = json.load(f)
    topic_description = lit_review["topic_description"]
    paper_bank = lit_review["paper_bank"]

    print ("topic: ", topic_description)
    print ("\n")
    print ("generating {} ideas...".format(str(args.ideas_n)))
    prompt, response, cost = idea_generation(paper_bank, args.grounding_k, args.ideas_n, topic_description, openai_client, args.engine, args.seed)
    print ("ideas: ", response)
    print ("idea generation cost: ", cost)

    ## cache the generated ideas
    response = json.loads(response.strip())
    ideas = {"topic_description": topic_description, "ideas": response}
    ideas_file = os.path.join("cache_results/ideas", args.cache_name+".json")
    
    ## if the idea_cache already exists, directly add to the current list
    if os.path.exists(ideas_file):
        with open(ideas_file, "r") as f:
            ideas_cache = json.load(f)
        ideas_cache["ideas"].update(response)
        ideas = ideas_cache

    ## save the cache
    if not os.path.exists("cache_results/ideas"):
        os.makedirs("cache_results/ideas")
    cache_output(ideas, ideas_file)

