from openai import OpenAI
from utils import call_api
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing
from utils import cache_output
import random 
random.seed(2024)

def idea_ranking(idea_proposal, topic_description, openai_client, model):
    prompt = "You are a professor in Natural Language Processing. You need to evaluate project proposals on the topic of: " + topic_description + ".\n\n"

    prompt += "The project proposal is:\n" + idea_proposal + "\n\n"
    prompt += "Please evaluate the proposal along the following four dimensions:\n"
    prompt += "1. Novelty: How novel is the idea? Is that creative and different from prior works on this topic?\n"
    prompt += "2. Feasibility: How feasible is the idea? Can it be executed successfully within two weeks by a PhD student? Does it require a lot of compute or human resources? Note that we do provide OpenAI and GPU credits so running inference with GPT series models or relatively small open-source models should be feasible. On the other hand, running large-scale training and human experiments could be expensive.\n"
    prompt += "3. Excitement: How exciting is the proposed project? Could it lead to an exciting and impactful project?\n"
    prompt += "4. Relevance: Is the project proposal relevant to the given topic?\n\n"
    prompt += "For each dimension, first give a rationale for the judgment, and then give a numeric score from 1 - 10 (with the format: \"Score: x\"). Index each dimension and separate them with new lines.\n"
    prompt += "In the end, give an overall assessment on the scale of 1 - 10 (integer only) in the end with the format \"Final Score: x\" Try to be critical and only give high scores to ideas that can get accepted into top venues."
    prompt += "The reference scale is:\n"
    prompt += "1: strong reject\n"
    prompt += "3: reject, not good enough\n"
    prompt += "5: marginally below the acceptance threshold\n"
    prompt += "6: marginally above the acceptance threshold\n"
    prompt += "8: accept, good paper\n"
    prompt += "10: strong accept, should be highlighted at the conference\n\n"
    prompt += "Now please evaluate the proposal."


    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=2400, json_output=False)
    return prompt, response, cost

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

    # topic_description = "better prompting strategies for large language models to improve mathematical problem solving abilities"
    # prompt, response, cost = idea_ranking(idea_proposal, topic_description, openai_client, MODEL)
    
    topic_description = "better prompting strategies for large language models to improve multi-step problem solving abilities"
    prompt, response, cost = idea_ranking(idea_proposal, topic_description, openai_client, MODEL)

    print (response)
    print ("Total cost: ", cost)

    cache_output(response, "eval_" + args.idea_file)
