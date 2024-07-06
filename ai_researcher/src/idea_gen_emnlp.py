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
def idea_generation(existing_ideas, examples, ideas_n, topic_description, openai_client, model, seed):
    prompt = "You are an expert researcher in Natural Language Processing. Now I want you to help me brainstorm some new research project ideas on the topic of: " + topic_description + ".\n\n"
    prompt += "You should generate {} different ideas on this topic. Try to be creative and diverse in the idea generation, and do not repeat any similar ideas. ".format(str(ideas_n))
    prompt += "We are targeting the EMNLP 2024 conference (The 2024 Conference on Empirical Methods in Natural Language Processing), and you should aim for timely and impactful new ideas that can potentially win best paper awards at EMNLP.\n"
    prompt += "EMNLP 2024 invites the submission of papers featuring substantial, original, and unpublished research on empirical methods for Natural Language Processing. The type of contribution can include: formulate new problems, propose new methods that outperform existing baselines, construct new datasets or benchmarks, propose new evaluation metrics, propose novel applications of NLP, conduct empirical analysis, or any other novel contribution that advances the field of NLP. Note that we do not take survey or position papers - there has to be some computational experiments involved.\n"
    prompt += "Each idea should be described as: (1) Problem: State the problem statement, which should be closely related to the given topic and within the scope of NLP research. (2) Existing Work: Mention the most relevant existing work. (3) Motivation: Explain the inspiration of the proposed study and why it would work well or be important to study. (4) Proposed Study: Propose your new method or analysis or benchmark and describe it in detail. The proposal should be maximally different from all existing work and baselines, and be more advanced and effective than the baselines. You should be as creative as possible, we love unhinged ideas that sound crazy. This should be the most detailed section of the proposal. (5) Experiment Plan: Specify the hypotheses, experiment steps, baselines, evaluation metrics, and/or any other relevant details.\n"
    prompt += "You can follow these examples to get a sense of how the ideas should be formatted (but don't borrow the ideas themselves):\n" + examples + "\n"
    prompt += "You should make sure to come up with your own novel and different ideas for the specified problem: " + topic_description + "\n"
    if "claude" in model:
        prompt += "You should make each idea standalone and not dependent on the other ideas.\n"
    if existing_ideas:
        prompt += "You should avoid repeating the following existing ideas and try to be different and diverse: " + existing_ideas + "\n"
    prompt += "Please write down your {} ideas (each idea should be described as one paragraph. Output the ideas in json format as a dictionary, where you should generate a short idea name (e.g., \"Non-Linear Story Understanding\", or \"Multi-Agent Negotiation\") as the key and the actual idea description as the value (following the above format). Do not repeat idea names or contents.".format(str(ideas_n))

    print (prompt)

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0.9, max_tokens=4096, seed=seed, json_output=True)
    return prompt, response, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-opus-20240229', help='api engine; https://openai.com/api/')
    parser.add_argument('--idea_cache', type=str, default=None, required=True, help='where to store the generated ideas')
    parser.add_argument('--topic_description', type=str, default=None, required=True, help='topic description')
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
    
    topic_description = args.topic_description
    ideas_file = args.idea_cache
    
    try:
        ## extract existing ideas
        existing_ideas = None
        if os.path.exists(ideas_file):
            with open(ideas_file, "r") as f:
                ideas_cache = json.load(f)
            if "ideas" in ideas_cache:
                existing_ideas = [key for idea in ideas_cache["ideas"] for key in idea.keys()]
                existing_ideas = list(set(existing_ideas))
                existing_ideas = "; ".join(existing_ideas)
        
        ## load few-shot examples
        with open("prompts/idea_examples_method.json", "r") as f:
            method_idea_examples = json.load(f)
            method_idea_examples = shuffle_dict_and_convert_to_string(method_idea_examples)
        
        print ("topic: ", topic_description)
        print ("existing ideas: ", existing_ideas)
        print ("\n")
        print ("generating {} ideas...".format(str(args.ideas_n)))
        
        prompt, response, cost = idea_generation(existing_ideas, method_idea_examples, args.ideas_n, topic_description, client, args.engine, args.seed)
        
        print ("idea generation cost: ", cost)

        response = json.loads(response.strip())
        ideas = {"topic_description": topic_description, "ideas": [response]}
        
        ## if the idea_cache already exists, directly add to the current list
        if os.path.exists(ideas_file):
            with open(ideas_file, "r") as f:
                ideas_cache = json.load(f)
            ideas_cache["ideas"].append(response)
            ideas = ideas_cache
        
        print ("#ideas generated so far: ", sum(len(d) for d in ideas["ideas"]))

        ## save the cache
        cache_dir = os.path.dirname(ideas_file)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_output(ideas, ideas_file)

    except: 
        print ("Error in idea generation...")
