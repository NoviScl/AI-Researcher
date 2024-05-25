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
def idea_generation_method(method, existing_ideas, paper_bank, grounding_k, examples, ideas_n, topic_description, openai_client, model, seed):
    ## retrieve top papers (with some randomization)
    top_papers = paper_bank[ : int(grounding_k * 2)]
    random.shuffle(top_papers)
    grounding_papers = top_papers[ : grounding_k]

    prompt = "You are an expert researcher in Large Language Models. Now I want you to help me brainstorm some new research project ideas on the topic of: " + topic_description + ".\n\n"
    prompt += "Here are some relevant papers on this topic just for your background knowledge:\n" + format_papers_for_printing(grounding_papers, include_score=False, include_id=False) + "\n"
    prompt += "You should generate {} different ideas on this topic. Try to be creative and diverse in the idea generation, and do not repeat any similar ideas. The above papers are only for inspiration and you should not cite them and just make some incremental modifications. Instead, you should make sure your ideas are novel and distinct from the prior literature. You should aim for projects that can potentially win best paper awards at top conferences like ACL and NeurIPS.\n".format(str(ideas_n))
    prompt += "Each idea should be descibed as: (1) Problem: State the problem statement, which should be closely related to the topic description and something that large language models cannot solve well yet. (2) Existing Methods: Mention some existing benchmarks and baseline methods if there are any. (3) Motivation: Explain the inspiration of the proposed method and why it would work well. (4) Proposed Method: Propose your new method and describe it in detail. The proposed method should be maximally different from all existing work and baselines, and be more advanced and effective than the baselines. You should be as creative as possible in proposing new methods, we love unhinged ideas that sound crazy. This should be the most detailed section of the proposal. (5) Experiment Plan: Specify the experiment steps, baselines, and evaluation metrics.\n"
    prompt += "You can follow these examples to get a sense of how the ideas should be formatted (but don't borrow the ideas themselves):\n" + examples + "\n"
    prompt += "You should make sure to come up with your own novel and different ideas for the specified problem: " + topic_description + ". You should try to tackle important problems that are well recognized in the field and considered challenging for current models. For example, think of novel solutions for problems with existing benchmarks and baselines. In rare cases, you can propose to tackle a new problem, but you will have to justify why it is important and how to set up proper evaluation.\n"
    if "claude" in model:
        prompt += "You should make each idea standalone and not dependent on the other ideas.\n"
    if method == "prompting":
        prompt += "Focus on novel prompting ideas for now. The proposed method section should specify how to construct the prompts for all steps involved. Try to avoid large-scale pretraining experiments or human studies.\n"
    elif method == "finetuning":
        prompt += "Focus on novel finetuning ideas for now. The proposed method section should specify how to get the finetuning data and what's the training objective.\n"
    if existing_ideas:
        prompt += "You should avoid repeating the following existing ideas and try to be different and diverse: " + existing_ideas + "\n"
    prompt += "Please write down your {} ideas (each idea should be described as one paragraph. Output the ideas in json format as a dictionary, where you should generate a short idea name (e.g., \"Non-Linear Story Understanding\", or \"Multi-Agent Negotiation\") as the key and the actual idea description as the value (following the above format). Do not repeat idea names or contents.".format(str(ideas_n))

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0.9, max_tokens=4096, seed=seed, json_output=True)
    return prompt, response, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-opus-20240229', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default=None, required=True, help='cache file name for the retrieved papers')
    parser.add_argument('--method', type=str, default='prompting', help='either prompting or finetuning')
    parser.add_argument('--grounding_k', type=int, default=10, help='how many papers to use for grounding')
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

    if "claude" in args.engine:
        with open(os.path.join("../cache_results_claude_may/lit_review", args.cache_name + ".json"), "r") as f:
            lit_review = json.load(f)
    else:
        with open(os.path.join("../cache_results_gpt4/lit_review", args.cache_name + ".json"), "r") as f:
            lit_review = json.load(f)
    topic_description = lit_review["topic_description"]
    paper_bank = lit_review["paper_bank"]

    ## cache dir and file
    if "claude" in args.engine:
        cache_dir = "../cache_results_claude_may/ideas_1k"
    else:
        cache_dir = "../cache_results_gpt4/ideas"
    ideas_file = os.path.join(cache_dir, args.cache_name + '_' + args.method + ".json")

    ## extract existing ideas
    existing_ideas = None
    if os.path.exists(ideas_file):
        with open(ideas_file, "r") as f:
            ideas_cache = json.load(f)
        if "ideas" in ideas_cache:
            existing_ideas = list(ideas_cache["ideas"].keys())
            existing_ideas = "; ".join(existing_ideas)
    
    if args.method == "prompting":
        with open("prompts/idea_examples_prompting_method.json", "r") as f:
            method_idea_examples = json.load(f)
            method_idea_examples = shuffle_dict_and_convert_to_string(method_idea_examples)
    elif args.method == "finetuning":
        with open("prompts/idea_examples_finetuning_method.json", "r") as f:
            method_idea_examples = json.load(f)
            method_idea_examples = shuffle_dict_and_convert_to_string(method_idea_examples)
    
    print ("topic: ", topic_description)
    # print ("method: ", args.method)
    print ("existing ideas: ", existing_ideas)
    print ("\n")
    print ("generating {} ideas...".format(str(args.ideas_n)))
    
    if "method" in args.cache_name:
        prompt, response, cost = idea_generation_method(args.method, existing_ideas, paper_bank, args.grounding_k, method_idea_examples, args.ideas_n, topic_description, client, args.engine, args.seed)
    
    # print ("prompt: ", prompt)
    # print ("ideas: ", response)
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
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_output(ideas, ideas_file)

