from openai import OpenAI
from utils import call_api
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing
from utils import cache_output
import random 
import retry

@retry.retry(tries=3, delay=2)
def idea_generation_method(method, paper_bank, grounding_k, examples, ideas_n, topic_description, openai_client, model, seed):
    ## retrieve top papers (with some randomization)
    top_papers = paper_bank[ : int(grounding_k * 2)]
    random.shuffle(top_papers)
    grounding_papers = top_papers[ : grounding_k]

    prompt = "You are an expert researcher in Large Language Models. Now I want you to help me brainstorm some new research project ideas on the topic of: " + topic_description + ".\n\n"
    prompt += "Here are some relevant papers on this topic just for your background knowledge:\n" + format_papers_for_printing(grounding_papers, include_score=False, include_id=False) + "\n"
    prompt += "You should generate {} different ideas on this topic. Try to be creative and diverse in the idea generation, and do not repeat any similar ideas. The above papers are only for inspiration and you should not cite them and just make some incremental modifications. Instead, you should make sure your ideas are novel and distinct from the prior literature. You should aim for projects that can potentially win best paper awards at top conferences like ACL and NeurIPS.\n".format(str(ideas_n))
    prompt += "Each idea should be descibed as: (1) Problem: State the problem statement, which should be closely related to the topic description and something that large language models cannot solve well yet. (2) Existing Methods: Mention some existing benchmarks and baseline methods if there are any. (3) Motivation: Explain the inspiration of the proposed method and why it would work well. (4) Proposed Method: Propose your new method and describe it in detail. The proposed method should be maximally different from all existing work and baselines, and be more advanced and effective than the baselines. You should be as creative as possible in proposing new methods, we love unhinged ideas that sound crazy. This should be the most detailed section of the proposal. (5) Experiment Plan: Specify the experiment steps, baselines, and evaluation metrics.\n"
    prompt += "You can follow these examples to get a sense of how the ideas should be formatted (but don't borrow the ideas themselves):\n" + examples + "\n"
    prompt += "You should make sure to come up with your own novel and different ideas for the specified problem: " + topic_description + ".\n"
    if method == "prompting":
        prompt += "Focus on novel and more complex prompting ideas for now, and we will generate finetuning ideas later. The proposed method section should specify how to construct the prompts for all steps involved.\n"
    elif method == "finetuning":
        prompt += "Focus on novel and more complex finetuning ideas for now, and we will generate prompting ideas later. The proposed method section should specify how to get the finetuning data and what's the training objective.\n"
    prompt += "Please write down your {} ideas (each idea should be described as one paragraph. Output the ideas in json format as a dictionary, where you should generate a short idea name (e.g., \"Non-Linear Story Understanding\", or \"Multi-Agent Negotiation\") as the key and the actual idea description as the value (following the above format). Do not repeat idea names or contents.".format(str(ideas_n))

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0.9, max_tokens=4096, seed=seed, json_output=True)
    return prompt, response, cost

@retry.retry(tries=3, delay=2)
def idea_generation_analysis(paper_bank, grounding_k, ideas_n, topic_description, openai_client, model, seed):
    ## retrieve top papers (with some randomization)
    top_papers = paper_bank[ : int(grounding_k * 2)]
    random.shuffle(top_papers)
    grounding_papers = top_papers[ : grounding_k]

    prompt = "You are an expert researcher in Natural Language Processing. Now I want you to help me brainstorm some new research project ideas on the topic of: " + topic_description + ".\n\n"
    prompt += "Here are some relevant papers that I have found for you:\n" + format_papers_for_printing(grounding_papers) + "\n"
    prompt += "You should generate {} ideas on this topic, but should be novel and different from the papers above. Try to be creative and diverse in the idea generation, and do not repeat any similar ideas multiple times. You can use the papers above as inspiration, but you should not copy them. You should also make sure that your ideas are not too broad or complicated. Include all the necessary methodology details and experiment setups.\n".format(str(ideas_n))
    # prompt += "Before generating the actual idea description, first specify the intended type of contribution, which should be either: 1) analysis; or 2) method. Analysis papers should propose a novel and interesting behavior worth studying, and then describe the methods to gather relevant data and the experiment steps for conducting the evaluation. Method papers should propose a novel method to improve model performance on a speific tasks, as measured on corresponding benchmarks. The method should be described in detail, and the experiment steps should be clearly specified.\n"
    # prompt += "You do not necessarily need to balance the number of analysis and method paper ideas, if certain topics are more suitable for analysis projects, feel free to prioritize those, and vice versa.\n"
    prompt += "To make the projects as feasible as possible (preferably something that can be executed by a student within two weeks), please avoid projects involveing large-scale model training, or human studies; insead, please prefer ideas that mostly only involve prompting.\n"
    prompt += "Please write down your ideas (each idea should be described as one paragraph. Output the ideas in json format as a dictionary, where you should generate a short idea name (e.g., \"Non-Linear Story Understanding\", or \"Multi-Agent Negotiation\") as the key and the actual idea description as the value."

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0.9, max_tokens=4000, seed=seed, json_output=True)
    return prompt, response, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-0125-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default=None, required=True, help='cache file name for the retrieved papers')
    parser.add_argument('--method', type=str, default='prompting', help='either prompting or finetuning')
    parser.add_argument('--grounding_k', type=int, default=10, help='how many papers to use for grounding')
    parser.add_argument('--ideas_n', type=int, default=5, help="how many ideas to generate")
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()

    with open("../keys.json", "r") as f:
        keys = json.load(f)
    random.seed(args.seed)

    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    openai_client = OpenAI(
        organization=ORG_ID,
        api_key=OAI_KEY
    )

    with open(os.path.join("../cache_results/lit_review", args.cache_name + ".json"), "r") as f:
        lit_review = json.load(f)
    topic_description = lit_review["topic_description"]
    paper_bank = lit_review["paper_bank"]

    if args.method == "prompting":
        with open("prompts/idea_examples_prompting_method.txt", "r") as f:
            method_idea_examples = f.read().strip()
    elif args.method == "finetuning":
        with open("prompts/idea_examples_finetuning_method.txt", "r") as f:
            method_idea_examples = f.read().strip()

    # with open("prompts/idea_examples_method.txt", "r") as f:
    #     method_idea_examples = f.read().strip()
    
    print ("topic: ", topic_description)
    print ("method: ", args.method)
    print ("\n")
    print ("generating {} ideas...".format(str(args.ideas_n)))
    if "method" in args.cache_name:
        prompt, response, cost = idea_generation_method(args.method, paper_bank, args.grounding_k, method_idea_examples, args.ideas_n, topic_description, openai_client, args.engine, args.seed)
    elif "analysis" in args.cache_name:
        prompt, response, cost = idea_generation_analysis(paper_bank, args.grounding_k, args.ideas_n, topic_description, openai_client, args.engine, args.seed)
    # print ("prompt: ", prompt)
    # print ("ideas: ", response)
    print ("idea generation cost: ", cost)

    ## cache the generated ideas
    response = json.loads(response.strip())
    ideas = {"topic_description": topic_description, "ideas": response}
    ideas_file = os.path.join("../cache_results/ideas", args.cache_name + '_' + args.method + ".json")
    
    ## if the idea_cache already exists, directly add to the current list
    if os.path.exists(ideas_file):
        with open(ideas_file, "r") as f:
            ideas_cache = json.load(f)
        ideas_cache["ideas"].update(response)
        ideas = ideas_cache

    ## save the cache
    if not os.path.exists("../cache_results/ideas"):
        os.makedirs("../cache_results/ideas")
    cache_output(ideas, ideas_file)

