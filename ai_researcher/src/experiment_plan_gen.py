from openai import OpenAI
import anthropic
from utils import call_api
import argparse
import json
import os
from utils import cache_output, format_plan_json
import retry
from tqdm import tqdm
import random 
random.seed(2024)

@retry.retry(tries=3, delay=2)
def plan_generation_method(method, idea, demo_examples, topic_description, openai_client, model, seed):
    ## forumate an idea from a paragraph into a full experiment plan based on our template

    prompt = "You are an expert researcher in AI and your job is to expand a brief project idea into a full project proposal with detailed experiment plans so that your students can follow the steps and execute the full project. I will provide you with an idea on the topic of: " + topic_description + ".\n\n"
    prompt += "The idea is:\n" + format_plan_json(idea) + "\n"
    prompt += "Now you should come up with the full experiment plan covering:\n"
    prompt += "1. Title: A concise statement of the main research question to be used as the paper title.\n"
    prompt += "2. Problem Statement: Clearly define the problem your research intends to address. Explain clearly why this problem is interesting and important.\n"
    prompt += "3. Motivation: Explain why existing methods (both classic ones and recent ones) are not good enough to solve the problem, and explain the inspiration behind the new proposed method. You should also motivate why the proposed method would work better than existing baselines on the problem.\n"
    prompt += "4. Proposed Method: Explain how the proposed method works, describe all the steps. Make sure every step is clearly described and feasible to implement.\n"
    prompt += "5. Step-by-Step Experiment Plan: Break down every single step of the experiments, make sure every step is executable. Cover all essential details such as the datasets, models, and metrics to be used. If the project involves prompting, give example prompts for each step. If the project involves training, describe the training data, objective, and evaluation metrics.\n"
    prompt += "6. Test Case Examples: Give two concrete examples. The first example should show how the baseline method fails on the test case. If there are multiple baselines, give examples for all of them. The second example should show how the proposed method succeeds on the test case. For each test case, include the input and the expected output. You should also provide an explanation for why the outputs from the proposed prompt are better. If the proposed method has multiple steps, break them down into intermediate steps.\n"
    prompt += "7. Fallback Plan: Propose some alternative plans for what should the students do if the proposed method didn't manage to satisfy the success criteria. For example, you can suggest additional analysis to help debug why the proposed method didn't work, which could inform alternative new methods, or turn the project into an analysis paper instead by offering some interesting ablation and insights. Write a coherent paragraph rather than a list of bullet points.\n"
    prompt += "The experiment plan should not include any background introduction (you can skip the literature review, paper writing tips, and ethical discussion). Just give instructions on the experiments.\n"
    if method == "prompting":
        prompt += "When designing experiments, avoid pretraining new models from scratch and instead prefer prompting-based methods. In rare cases, finetuning small open-source language models is also acceptable. Avoid large-scale human evaluation or human data collection, which is time-consuming and expensive. In general, you should assume that the students have access to abundant black-box LLM API credits (e.g., GPT, Claude, and Gemini) but very limited GPU resources for running training jobs (but running inferences on open-source models is affordable).\n"
    prompt += "Be consistent in your methodology and experiment design, for example, if you will use black-box LLM APIs such as GPT and Claude for your experiments, then you shouldn't propose any experiments that require white-box model weights or data access and you should edit them accordingly to follow the black-box assumptions.\n"
    prompt += "Below are a few examples of how the full experiment plans should look like:\n"
    prompt += demo_examples + "\n\n"
    prompt += "Now please write down your experiment plan in JSON format (keys should be the section names, just like the above examples). Make sure to be as detailed as possible so that a student can directly follow the plan to implement the project."
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4096, seed=seed, json_output=True)
    return prompt, response, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--idea_cache_dir', type=str, default=None, required=True, help='dir that stores all the raw ideas')
    parser.add_argument('--experiment_plan_cache_dir', type=str, default=None, required=True, help='dir to store all the generated experiment plans')
    parser.add_argument('--cache_name', type=str, default=None, required=True, help='the specific cache (topic)')
    parser.add_argument('--idea_name', type=str, default=None, required=True, help='the specific idea to be formulated into an experiment plan')
    parser.add_argument('--method', type=str, default='prompting', help='either prompting or finetuning')
    parser.add_argument('--grounding_k', type=int, default=10, help='how many papers to use for grounding')
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()

    with open("../keys.json", "r") as f:
        keys = json.load(f)

    ANTH_KEY = keys["anthropic_key"]
    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    S2_KEY = keys["s2_key"]
    
    if "claude" in args.engine:
        client = anthropic.Anthropic(
            api_key=ANTH_KEY,
        )
    else:
        client = OpenAI(
            organization=ORG_ID,
            api_key=OAI_KEY
        )

    ## load the demo examples
    with open("prompts/experiment_plan_examples_prompting.txt", "r") as f:
        demo_examples = f.read().strip()
    
    with open(args.idea_cache_dir + args.cache_name + ".json") as f:
        idea_file = json.load(f)
    topic_description = idea_file["topic_description"]
    all_ideas = idea_file["ideas"]

    if args.idea_name == "all":
        idea_names = list(all_ideas.keys())
    else:
        idea_names = [args.idea_name]
    
    if not os.path.exists(args.experiment_plan_cache_dir + args.cache_name + "/"):
        os.makedirs(args.experiment_plan_cache_dir + args.cache_name + "/")
        
    all_costs = 0
    for idea_name in tqdm(idea_names):
        cache_file = os.path.join(args.experiment_plan_cache_dir + args.cache_name + "/" + '_'.join(idea_name.lower().split()) + ".json")
        
        try:            
            idea_file = {}
            idea_file["topic_description"] = topic_description
            idea_file["idea_name"] = idea_name
            idea_file["raw_idea"] = all_ideas[idea_name]

            print ("working on: ", idea_name)
            idea = all_ideas[idea_name]
            
            prompt, response, cost = plan_generation_method(args.method, idea, demo_examples, topic_description, client, args.engine, args.seed)
            # print (response)
            print ("cost: ", cost)

            all_costs += cost
            experiment_plan = json.loads(response.strip())
            idea_file["full_experiment_plan"] = experiment_plan

            cache_output(idea_file, cache_file)

        except: 
            print ("error in generating experiment plan for idea: ", idea_name)
    
    print ("Total cost: ", all_costs)

    