from openai import OpenAI
from utils import call_api
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing
from utils import cache_output
import random 
random.seed(2024)

def plan_generation(grounding_papers, idea, topic_description, openai_client, model):
    ## forumate an idea from a paragraph into a full experiment plan based on our template

    prompt = "You are an expert researcher in Natural Language Processing and your job is to turn a vague project idea into a detailed experiment plan. I will provide you with an idea on the topic of: " + topic_description + ".\n\n"
    prompt += "The idea is:\n" + idea + "\n\n"
    prompt += "Here are some relevant papers that are relevant to the idea:\n" + format_papers_for_printing(grounding_papers) + "\n\n"
    prompt += "Now you should come up with the full proposal following my template:\n"
    prompt += "1. Title: A concise statement of the main research question\n"
    prompt += "2. Problem Statement: Clearly define the problem your research intends to address.\n"
    prompt += "3. Research Objectives and Questions: List the specific objectives and research questions. Formulate the hypotheses.\n"
    prompt += "4. Background & Motivation: Summarize related works and motivations for the proposed project. Discuss how your work is different from or builds upon previous studies.\n"
    prompt += "5. Data: Describe the data you will use, if you want to use existing datasets: include sources; if you want to collect new datasets, describe how it should be collected or constructed.\n"
    prompt += "6. Method: Describe the models to be used, and the method to be implemented. Include all necessary details.\n"
    prompt += "7. Experimental Design: Detail the experimental setup, including any control (baselines) and experimental groups, variables, and the process of experimentation.\n"
    prompt += "8. Evaluation Metrics: Define how you will measure the success of your experiments (accuracy, F1 score, etc.).\n"
    prompt += "9. Expected Results: Discuss the potential outcomes and what you expect to discover or demonstrate through your research.\n\n"
    prompt += "Now please write down your experiment plan (each section should be described as a few concise sentences; index the sections and separate them with new lines). Make sure to be as detailed as possible especially for the data and methods, so that a student can directly follow the plan to implement the experiment. For example, if the method involves prompting, describe the prompts in detail. Give prompt or algorithm description for all methods involved."

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4096, json_output=False)
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
    grounding_papers = paper_bank[ : args.grounding_k]
    
    topic_description = "better prompting strategies for large language models to improve mathematical problem solving abilities"
    idea = "Develop a \"Mathematical Reasoning Augmentation through Contradiction (MRAC)\" prompting strategy for large language models. This approach involves training LLMs to identify contradictions in their own mathematical reasoning by comparing multiple generated solutions to a given problem. Unlike existing methods that attempt to rectify errors post-hoc or rely on self-consistency, MRAC prompts the model to generate diverse solutions and then explicitly identifies and reconciles contradictions among these solutions. This strategy forces the model to reflect critically on its reasoning process, leading to better problem-solving skills. We would use a diverse set of mathematical problems and evaluate the model's performance by comparing its reasoning and solutions against a benchmark dataset. The methodology would involve a controlled A/B testing setup where Group A receives MRAC prompts, and Group B receives traditional CoT prompts, to assess the efficacy of the MRAC strategy."
    
    prompt, response, cost = plan_generation(grounding_papers, idea, topic_description, openai_client, MODEL)
    print (response)
    print ("Total cost: ", cost)

    cache_output(response, "experiment_plan_math.txt")