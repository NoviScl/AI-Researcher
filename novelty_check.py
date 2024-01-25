from openai import OpenAI
from utils import call_api
import argparse
import json
import os
from lit_review_tools import parse_and_execute, format_papers_for_printing, print_top_papers_from_paper_bank
from utils import cache_output
import random 
import retry
random.seed(2024)

def paper_query(idea, topic_description, openai_client, model, seed):
    # prompt = "You are a professor in Natural Language Processing. You need to evaluate the novelty of a proposed research idea on the topic of: " + topic_description + ".\n\n"
    prompt = "You are a professor in Natural Language Processing. You need to evaluate the novelty of a proposed research idea.\n"

    prompt += "The idea is:\n" + idea + "\n\n"
    prompt += "You want to do a round of paper search in order to find out whether the proposed project has already been done. "
    prompt += "You should propose some keywords for using the Semantic Scholar API to find the most relevant papers to this proposed idea. Formulate your query as: KeywordQuery(\"keyword\"). Give me 1 - 3 queries, the keyword can be a concatenation of multiple keywords (just put a space between every word) but please be concise and try to cover all the main aspects.\n"
    prompt += "The query keywords should be specific to the proposed research idea, in order to find whether there are similar ideas in the literature. try to include language model to find relevant papers within NLP. "
    prompt += "Your query (just return the queries with no additional text, put each one in a new line without any other explanation):"
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=100, seed=seed, json_output=False)
    
    return prompt, response, cost

def paper_scoring(paper_lst, idea, topic_description, openai_client, model, seed):
    ## use gpt4 to score each paper 
    prompt = "You are a research assistant whose job is to read the below set of papers and score each paper based on how similar the paper is to the proposed idea.\n"
    prompt += "The proposed idea is: " + idea.strip() + ".\n"
    prompt += "The topic is " + topic_description.strip() + " and it should be related to large language models and NLP broadly.\n"
    prompt += "The papers are:\n" + format_papers_for_printing(paper_lst) + "\n"
    prompt += "Please score each paper from 1 to 10 based on the similarity and relevance to the proposed idea. 10 means the paper is essentially the same as the proposed idea; 1 means the paper is not even relevant to the topic; 5 means the paper shares some similarity but some key details are different.\n"
    prompt += "Write the response in JSON format with \"paperID: score\" as the key and value for each paper.\n"
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4000, seed=seed, json_output=True)
    return prompt, response, cost

def novelty_score(paper_lst, idea, openai_client, model, seed):
    ## use gpt4 to give novelty judgment
    prompt = "You are a professor whose job is to decide whether your student's proposed research idea is novel or not.\n"
    prompt += "The proposed idea is: " + idea.strip() + ".\n"
    prompt += "You have found a list of related works:\n" + format_papers_for_printing(paper_lst) + "\n"
    prompt += "Now decide if the proposed idea is novel and has not been done by the related works before. You should be critical and only consider the idea novel if none of related works proposed the same idea or did similar experiments.\n"
    prompt += "You should give a binary judgment (yes or no) and a short explanation for your judgment. Your explanation should mention the paper(s) that scooped the idea if you think the idea is not novel. If you think the idea is novel, you should mention why it is sufficiently different from prior works. "
    prompt += "Give the short explanation, then change to a new line and give the binary judgment by ending the response with either \"Yes\" or \"No\".\n"
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=1000, seed=seed, json_output=False)
    return prompt, response, cost

# @retry.retry(tries=3, delay=2)
def novelty_check(idea_name, idea, topic_description, openai_client, model, seed):
    paper_bank = {}
    total_cost = 0
    all_queries = []

    ## get KeywordSearch queries
    _, queries, cost = paper_query(idea, topic_description, openai_client, model, seed)
    total_cost += cost
    print ("queries: \n", queries)
    all_queries = queries.strip().split("\n")
    ## also add the idea name as an additional query
    all_queries.append("KeywordQuery(\"{}\")".format(idea_name + " NLP"))

    for query in all_queries:
        paper_lst = parse_and_execute(query)
        paper_bank.update({paper["paperId"]: paper for paper in paper_lst})

        ## score each paper
        prompt, response, cost = paper_scoring(paper_lst, idea, topic_description, openai_client, model, seed)
        total_cost += cost
        response = json.loads(response.strip())

        ## initialize all scores to 0 then fill in gpt4 scores
        for k,v in response.items():
            if k in paper_bank:
                paper_bank[k]["score"] = v
    
    ## the missing papers will have a score of 0 
    for k,v in paper_bank.items():
        if "score" not in v:
            v["score"] = 0
    
    ## rank all papers by score
    data_list = [{'id': id, **info} for id, info in paper_bank.items()]
    sorted_papers = sorted(data_list, key=lambda x: x['score'], reverse=True)

    ## novelty score
    prompt, novelty, cost = novelty_score(sorted_papers, idea, openai_client, model, seed)
    print (prompt, novelty)

    return novelty, sorted_papers, total_cost, all_queries



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default=None, required=True, help='cache file name for the retrieved papers')
    parser.add_argument('--idea_name', type=str, default=None, required=True, help='the specific idea to be formulated into an experiment plan')
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()

    with open("keys.json", "r") as f:
        keys = json.load(f)

    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    S2_KEY = keys["s2_key"]
    openai_client = OpenAI(
        organization=ORG_ID,
        api_key=OAI_KEY
    )

    ## load the idea
    with open(os.path.join("cache_results/ideas", args.cache_name+".json"), "r") as f:
        ideas = json.load(f)
    topic_description = ideas["topic_description"]
    idea = ideas["ideas"][args.idea_name]

    novelty, paper_bank, total_cost, all_queries = novelty_check(args.idea_name, idea, topic_description, openai_client, args.engine, args.seed)
    output = format_papers_for_printing(paper_bank[ : 10])
    print ("Top 10 papers: ")
    print (output)
    print ("Total cost: ", total_cost)

    if args.cache_name:
        if not os.path.exists("cache_results/novelty_check"):
            os.makedirs("cache_results/novelty_check")
        output_dict = {"topic_description": topic_description, "idea": idea, "all_queries": all_queries, "paper_bank": paper_bank}
        cache_output(output_dict, os.path.join("cache_results/novelty_check", args.cache_name+"_"+"_".join(args.idea_name.lower().split())+".json"))

    
    
