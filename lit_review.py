from openai import OpenAI
from utils import call_api
import argparse
import json
from lit_review_tools import parse_and_execute, format_papers_for_printing

def initial_search(topic_description, openai_client, model):
    prompt = "You are a researcher doing literature review on the topic of " + topic_description.strip() + ".\n"
    prompt += "You should propose some keywords for using the Semantic Scholar API to find the most relevant papers to this topic. Formulate your query as: KeywordQuery(\"keyword\"). Just give me one query, with the most important keyword, the keyword can be a concatenation of multiple keywords (just put a space between every word) but please be concise and try to cover all the main aspects.\n"
    prompt += "Your query:"
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=100, json_output=False)
    
    return prompt, response, cost

def paper_scoring(paper_lst, topic_description, openai_client, model):
    ## use gpt4 to score each paper 
    prompt = "You are a helpful literature review assistant whose job is to read the below set of papers and score each paper. The criteria for scoring are:\n"
    prompt += "(1) The paper is relevant to the topic of: " + topic_description.strip() + ".\n"
    prompt += "(2) The paper is an empirical paper that proposes new methods and conducts experiments.\n"
    prompt += "(3) The paper is interesting and meaningful, with potential to inspire new follow-up projects.\n"
    prompt += "The papers are:\n" + format_papers_for_printing(paper_lst) + "\n"
    prompt += "Please score each paper from 1 to 10. Write the response in JSON format with \"paperID (first 4 digits): score\" for each paper.\n"
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=1000, json_output=True)
    return prompt, response, cost

def collect_papers(topic_description, openai_client, model, max_papers=10):
    paper_bank = {}
    total_cost = 0

    _, response, cost = initial_search(topic_description, openai_client, model)
    total_cost += cost
    paper_lst = parse_and_execute(response)
    paper_bank = {paper["paperId"][:4]: paper for paper in paper_lst}

    prompt, response, cost = paper_scoring(paper_lst, topic_description, openai_client, model)

    response = json.loads(response.strip())

    for k,v in response.items():
        paper_bank[k]["score"] = v
    
    ## rank papers by score
    data_list = [{'id': id, **info} for id, info in paper_bank.items()]
    sorted_data = sorted(data_list, key=lambda x: x['score'])

    return sorted_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    # parser.add_argument('--topic_description', type=str, default=None, help='one-sentence summary of the research topic')
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

    topic_description = "better prompting strategies for large language models to enhance reasoning abilities"
    # prompt, response, cost = initial_search(topic_description, openai_client, MODEL)
    
    # print (prompt)
    # print (response)
    # print (cost)    
    # print (parse_and_execute(response))

    paper_bank = collect_papers(topic_description, openai_client, MODEL, max_papers=10)
    print (paper_bank)


