from openai import OpenAI
from utils import call_api
import argparse
import json
from lit_review_tools import parse_and_execute, format_papers_for_printing, print_top_papers_from_paper_bank
from utils import cache_output

def initial_search(topic_description, openai_client, model):
    prompt = "You are a researcher doing literature review on the topic of " + topic_description.strip() + ".\n"
    prompt += "You should propose some keywords for using the Semantic Scholar API to find the most relevant papers to this topic. Formulate your query as: KeywordQuery(\"keyword\"). Just give me one query, with the most important keyword, the keyword can be a concatenation of multiple keywords (just put a space between every word) but please be concise and try to cover all the main aspects.\n"
    prompt += "Your query (just return the query itself with no additional text):"
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=100, json_output=False)
    
    return prompt, response, cost

def next_query(topic_description, openai_client, model, grounding_papers, past_queries):
    grounding_papers_str = format_papers_for_printing(grounding_papers)
    prompt = "You are a researcher doing literature review on the topic of " + topic_description.strip() + ".\n"
    prompt += "You should propose some queries for using the Semantic Scholar API to find the most relevant papers to this topic. You are allowed to use the following functions for making queries:\n"
    prompt += "(1) KeywordQuery(\"keyword\"): find most relevant papers to the given keyword (the keyword shouldn't be too long and specific, otherwise the search engine will fail; it is ok to combine a few shor keywords with spaces, such as \"lanaguage model reasoning\").\n"
    prompt += "(2) PaperQuery(\"paperId\"): find the most similar papers to the given paper (as specified by the paperId).\n"
    prompt += "(3) GetReferences(\"paperId\"): get the list of papers referenced in the given paper (as specified by the paperId).\n"
    # prompt += "(3) GetCitations(\"paperId\"): get the list of papers that cited the given paper (as specified by the paperId).\n"
    prompt += "Right now you have already collected the following relevant papers:\n" + grounding_papers_str + "\n"
    prompt += "You can formulate new search queries based on these papers. And you have already asked the following queries:\n" + "\n".join(past_queries) + "\n"
    prompt += "Please formulate a new query to expand our paper collection with more diverse and relevant papers (you can do so by diversifying the types of queries to generate and minimize repeating previous queries; e.g., try not to only use KeywordSearch). Directly give me your new query without any explanation or additional text, just the query itself:"
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=100, json_output=False)

    return prompt, response, cost

def paper_scoring(paper_lst, topic_description, openai_client, model):
    ## use gpt4 to score each paper 
    prompt = "You are a helpful literature review assistant whose job is to read the below set of papers and score each paper. The criteria for scoring are:\n"
    prompt += "(1) The paper is relevant to the topic of: " + topic_description.strip() + ".\n"
    prompt += "(2) The paper is an empirical paper that proposes new methods and conducts experiments (position/opinion papers, review/survey papers, and resource/benchmark/evaluation papers should get low scores for this purpose).\n"
    prompt += "(3) The paper is interesting and meaningful, with potential to inspire new follow-up projects.\n"
    prompt += "The papers are:\n" + format_papers_for_printing(paper_lst) + "\n"
    prompt += "Please score each paper from 1 to 10. Write the response in JSON format with \"paperID (first 4 digits): score\" for each paper.\n"
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=1000, json_output=True)
    return prompt, response, cost

def collect_papers(topic_description, openai_client, model, grounding_k = 10, max_papers=50, print_all=True):
    paper_bank = {}
    total_cost = 0
    all_queries = []

    ## get initial set of seed papers by KeywordSearch 
    _, query, cost = initial_search(topic_description, openai_client, model)
    total_cost += cost
    all_queries.append(query)
    paper_lst = parse_and_execute(query)
    paper_bank = {paper["paperId"][:4]: paper for paper in paper_lst}
    ## initialize all scores to 0
    for k,v in paper_bank.items():
        v["score"] = 0
    _, response, cost = paper_scoring(paper_lst, topic_description, openai_client, model)
    total_cost += cost
    response = json.loads(response.strip())
    for k,v in response.items():
        paper_bank[k]["score"] = v
    
    ## print stats 
    if print_all:
        print ("initial query: ", query)
        print ("current total cost: ", total_cost)
        print ("current size of paper bank: ", len(paper_bank))
        print_top_papers_from_paper_bank(paper_bank, top_k=10)
        print ("\n")
    
    iter = 0
    ## keep expanding the paper bank until limit is reached
    while len(paper_bank) < max_papers and iter < 10:
        ## select the top k papers with highest scores for grounding
        data_list = [{'id': id, **info} for id, info in paper_bank.items()]
        grounding_papers = sorted(data_list, key=lambda x: x['score'], reverse=True)[ : grounding_k]
        _, new_query, cost = next_query(topic_description, openai_client, model, grounding_papers, all_queries)
        all_queries.append(new_query)
        total_cost += cost 
        if print_all:
            print ("new query: ", new_query)
        paper_lst = parse_and_execute(new_query)
        
        if paper_lst:
            ## filter out papers already in paper bank 
            paper_lst = [paper for paper in paper_lst if paper["paperId"][:4] not in paper_bank]
            ## initialize all scores to 0
            for paper in paper_lst:
                paper["score"] = 0
            paper_bank.update({paper["paperId"][:4]: paper for paper in paper_lst})
            _, response, cost = paper_scoring(paper_lst, topic_description, openai_client, model)
            total_cost += cost
            response = json.loads(response.strip())
            for k,v in response.items():
                paper_bank[k]["score"] = v
        else:
            print ("No new papers found in this round.")
        
        ## print stats 
        if print_all:
            print ("current total cost: ", total_cost)
            print ("current size of paper bank: ", len(paper_bank))
            print_top_papers_from_paper_bank(paper_bank, top_k=10)
            print ("\n")
        
        iter += 1
    
    ## rank all papers by score
    data_list = [{'id': id, **info} for id, info in paper_bank.items()]
    sorted_data = sorted(data_list, key=lambda x: x['score'], reverse=True)

    return sorted_data, total_cost


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

    topic_description = "automatic evaluation methods and metrics for text-to-image diffusion models"
    
    paper_bank, total_cost = collect_papers(topic_description, openai_client, MODEL, max_papers=60)
    output = format_papers_for_printing(paper_bank[ : 10])
    print (output)
    print ("Total cost: ", total_cost)

    # cache_output(output, "paper_bank_math_reasoning_three_functions.txt")
    cache_output(paper_bank, "paper_bank_diffusion_eval_max60.json")
