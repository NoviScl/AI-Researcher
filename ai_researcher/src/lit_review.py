from openai import OpenAI
import anthropic
from utils import call_api, format_plan_json
import argparse
import json
from lit_review_tools import parse_and_execute, format_papers_for_printing, print_top_papers_from_paper_bank, dedup_paper_bank
from utils import cache_output
import os
import retry

def initial_search(topic_description, openai_client, model, seed, mode="topic", idea=None):
    if mode == "topic":
        prompt = "You are a researcher doing literature review on the topic of " + topic_description.strip() + ".\n"
        prompt += "You should propose some keywords for using the Semantic Scholar API to find the most relevant papers to this topic. "
    elif mode == "idea":
        prompt = "You are a professor. You need to evaluate the novelty of a proposed research idea.\n"
        prompt += "The idea is:\n" + format_plan_json(idea) + "\n\n"
        prompt += "You want to do a round of paper search in order to find out whether the proposed project has already been done. "
        prompt += "You should propose some keywords for using the Semantic Scholar API to find the most relevant papers to this proposed idea. "
    
    prompt += "Formulate your query as: KeywordQuery(\"keyword\"). Just give me one query, with the most important keyword, the keyword can be a concatenation of multiple keywords (just put a space between every word) but please be concise and try to cover all the main aspects.\n"
    prompt += "Your query (just return the query itself with no additional text):"
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=100, seed=seed, json_output=False)
    
    return prompt, response, cost

def next_query(topic_description, openai_client, model, seed, grounding_papers, past_queries, mode="topic", idea=None):
    grounding_papers_str = format_papers_for_printing(grounding_papers)
    if mode == "topic":
        prompt = "You are a researcher. You are doing literature review on the topic of " + topic_description.strip() + ".\n" + "You should propose some queries for using the Semantic Scholar API to find the most relevant papers to this topic. "
    elif mode == "idea":
        prompt = "You are a professor. You need to evaluate the novelty of a proposed research idea.\n"
        prompt += "The idea is:\n" + format_plan_json(idea) + "\n\n"
        prompt += "You want to do a round of paper search in order to find out whether the proposed project has already been done. "
        prompt += "You should propose some queries for using the Semantic Scholar API to find the most relevant papers to this proposed idea. "
    prompt += "You are allowed to use the following functions for making queries:\n"
    prompt += "(1) KeywordQuery(\"keyword\"): find most relevant papers to the given keyword (the keyword shouldn't be too long and specific, otherwise the search engine will fail; it is ok to combine a few shor keywords with spaces, such as \"lanaguage model reasoning\").\n"
    prompt += "(2) PaperQuery(\"paperId\"): find the most similar papers to the given paper (as specified by the paperId).\n"
    prompt += "(3) GetReferences(\"paperId\"): get the list of papers referenced in the given paper (as specified by the paperId).\n"
    prompt += "Right now you have already collected the following relevant papers:\n" + grounding_papers_str + "\n"
    prompt += "You can formulate new search queries based on these papers. And you have already asked the following queries:\n" + "\n".join(past_queries) + "\n"
    prompt += "Please formulate a new query to expand our paper collection with more diverse and relevant papers (you can do so by diversifying the types of queries to generate and minimize the overlap with previous queries). Directly give me your new query without any explanation or additional text, just the query itself:"
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=100, seed=seed, json_output=False)

    return prompt, response, cost

def paper_score(paper_lst, topic_description, openai_client, model, seed, mode="topic", idea=None):
    ## use gpt4 to score each paper, focusing on method papers
    prompt = "You are a helpful literature review assistant whose job is to read the below set of papers and score each paper. The criteria for scoring are:\n"
    
    if mode == "topic": 
        prompt += "(1) The paper is directly relevant to the topic of: " + topic_description.strip() + ". Note that it should be specific to solve the problem of focus, rather than just generic methods. "
    elif mode == "idea":
        prompt += "(1) The paper is directly relevant to the proposed idea: " + format_plan_json(idea) + "\nNote that it should be specific to solve a similar problem or use a similar approach, rather than just generic methods. "

    if "prompting" in topic_description:
        prompt += "The proposed method should be about new prompting methods, rather than other techniques like finetuning or pretraining.\n"
    elif "finetuning" in topic_description:
        prompt += "The proposed method should be about new finetuning methods, rather than other techniques like prompting.\n"
    else:
        prompt += "\n"

    prompt += "(2) The paper is an empirical paper that proposes a novel method and conducts empirical experiments to show improvement over baselines (position or opinion papers, review or survey papers, and analysis papers should get low scores for this purpose).\n"
    prompt += "(3) The paper is interesting, exciting, and meaningful, with potential to inspire many new projects.\n"
    prompt += "The papers are:\n" + format_papers_for_printing(paper_lst) + "\n"
    prompt += "Please score each paper from 1 to 10. Write the response in JSON format with \"paperID: score\" as the key and value for each paper.\n"
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4000, seed=seed, json_output=True)
    return prompt, response, cost


@retry.retry(tries=3, delay=2)
def collect_papers(topic_description, openai_client, model, seed, grounding_k = 10, max_papers=60, print_all=True, mode = "topic", idea=None):
    paper_bank = {}
    total_cost = 0
    all_queries = []

    ## get initial set of seed papers by KeywordSearch 
    _, query, cost = initial_search(topic_description, openai_client, model, seed, mode=mode, idea=idea)
    total_cost += cost
    all_queries.append(query)
    paper_lst = parse_and_execute(query)
    if paper_lst:
        ## filter out those with incomplete abstracts
        paper_lst = [paper for paper in paper_lst if paper["abstract"] and len(paper["abstract"].split()) > 50]
        paper_bank = {paper["paperId"]: paper for paper in paper_lst}

        ## score each paper
        _, response, cost = paper_score(paper_lst, topic_description, openai_client, model, seed, mode=mode, idea=idea)
        total_cost += cost
        response = json.loads(response.strip())

        ## initialize all scores to 0 then fill in gpt4 scores
        for k,v in paper_bank.items():
            v["score"] = 0
        for k,v in response.items():
            try:
                paper_bank[k]["score"] = v
            except:
                continue 
    else:
        paper_lst = []

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
        
        ## generate the next query
        _, new_query, cost = next_query(topic_description, openai_client, model, seed, grounding_papers, all_queries, mode=mode, idea=idea)
        all_queries.append(new_query)
        total_cost += cost 
        if print_all:
            print ("new query: ", new_query)
        try:
            paper_lst = parse_and_execute(new_query)
        except:
            paper_lst = None 
        
        if paper_lst:
            ## filter out papers already in paper bank 
            paper_lst = [paper for paper in paper_lst if paper["abstract"] and len(paper["abstract"].split()) > 50]
            paper_lst = [paper for paper in paper_lst if paper["paperId"] not in paper_bank]
            
            ## initialize all scores to 0 and add to paper bank
            for paper in paper_lst:
                paper["score"] = 0
            paper_bank.update({paper["paperId"]: paper for paper in paper_lst})
            
            ## gpt4 score new papers
            _, response, cost = paper_score(paper_lst, topic_description, openai_client, model, seed, mode=mode, idea=idea)
            total_cost += cost
            response = json.loads(response.strip())
            for k,v in response.items():
                try:
                    paper_bank[k]["score"] = v
                except:
                    continue

        elif print_all:
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
    sorted_data = dedup_paper_bank(sorted_data)

    return sorted_data, total_cost, all_queries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--seed', type=int, default=2024, help='seed for GPT-4 generation')
    parser.add_argument('--mode', type=str, default='topic', help='do lit review either based on topic or idea')
    parser.add_argument('--topic_description', type=str, default='automatic evaluation methods and metrics for text-to-image diffusion models', help='one-sentence topic description')
    parser.add_argument('--idea_cache', type=str, default='cache_results_claude_may/experiment_plans_claude3-5/uncertainty_prompting', help='cache file for ideas')
    parser.add_argument('--idea_name', type=str, default='adaptive_uncertainty_sampling', help='idea name')
    parser.add_argument('--max_paper_bank_size', type=int, default=60, help='max number of papers to score and store in the paper bank')
    parser.add_argument('--grounding_k', type=int, default=10, help='how many papers for grounding when generating next queries')
    parser.add_argument('--cache_name', type=str, help='give a name for the output cache file; leave it None if no need caching')
    parser.add_argument('--print_all', action='store_true', help='whether to print out the intermediate process')
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

    idea = {}
    if args.mode == "idea":
        with open("{}/{}".format(args.idea_cache, args.idea_name), "r") as f:
            idea_json = json.load(f)
        idea = idea_json["full_experiment_plan"]
        topic_description = idea_json["topic_description"]
    elif args.mode == "topic":
        topic_description = args.topic_description

    paper_bank, total_cost, all_queries = collect_papers(topic_description, client, args.engine, args.seed, args.grounding_k, max_papers=args.max_paper_bank_size, print_all=args.print_all, mode=args.mode, idea=idea)
    output = format_papers_for_printing(paper_bank[ : 10])
    print ("Top 10 papers: ")
    print (output)
    print ("Total cost: ", total_cost)

    if args.cache_name:
        cache_dir = os.path.dirname(args.cache_name)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        output_dict = {"topic_description": args.topic_description, "all_queries": all_queries, "paper_bank": paper_bank}
        cache_output(output_dict, args.cache_name)


