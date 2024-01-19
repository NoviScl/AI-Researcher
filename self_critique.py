from openai import OpenAI
from utils import call_api
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing, parse_and_execute
from utils import cache_output
import random 
random.seed(2024)

def critique(idea_proposal, topic_description, openai_client, model):
    prompt = "You are a reviewer with expertise in Natural Language Processing. You need to criticize project proposals on the topic of: " + topic_description + ".\n\n"

    prompt += "The project proposal is:\n" + idea_proposal + "\n\n"
    prompt += "Please come up with a list questions and critical comments targeting the weaknesses of proposal.\n"
    prompt += "You should focus solely on missing details in the proposal, mainly:\n"
    prompt += "1. did the proposal list the datasets to be used?\n"
    prompt += "2. did the proposal describe every single step of the proposed method or experiment?\n"
    prompt += "3. did the proposal describe the evaluation metrics?\n"
    prompt += "You do not have to worry about subjective matters like broader impact of ethical concerns. Your questions and critic:"

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=2400, json_output=False)
    return prompt, response, cost

def more_lit_review(grounding_papers, idea_proposal, critic, topic_description, openai_client, model):
    grounding_papers_str = format_papers_for_printing(grounding_papers)

    prompt = "You are a researcher with expertise in Natural Language Processing. You have received a project proposal on the topic of: " + topic_description + ".\n"
    prompt += "The proposal is:\n" + idea_proposal + "\n\n"
    prompt += "The proposal has received some feedback from reviewers:\n" + critic + "\n"
    prompt += "You should do a round of literature review to find relevant papers that can help address these feedback."
    prompt += "You are allowed to use the following functions for making queries with Semantic Scholar API:\n"
    prompt += "(1) KeywordQuery(\"keyword\"): find most relevant papers to the given keyword (the keyword shouldn't be too long and specific, otherwise the search engine will fail; it is ok to combine a few shor keywords with spaces, such as \"lanaguage model reasoning\").\n"
    prompt += "(2) PaperQuery(\"paperId\"): find the most similar papers to the given paper (as specified by the paperId).\n"
    prompt += "(3) GetReferences(\"paperId\"): get the list of papers referenced in the given paper (as specified by the paperId).\n"
   
    prompt += "Right now you have already collected the following relevant papers:\n" + grounding_papers_str + "\n"
    prompt += "You should formulate new search queries based on the above to find additional papers that can help address the feeback and improve the project proposal."
    prompt += "Now directly give me your new queries (following the function definitions above). You are allowed to generate at most 3 different queries (focus on the most important ones), put each one in a new line without any other texts. If you feel that you have already collected enough papers, you can just give me an empty line to skip this step."
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=500, json_output=False)

    all_papers = []
    query_lst = response.strip().split("\n")
    for query in query_lst:
        paper_lst = parse_and_execute(query)
        if paper_lst:
            all_papers.extend(paper_lst)

    return prompt, response, cost, all_papers

def paper_scoring(paper_lst, topic_description, critic, openai_client, model):
    ## use gpt4 to score each paper 
    prompt = "You are a helpful literature review assistant whose job is to read the below set of papers and score each paper. The criteria for scoring are:\n"
    prompt += "(1) The paper is relevant to the topic of: " + topic_description.strip() + ".\n"
    prompt += "(2) The paper is directly relevant to help address one of the critics here:\n" + critic.strip() + "\n\n"
    prompt += "The papers are:\n" + format_papers_for_printing(paper_lst) + "\n"
    prompt += "Please score each paper from 1 to 10. Write the response in JSON format with \"paperID (first 4 digits): score\" for each paper.\n"
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=1000, json_output=True)
    return prompt, response, cost

def improve_idea(grounding_papers, new_grounding_papers, idea_proposal, critique, topic_description, openai_client, model):
    grounding_papers_str = format_papers_for_printing(grounding_papers)
    new_grounding_papers_str = format_papers_for_printing(new_grounding_papers)

    prompt = "You are a researcher with expertise in Natural Language Processing. You have a project proposal on the topic of: " + topic_description + ".\n"
    prompt += "The original proposal is:\n" + idea_proposal + "\n\n"
    # prompt += "It is grounded on a list of relevant papers here:\n" + grounding_papers_str + "\n"
    prompt += "The proposal has received some feedback from reviewers:\n" + critique + "\n"
    # prompt += "You have done a round of literature review to find additional papers that can help address these feedback:\n" + new_grounding_papers_str + "\n"
    prompt += "Now you should improve the proposal based on the feedback and new papers."
    prompt += "Please write down your improved proposal in the same format as the original proposal, make sure to expand the method section with all necessary details to address all relevant comments. Do not only refer to the reference papers, instead give the exact algorithm or method details."

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=3200, json_output=False)
    return prompt, response, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--idea_file', type=str, default="ai_idea_0.txt", help='a txt file containing the idea proposal')
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
    
    with open(os.path.join("cache_results/old", args.idea_file), "r") as f:
        idea_proposal = f.read()

    topic_description = "better prompting strategies for large language models to improve mathematical problem solving abilities"
    # prompt, response, cost = critique(idea_proposal, topic_description, openai_client, MODEL)
    
    # topic_description = "better prompting strategies for large language models to improve multi-step problem solving abilities"
    # prompt, response, cost = critique(idea_proposal, topic_description, openai_client, MODEL)

    # print (response)
    # print ("Total cost: ", cost)

    # cache_output(response, "critique_" + args.idea_file)

    with open(os.path.join("cache_results/old", "paper_bank_math_reasoning_max60.json"), "r") as f:
        paper_bank = json.load(f)
    grounding_papers = paper_bank[ : 10]

    with open(os.path.join("cache_results/old", "critique_ai_idea_0.txt"), "r") as f:
        critic = f.read()

    # prompt, response, cost, more_papers = more_lit_review(grounding_papers, idea_proposal, critic, topic_description, openai_client, MODEL)
    # print (response)

    # ## dedup
    # new_papers = []
    # old_papers = format_papers_for_printing(paper_bank)
    # for paper in more_papers:
    #     if paper["paperId"] not in old_papers:
    #         paper["id"] = paper["paperId"][:4]
    #         new_papers.append(paper)
    
    # print ("# new papers: ", len(new_papers))
    
    # new_papers_bank = {paper["paperId"][:4]: paper for paper in new_papers}
    # # print ("new papers: ", format_papers_for_printing(new_papers))
    # _, response, cost = paper_scoring(new_papers, topic_description, critic, openai_client, MODEL)
    # response = json.loads(response.strip())
    # # print ("scoring response: ", response)
    
    # ## initialize all scores to 0
    # for k,v in new_papers_bank.items():
    #     new_papers_bank[k]["score"] = 0
    # for k,v in response.items():
    #     new_papers_bank[k]["score"] = v
    
    # ## rank new papers by score
    # data_list = [{'id': id, **info} for id, info in new_papers_bank.items()]
    # sorted_data = sorted(data_list, key=lambda x: x['score'], reverse=True)
    # cache_output(sorted_data, "paper_bank_math_reasoning_new_grounding_papers_method.json")   

    # new_grounding_papers = sorted_data[ : 10]
    # print ("new grounding papers: ", format_papers_for_printing(new_grounding_papers))

    with open(os.path.join("cache_results", "paper_bank_math_reasoning_new_grounding_papers_method.json"), "r") as f:
        new_grounding_papers = json.load(f)
    new_grounding_papers = new_grounding_papers[ : 10]

    prompt, response, cost = improve_idea(grounding_papers, new_grounding_papers, idea_proposal, critic, topic_description, openai_client, MODEL)
    print (response)
    