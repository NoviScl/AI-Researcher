from openai import OpenAI
from utils import call_api
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing, parse_and_execute
from utils import cache_output, format_plan_json
from tqdm import tqdm
import random 
random.seed(2024)

def critique(self_critique_prompt, idea_proposal, topic_description, openai_client, model):
    prompt = "You are a professor with expertise in Natural Language Processing. You need to provide some constructive feedback to the given project proposal on the topic of: " + topic_description + ".\n\n"

    prompt += "The project proposal is:\n" + format_plan_json(idea_proposal) + "\n\n"
    prompt += self_critique_prompt

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4000, json_output=False)
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
    # print (prompt)

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=1000, json_output=True)
    return prompt, response, cost

def improve_idea(self_improvement_prompt, criticisms, idea_proposal, topic_description, openai_client, model):
  
    prompt = "You are a researcher with expertise in Natural Language Processing. You have written a project proposal on the topic of: " + topic_description + ".\n"
    prompt += "The original proposal is:\n" + format_plan_json(idea_proposal) + "\n\n"
    prompt += "The proposal has received some feedback and criticisms from reviewers:\n" + criticisms + "\n"
    prompt += self_improvement_prompt
    # print (prompt)

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4000, json_output=True)
    return prompt, response, cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default=None, required=True, help='cache file name for the retrieved papers')
    parser.add_argument('--idea_name', type=str, default=None, required=True, help='the specific idea to be formulated into an experiment plan')
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()

    with open("../keys.json", "r") as f:
        keys = json.load(f)

    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    S2_KEY = keys["s2_key"]
    openai_client = OpenAI(
        organization=ORG_ID,
        api_key=OAI_KEY
    )
    
    with open("prompts/self_critique_prompt.txt", "r") as f:
        self_critique_prompt = f.read()
    with open("prompts/self_improvement_prompt.txt", "r") as f:
        self_improvement_prompt = f.read()

    if args.idea_name == "all":
        filenames = os.listdir("../cache_results/experiment_plans/"+args.cache_name)
    else:
        filenames = ["_".join(args.idea_name.lower().split())+".json"]
    
    for filename in tqdm(filenames):
        print ("working on: ", filename)
        ## load the idea
        cache_file = os.path.join("../cache_results/experiment_plans/"+args.cache_name, filename)
        with open(cache_file, "r") as f:
            ideas = json.load(f)
        idea_name = ideas["idea_name"]
        idea = ideas["raw_idea"]
        topic_description = ideas["topic_description"]
        experiment_plan = ideas["improved_experiment_plan"]

        prompt, criticisms, cost = critique(self_critique_prompt, experiment_plan, topic_description, openai_client, args.engine)
        print ("criticisms: \n", criticisms)

        prompt, new_plan, cost = improve_idea(self_improvement_prompt, criticisms, experiment_plan, topic_description, openai_client, args.engine)
        print ("\nnew plan: \n", new_plan)

        ## cache the improved idea
        if not os.path.exists("../cache_results/experiment_plans/"+args.cache_name):
            os.makedirs("../cache_results/experiment_plans/"+args.cache_name)
        ideas["criticisms"] = criticisms
        final_plan_json = json.loads(new_plan.strip())
        ideas["final_revised_plan"] = final_plan_json
        cache_file = os.path.join("../cache_results/experiment_plans/"+args.cache_name, filename)
        cache_output(ideas, cache_file)
