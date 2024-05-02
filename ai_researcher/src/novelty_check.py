from openai import OpenAI
import anthropic
from utils import call_api
import argparse
import json
import os
from lit_review_tools import parse_and_execute, format_papers_for_printing, print_top_papers_from_paper_bank, dedup_paper_bank
from utils import cache_output, format_plan_json
from self_improvement import get_related_works
import random 
from tqdm import tqdm
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

def novelty_score(experiment_plan, related_paper, openai_client, model, seed):
    ## use gpt4 to give novelty judgment wrt one individual paper 
    prompt = "You are a professor specialized in Natural Language Processing. You have a project proposal and want to find related works to cite in the paper. Your job is to decide whether the given paper is directly relevant to the project and should be cited as similar work.\n"
    prompt += "The project proposal is:\n" + json.dumps(experiment_plan).strip() + ".\n"
    prompt += "The paper is:\n" + format_papers_for_printing([related_paper], include_score=False) + "\n"
    # prompt += "The project proposal and the paper abstract is considered a good match if they are studying the same research problem with similar approaches. Note that the paper abstract must cover all topics mentioned in the project proposal. For example, if the proposal is about retrieval augmentation for improving code generation, then the paper needs to mention both retrieval augmentation and improving code generation in order to consider a good match. Being only partially relevant does not count. For example, if the project is about retrieval augmented code generation while the paper is about self-improving code generation, then it doesn't count. Note that the details do not matter (such as which datasets are used), you should only focus on the high-level idea.\n"
    # prompt += "For analysis type of project proposal, the project proposal and paper abstract are considered a match as long as the research problem is the same. For example, if they are both studying how dialect variations affect language model confidence. The exact experiments do not matter. But note that the paper should mention all key concepts of the project proposal rather than just partial match. For example, for a proposal analyzing dialect variations' impact on language model confidence, the paper should be related to both dialect variations as well as model confidence. Only mentioning one does not count as a match.\n"
    # prompt += "For new method type of project proposal, the project proposal and paper abstract are considered a match if both the research problem and the approach are the same. For example, if they are both trying to improve code generation accuracy and both proposing to use retrieval augmentation. Note that the method details do not matter, you should only focus on the high-level concepts and judge whether they are directly relevant.\n"
    prompt += "The project proposal and paper abstract are considered a match if both the research problem and the approach are the same. For example, if they are both trying to improve code generation accuracy and both propose to use retrieval augmentation. Note that the method details do not matter, you should only focus on the high-level concepts and judge whether they are directly relevant.\n"
    prompt += "You should first specify what is the proposed research problem and approach. If answering yes, your explanation should be the one-sentence summary of both the abstract and the proposal and their similarity (e.g., they are both about probing biases of language models via fictional characters). If answering no, give the short summaries of the abstract and proposal separately, then highlight their differences. Then end your response with a binary judgment, saying either \"Yes\" or \"No\". Change to a new line after your explanation and just say Yes or No with no punctuation in the end.\n"
    # print (prompt)

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=1000, seed=seed, json_output=False)
    return prompt, response, cost


@retry.retry(tries=3, delay=2)
def novelty_check(idea_name, idea, topic_description, openai_client, model, seed):
    paper_bank = {}
    total_cost = 0
    all_queries = []

    ## get KeywordSearch queries
    _, queries, cost = paper_query(idea, topic_description, openai_client, model, seed)
    total_cost += cost
    # print ("queries: \n", queries)
    all_queries = queries.strip().split("\n")
    ## also add the idea name as an additional query
    all_queries.append("KeywordQuery(\"{}\")".format(idea_name + " NLP"))

    for query in all_queries:
        print ("current query: ", query.strip())
        paper_lst = parse_and_execute(query.strip())
        if paper_lst is None:
            continue
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
            
        # print (paper_bank)
        print_top_papers_from_paper_bank(paper_bank, top_k=10)
        print ("-----------------------------------\n")
    
    ## the missing papers will have a score of 0 
    for k,v in paper_bank.items():
        if "score" not in v:
            v["score"] = 0
    
    ## rank all papers by score
    data_list = [{'id': id, **info} for id, info in paper_bank.items()]
    sorted_papers = sorted(data_list, key=lambda x: x['score'], reverse=True)
    sorted_data = dedup_paper_bank(sorted_data)

    ## novelty score
    prompt, novelty, cost = novelty_score(sorted_papers, idea, openai_client, model, seed)
    # print (prompt, novelty)

    return novelty, sorted_papers, total_cost, all_queries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--cache_name', type=str, default=None, required=True, help='cache file name for the retrieved papers')
    parser.add_argument('--idea_name', type=str, default=None, required=True, help='the specific idea to be formulated into an experiment plan')
    parser.add_argument('--check_n', type=int, default=5, help="number of top papers to check for novelty")
    parser.add_argument('--retrieve', action='store_true', help='whether to do the paper retrieval')
    parser.add_argument('--novelty', action='store_true', help='whether to do the novelty check')
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

    if "claude" in args.engine:
        cache_dir = "../cache_results_claude_may/"
    else:
        cache_dir = "../cache_results_gpt4/"
    
    # with open(cache_dir + "ideas/" + args.cache_name + ".json") as f:
    #     idea_file = json.load(f)
    # topic_description = idea_file["topic_description"]
    # all_ideas = idea_file["ideas"]

    if args.idea_name == "all":
        filenames = os.listdir(cache_dir + "experiment_plans/" + args.cache_name)
        # idea_names = list(all_ideas.keys())
    else:
        filenames = ["_".join(args.idea_name.lower().split())+".json"]
        # idea_names = [args.idea_name]
    
    novel_idea = 0
    for filename in tqdm(filenames):
        print ("working on: ", filename)
        cache_file = os.path.join(cache_dir + "experiment_plans/" + args.cache_name + "/" + filename)
        with open(cache_file, "r") as f:
            idea_file = json.load(f)
        idea_name = idea_file["idea_name"]
        idea = idea_file["full_experiment_plan"]
        topic_description = idea_file["topic_description"]
        
        if args.retrieve:
            print ("Retrieving related works...")
            paper_bank, total_cost, all_queries = get_related_works(idea_name, idea, topic_description, client, args.engine, args.seed)
            output = format_papers_for_printing(paper_bank[ : 10])
            print ("Top 10 papers: ")
            print (output)
            print ("Total cost: ", total_cost)

            ## save the paper bank
            if not os.path.exists(cache_dir + "experiment_plans/" + args.cache_name + "/"):
                os.makedirs(cache_dir + "experiment_plans/" + args.cache_name + "/")
            idea_file["novelty_queries"] = all_queries
            idea_file["novelty_papers"] = paper_bank
            cache_output(idea_file, cache_file)
        
        if args.novelty: 
            with open(cache_file, "r") as f:
                idea_file = json.load(f)

            topic_description = idea_file["topic_description"]
            plan_json = idea_file["full_experiment_plan"]
            related_papers = idea_file["novelty_papers"]
            # idea_file["novelty_check_papers"] = related_papers[ : args.check_n].copy()

            novel = True 
            print ("checking through top {} papers".format(str(args.check_n)))
            for i in range(args.check_n):
                prompt, response, cost = novelty_score(plan_json, related_papers[i], client, args.engine, args.seed)
                idea_file["novelty_papers"][i]["novelty_score"] = response.strip()
                final_judgment = response.strip().split()[-1].lower()
                # print ("novelty judgment: ", final_judgment)
                # print ("\n\n")
                
                if final_judgment == "yes":
                    novel = False
                idea_file["novelty_papers"][i]["novelty_judgment"] = final_judgment
                
                # print (format_papers_for_printing([related_papers[i]]))
                # print (response)
                # print (cost)
            
            idea_file["novelty"] = "yes" if novel else "no"
            if idea_file["novelty"] == "yes":
                novel_idea += 1
            cache_output(idea_file, cache_file)
            print ("Novelty judgment: ", idea_file["novelty"])

    if args.novelty:
        print ("Novelty rate: {} / {} = {}%".format(novel_idea, len(filenames), novel_idea / len(filenames) * 100))
    
