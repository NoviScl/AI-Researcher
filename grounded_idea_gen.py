from openai import OpenAI
from utils import call_api
import argparse
import json
import os
from lit_review_tools import format_papers_for_printing
from utils import cache_output
import random 
from lit_review import collect_papers
random.seed(2024)

def idea_generation(paper_bank, grounding_k, papers_n, topic_description, openai_client, model):
    ## retrieve top papers 
    grounding_papers = paper_bank[ : grounding_k]
    random.shuffle(grounding_papers)

    prompt = "You are an expert researcher in Natural Language Processing. Now I want you to help me brainstorm some new research project ideas on the topic of: " + topic_description + ".\n\n"
    prompt += "Here are some relevant papers that I have found for you:\n" + format_papers_for_printing(grounding_papers) + "\n"
    prompt += "You should generate {} ideas that are within the same scope, but should be novel and different from the papers above. You can use the papers above as inspiration, but you should not copy them directly. You should also make sure that your ideas are not too broad or complicated. Try to include all the necessary methodology details and experiment setups.\n".format(str(papers_n))
    prompt += "Please write down your ideas (each idea should be described as one paragraph. Output the ideas in json format as a dictionary, where you should generate a short idea name as the key and the actual idea description as the value."

    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0.7, max_tokens=2400, json_output=True)
    return prompt, response, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    parser.add_argument('--topic_description', type=str, default=None, help='one-sentence summary of the research topic')
    parser.add_argument('--grounding_k', type=int, default=10, help='how many papers to use for grounding')
    parser.add_argument('--papers_n', type=int, default=5, help="how many ideas to generate")
    args = parser.parse_args()

    with open("keys.json", "r") as f:
        keys = json.load(f)

    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    MODEL = args.engine
    openai_client = OpenAI(
        organization=ORG_ID,
        api_key=OAI_KEY
    )

    print ("topic: ", args.topic_description)
    print ("collecting papers...")
    
    # paper_bank = []
    # while len(paper_bank) == 0:
    #     try:
    paper_bank, total_cost, all_queries = collect_papers(args.topic_description, openai_client, MODEL, max_papers=60, print_all=True)
        # except:
        #     continue 

    print ("")
    top_10 = format_papers_for_printing(paper_bank[ : 10])
    print ("top 10 papers: ", top_10)
    print ("paper retrieval cost: ", total_cost)

    print ("\n")
    print ("generating ideas...")
    prompt, response, cost = idea_generation(paper_bank, args.grounding_k, args.topic_description, openai_client, MODEL)
    print ("ideas: ", response)
    print ("idea generation cost: ", cost)


