from openai import OpenAI
from utils import call_api
import argparse
import json
from utils import cache_output

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

    topic_description = "better prompting strategies for large language models to improve mathematical problem solving abilities"
    
    paper_bank, total_cost = collect_papers(topic_description, openai_client, MODEL, max_papers=50)
    output = format_papers_for_printing(paper_bank[ : 10])
    print (output)
    print ("Total cost: ", total_cost)

    cache_output(output, "paper_bank_math_reasoning_three_functions.txt")