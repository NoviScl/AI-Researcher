from openai import OpenAI
from utils import call_api
import argparse
import json

def initial_search(topic_description, openai_client, model):
    prompt = "You are a researcher doing literature review on the topic of " + topic_description.strip() + ".\n"
    prompt += "You should propose some keywords for using the Semantic Scholar API to find the most relevant papers to this topic. Formulate your query as: KeywordQuery(keyword). Just give me one query, with the most important keyword.\n"
    prompt += "Your query:"
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=100)
    
    return prompt, response, cost
    

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
    prompt, response, cost = initial_search(topic_description, openai_client, MODEL)
    
    print (prompt)
    print (response)
    print (cost)    


