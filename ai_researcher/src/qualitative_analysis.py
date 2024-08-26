from openai import OpenAI
import anthropic
from utils import call_api
import argparse
import json
import os
from tqdm import tqdm
import random 
import retry

@retry.retry(tries=3, delay=2)
def summarize_reviews(reviews, openai_client, model, seed):
    prompt = "Help me do some qualitative analysis. I have collected expert reviews on a set of AI ideas (including AI and AI_Rerank) and a set of human ideas. I want you to summarize the main differences between AI ideas and human ideas, focusing their unique strengths and weaknesses. For the output format, I want you to first list the main conclusions, and then for each conclusion, make sure to quote some examples as evidence. The conclusions should be things like \"Human ideas are generally more grounded in existing research and practical considerations, but may be less innovative.\" or \"AI ideas sometimes make unrealistic assumptions about model capabilities.\", rather than just comparing the scores.\n\n"
    prompt += "Here are the reviews:\n\n"
    prompt += reviews + "\n\n"
    
    prompt_messages = [{"role": "user", "content": prompt}]
    response, cost = call_api(openai_client, model, prompt_messages, temperature=0., max_tokens=4096, seed=seed, json_output=False)
    return prompt, response, cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='claude-3-5-sonnet-20240620', help='api engine')
    parser.add_argument('--seed', type=int, default=2024, help="seed for GPT-4 generation")
    args = parser.parse_args()

    with open("../keys.json", "r") as f:
        keys = json.load(f)
    random.seed(args.seed)

    ANTH_KEY = keys["anthropic_key"]
    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    
    if "claude" in args.engine:
        client = anthropic.Anthropic(
            api_key=ANTH_KEY,
        )
    else:
        client = OpenAI(
            organization=ORG_ID,
            api_key=OAI_KEY
        )
    
    with open("../results/data_points_dedup.json", "r") as f:
        data = json.load(f)
    
    # Initialize variables for batching
    review_batch = []
    counter = 0
    bucket_size = 30
    
    for i in range(len(data["name"])):
        review = f"""Idea ID: {data['idea_id'][i]}
Novelty Score: {data['novelty_score'][i]}
Novelty Rationale: {data['novelty_rationale'][i]}
Feasibility Score: {data['feasibility_score'][i]}
Feasibility Rationale: {data['feasibility_rationale'][i]}
Effectiveness Score: {data['effectiveness_score'][i]}
Effectiveness Rationale: {data['effectiveness_rationale'][i]}
Excitement Score: {data['excitement_score'][i]}
Excitement Rationale: {data['excitement_rationale'][i]}
Overall Score: {data['overall_score'][i]}
Overall Rationale: {data['overall_rationale'][i]}
"""

        review_batch.append(review)
        counter += 1
        
        if counter == bucket_size:
            # Concatenate reviews into a single string
            reviews_concatenated = "\n".join(review_batch)
            # Summarize the batch
            prompt, response, cost = summarize_reviews(reviews_concatenated, client, args.engine, args.seed)
            print(response)
            print ("-------------------------------------\n")
            
            # Reset batch
            review_batch = []
            counter = 0
    
    # Handle any remaining reviews that didn't complete a full batch 
    if review_batch:
        reviews_concatenated = "\n".join(review_batch)
        prompt, response, cost = summarize_reviews(reviews_concatenated, client, args.engine, args.seed)
        print(response)
        print ("-------------------------------------\n")
