from openai import OpenAI
from utils import calc_price
import argparse

def simulate_response(prompts, openai_client, model, survey_record):
    full_prompt = prompts["response_prompt"] + survey_record
    completion = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=1.0,
    )
    cost = calc_price(model, completion.usage)
    response = completion.choices[0].message.content.strip()
    response = response.replace("Response (1 - 5):", "").replace("Response:", "").strip()

    return response, cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--apikey', type=str, required=True, help='api key; https://openai.com/api/')
    parser.add_argument('--engine', type=str, default='gpt-4-1106-preview', help='api engine; https://openai.com/api/')
    args = parser.parse_args()

    OAI_KEY = args.apikey
    openai_client = OpenAI(
        organization='',
        api_key=OAI_KEY
    )
    if args.engine:
        MODEL = args.engine

