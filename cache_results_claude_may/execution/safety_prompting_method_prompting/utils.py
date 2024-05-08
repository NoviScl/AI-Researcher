import os
import json
import random 
from openai import OpenAI
import anthropic
from datasets import load_dataset


def calc_price(model, usage):
    if "claude" in model:
        return (0.015 * usage.input_tokens + 0.075 * usage.output_tokens) / 1000.0
    if model == "gpt-4-1106-preview" or model == "gpt-4-0125-preview":
        return (0.01 * usage.prompt_tokens + 0.03 * usage.completion_tokens) / 1000.0
    if model == "gpt-4":
        return (0.03 * usage.prompt_tokens + 0.06 * usage.completion_tokens) / 1000.0
    if (model == "gpt-3.5-turbo") or (model == "gpt-3.5-turbo-1106"):
        return (0.0015 * usage.prompt_tokens + 0.002 * usage.completion_tokens) / 1000.0

def call_api(client, model, prompt_messages, temperature=1.0, max_tokens=100, seed=2024, json_output=False):
    if "claude" in model:
        if json_output:
            prompt = prompt_messages[0]["content"] + " Directly output the JSON dict with no additional text. "
            prompt_messages = [{"role": "user", "content": prompt}]
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=prompt_messages
        )
        cost = calc_price(model, message.usage)
        response = message.content[0].text
    else:   
        response_format = {"type": "json_object"} if json_output else {"type": "text"}
        completion = client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            response_format=response_format
        )
        cost = calc_price(model, completion.usage)
        response = completion.choices[0].message.content.strip()
    
    return response, cost

def call_api_claude(client, model, prompt_messages, temperature=1.0, max_tokens=100):
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=prompt_messages
    )
    cost = calc_price(model, message.usage)
    response = message.content[0].text

    return response, cost

def cache_output(output, file_name):
    if file_name.endswith(".txt"):
        ## store GPT4 output into a txt file
        with open(file_name, "w") as f:
            f.write(output)
    elif file_name.endswith(".json"):
        ## store GPT4 output into a json file
        with open(file_name, "w") as f:
            json.dump(output, f, indent=4)
    return 

def print_idea_json(filename):
    with open(filename, "r") as f:
        idea_json = json.load(f)
    idea = idea_json["final_plan_json"]
    name = idea_json["idea_name"]
    print (name)
    for k,v in idea.items():
        if len(v) > 5:
            print ('- ' + k)
            print (v.strip() + '\n')

def format_plan_json(experiment_plan_json):
    output_str = ""
    for k,v in experiment_plan_json.items():
        if isinstance(v, str):
            output_str += k + ": " + v.strip() + "\n\n"
        else:
            output_str += k + ": " + "\n"
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, str):
                    output_str += "  - " + sub_k + ": " + sub_v.strip() + "\n"
                else:
                    output_str += "  - " + sub_k + ": " + "\n"
                    for sub_sub_k, sub_sub_v in sub_v.items():
                        output_str += "    - " + sub_sub_k + ": " + sub_sub_v.strip() + "\n"
            output_str += "\n"
    return output_str

def shuffle_dict_and_convert_to_string(input_dict):
    # Convert dict items to a list and shuffle
    items = list(input_dict.items())
    random.shuffle(items)
    
    # Convert back to dict and then to a JSON-formatted string
    shuffled_dict = dict(items)
    json_str = json.dumps(shuffled_dict, indent=4)
    
    return json_str


## Load the model being evaluated 
def load_model(model_name):
    with open("../keys.json", "r") as f:
        keys = json.load(f)

    ANTH_KEY = keys["anthropic_key"]
    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    
    if "claude" in model_name:
        client = anthropic.Anthropic(
            api_key=ANTH_KEY,
        )
    else:
        client = OpenAI(
            organization=ORG_ID,
            api_key=OAI_KEY
        )
    
    return client


## Define the metric
def evaluator(client, model_name, seed, question, gold_label, prediction):
    ## we use the simple evaluator of asking the LLM to judge whether the prediction is correct given the gold label
    prompt = "Given the following question and reference answer, determine if the prediction is correct. Just tell me 'yes' or 'no', nothing else is needed.\n\nQuestion: {}\n\nReference Answer: {}\n\nPrediction: {}\n\n".format(question, gold_label, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment

## Load the dataset 
def load_testset(dataset_name, config=None, sample_size=10):
    if not config:
        testset = load_dataset(dataset_name, split="test", cache_dir=cache_dir)
    else:
        testset = load_dataset(dataset_name, config, split="test", cache_dir=cache_dir)
    
    sampled_examples = random.sample(list(testset), sample_size)
    return sampled_examples


if __name__ == "__main__":
    pass