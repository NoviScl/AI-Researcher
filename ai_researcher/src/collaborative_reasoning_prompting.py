from openai import OpenAI
import anthropic
import json
import random 
from utils import call_api
from datasets import load_dataset
import random
random.seed(2024)


## Step 1: Load the dataset 
def load_testset(dataset_name, config=None, sample_size=10):
    if not config:
        testset = load_dataset(dataset_name, split="test")
    else:
        testset = load_dataset(dataset_name, config, split="test")
    
    sampled_examples = random.sample(list(testset), sample_size)
    return sampled_examples


## Step 2: Load the model being evaluated 
def load_model(model_name):
    with open("../../../keys.json", "r") as f:
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


## Step 3: Define the metric
def evaluator(client, model_name, seed, question, gold_label, prediction):
    ## we use the simple evaluator of asking the LLM to judge whether the prediction is correct given the gold label
    prompt = "Given the following question and reference answer, determine if the prediction is correct. Just tell me 'yes' or 'no', nothing else is needed.\n\nQuestion: {}\n\nReference Answer: {}\n\nPrediction: {}\n\n".format(question, gold_label, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    return response


## Step 4: Implement the baseline method 

## Step 5: Implement the proposed method 

## Step 6: Execute the experiments and compare performance 
if __name__ == "__main__":
    dataset_name = "gsm8k"
    testset = load_testset(dataset_name, config="main")
    print ("sampled {} examples from {} for evaluation.".format(len(testset), dataset_name))

    model_name = "claude-3-opus-20240229"
    client = load_model(model_name)
    print ("using model: ", model_name)

    seed = 2024 
    print (evaluator(client, model_name, seed, "What is the capital of France?", "Paris", "The capital is Paris."))

    # model_name = "claude"
    # testset = load_dataset(dataset_name)
    # client = load_model(model_name)
    # print (client)
    # print (testset)
    # print ("done")