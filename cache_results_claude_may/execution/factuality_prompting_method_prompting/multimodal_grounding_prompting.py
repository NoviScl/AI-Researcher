from openai import OpenAI
import anthropic
import json
import random 
from tqdm import tqdm 
from utils import call_api, load_model
import random
random.seed(2024)

## Step 1: Generate synthetic test examples
def generate_testset():
    test_data = [
        {
            "input": {
                "image": "An image of a person riding a bicycle on a city street.",
                "query": "What is the person doing?"
            },
            "output": "The person is riding a bicycle on a city street."
        },
        {
            "input": {
                "image": "An image of a group of people sitting around a table eating pizza.",
                "query": "What are the people eating?"
            },
            "output": "The people are eating pizza."
        },
        {
            "input": {
                "image": "An image of a dog catching a frisbee in a park.",
                "query": "What is the dog playing with?"
            },
            "output": "The dog is playing with a frisbee in the park."
        },
        {
            "input": {
                "image": "An image of a woman holding a bouquet of red roses.",
                "query": "What color are the flowers?"
            },
            "output": "The flowers in the bouquet are red roses."
        },
        {
            "input": {
                "image": "An image of a man standing in front of the Eiffel Tower.",
                "query": "Where is the man standing?"
            },
            "output": "The man is standing in front of the Eiffel Tower."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, image, query):
    ## direct prompting
    prompt = "Given the image: {} and the query: {}, generate a claim that is relevant to the image.".format(image, query)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, image, query, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("image:\n", image)
        print ("query:\n", query)

    ## visual grounding step 1: generate claim
    prompt = "Given the image: {} and the query: {}, generate a claim that is relevant to the image.".format(image, query)
    prompt_messages = [{"role": "user", "content": prompt}]
    claim, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "initial claim:\n" + claim + "\n"
    if print_all:
        print ("initial claim:\n", claim)

    ## visual grounding step 2: assess consistency
    prompt = "Given the image: {} and the generated claim: {}, assess whether the claim is consistent with the visual evidence in the image. Identify any inconsistencies.".format(image, claim)
    prompt_messages = [{"role": "user", "content": prompt}]
    inconsistencies, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "inconsistencies:\n" + inconsistencies + "\n"
    if print_all:
        print ("inconsistencies:\n", inconsistencies)

    ## visual grounding step 3: revise claim  
    prompt = "Given the image: {}, the original claim: {}, and the identified inconsistencies: {}, revise the claim to be more consistent with the image.".format(image, claim, inconsistencies)
    prompt_messages = [{"role": "user", "content": prompt}]
    revised_claim, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "revised claim:\n" + revised_claim + "\n"
    if print_all:
        print ("revised claim:\n", revised_claim)

    return revised_claim.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, image, query, baseline_prediction, proposed_prediction):
    ## check if the proposed method output contains all the desired components
    prompt = "Given the image: {}\nand the query: {}\n".format(image, query)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Now determine if the proposed method is better by checking if it has satisfied the following criteria:\n"
    prompt += "1. The proposed method's output should include an initial claim, identified inconsistencies, and a revised claim.\n"
    prompt += "2. The proposed method should provide a more detailed and factually grounded answer than the baseline method.\n"
    prompt += "Just tell me 'yes' or 'no' for whether the criteria are met, nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, image, query, gold_label, prediction):
    ## check if the prediction is correct given the gold label
    prompt = "Given the following image, query, and reference answer, determine if the prediction is correct. Just tell me 'yes' or 'no', nothing else is needed.\n\nImage: {}\n\nQuery: {}\n\nReference Answer: {}\n\nPrediction: {}\n\n".format(image, query, gold_label, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 6: Define the function that runs the experiments to obtain model predictions and performance
## you shouldn't need to modify this function in most cases
def run_experiment(client, model_name, seed, testset):
    sample_size = len(testset) 
    baseline_predictions = []
    proposed_predictions = []

    baseline_correctness = []
    proposed_correctness = []

    style_check = []

    for i in tqdm(range(sample_size)):
        image = testset[i]["input"]["image"].strip()
        query = testset[i]["input"]["query"].strip()
        gold_label = testset[i]["output"].strip()
        
        baseline_prediction = baseline_method(client, model_name, seed, image, query)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, image, query)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)
        
        baseline_correctness.append(output_evaluator(client, model_name, seed, image, query, gold_label, baseline_prediction))
        proposed_correctness.append(output_evaluator(client, model_name, seed, image, query, gold_label, proposed_prediction_final))

        style_check.append(style_evaluator(client, model_name, seed, image, query, baseline_prediction, proposed_prediction_intermediate))

    return baseline_correctness, proposed_correctness, style_check


## Step 7: Execute the experiments and compare performance 
if __name__ == "__main__":
    testset = generate_testset()
    print ("simulated {} test examples for evaluation.".format(len(testset)))

    model_name = "claude-3-opus-20240229" ## don't change this
    seed = 2024 
    client = load_model(model_name)
    print ("using model: ", model_name)

    ## output correctness 
    baseline_correctness, proposed_correctness, style_check = run_experiment(client, model_name, seed, testset)
    print ("baseline correctness: ", sum(baseline_correctness) / len(baseline_correctness))
    print ("proposed correctness: ", sum(proposed_correctness) / len(proposed_correctness))
    print ("style check pass rate: ", sum(style_check) / len(style_check))
