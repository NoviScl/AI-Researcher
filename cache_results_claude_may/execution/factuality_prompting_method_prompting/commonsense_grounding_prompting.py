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
            "input": "Concepts: dog, frisbee, catch, throw\nTask: Generate a coherent sentence describing a common scenario using the given concepts.",
            "output": "The dog jumped up to catch the frisbee that was thrown."
        },
        {
            "input": "Concepts: cat, yarn, play, paw\nTask: Generate a coherent sentence describing a common scenario using the given concepts.",
            "output": "The cat pawed at the yarn, playing with it."
        },
        {
            "input": "Premise: Alice was considering buying a new car. She had done a lot of research and decided on the one that she wanted.\nTask: Generate a hypothesis that is entailed by the premise.",
            "output": "Alice bought the car that she wanted."
        },
        {
            "input": "Premise: Bob went to the store to buy some groceries. He picked up eggs, milk, and bread.\nTask: Generate a hypothesis that is contradicted by the premise.",
            "output": "Bob did not go to the store."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, input_text):
    prompt = input_text
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, input_text, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("input:\n", input_text)

    ## commonsense grounding step 1: generate commonsense fact
    prompt = input_text + "\nFirst state a relevant commonsense fact about the concepts."
    prompt_messages = [{"role": "user", "content": prompt}]
    commonsense_fact, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "commonsense fact:\n" + commonsense_fact + "\n"
    if print_all:
        print ("commonsense fact:\n", commonsense_fact)

    ## commonsense grounding step 2: generate grounded output
    prompt = input_text + "\nCommonsense fact: " + commonsense_fact + "\nNow generate the output."
    prompt_messages = [{"role": "user", "content": prompt}]
    grounded_output, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "grounded output:\n" + grounded_output
    if print_all:
        print ("grounded output:\n", grounded_output)

    return grounded_output.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, input_text, baseline_prediction, proposed_prediction):
    prompt = "Given the input: {}\n".format(input_text)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Now determine if the proposed method is better by checking if it has satisfied the following criteria:\n"
    prompt += "1. The proposed method's output should include a relevant commonsense fact.\n"  
    prompt += "2. The proposed method's output should be grounded in the commonsense fact and be more consistent with commonsense.\n"
    prompt += "Just tell me 'yes' or 'no' for whether the criteria are met, nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, input_text, gold_label, prediction):
    prompt = "Given the following input and reference output, rate the predicted output on a scale of 1-10 based on its quality and consistency with the reference. 1 means completely inconsistent, 10 means equivalent to the reference.\n\nInput: {}\n\nReference Output: {}\n\nPredicted Output: {}\n\n".format(input_text, gold_label, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2, seed=seed, json_output=False)
    
    try:
        score = int(response.strip())
    except:
        score = 1

    return score


## Step 6: Define the function that runs the experiments to obtain model predictions and performance
def run_experiment(client, model_name, seed, testset):
    sample_size = len(testset) 
    baseline_predictions = []
    proposed_predictions = []

    baseline_scores = []
    proposed_scores = []

    style_check = []

    for i in tqdm(range(sample_size)):
        input_text = testset[i]["input"].strip()
        gold_label = testset[i]["output"].strip()
        
        baseline_prediction = baseline_method(client, model_name, seed, input_text)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, input_text)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)
        
        baseline_scores.append(output_evaluator(client, model_name, seed, input_text, gold_label, baseline_prediction))
        proposed_scores.append(output_evaluator(client, model_name, seed, input_text, gold_label, proposed_prediction_final))

        style_check.append(style_evaluator(client, model_name, seed, input_text, baseline_prediction, proposed_prediction_intermediate))

    return baseline_scores, proposed_scores, style_check


## Step 7: Execute the experiments and compare performance 
if __name__ == "__main__":
    testset = generate_testset()
    print ("simulated {} test examples for evaluation.".format(len(testset)))

    model_name = "claude-3-opus-20240229" ## don't change this
    seed = 2024 
    client = load_model(model_name)
    print ("using model: ", model_name)

    ## output scores 
    baseline_scores, proposed_scores, style_check = run_experiment(client, model_name, seed, testset)
    print ("baseline score: ", sum(baseline_scores) / len(baseline_scores))
    print ("proposed score: ", sum(proposed_scores) / len(proposed_scores))
    print ("style check pass rate: ", sum(style_check) / len(style_check))
