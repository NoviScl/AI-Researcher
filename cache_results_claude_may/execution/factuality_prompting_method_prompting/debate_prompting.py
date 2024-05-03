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
            "input": "The Eiffel Tower is the tallest building in Paris.",
            "output": "REFUTES"
        },
        {
            "input": "The Great Wall of China is visible from space.",
            "output": "REFUTES"
        },
        {
            "input": "The capital of Australia is Sydney.",
            "output": "REFUTES"
        },
        {
            "input": "The Earth is the third planet from the Sun.",
            "output": "SUPPORTS"
        },
        {
            "input": "The Mona Lisa was painted by Leonardo da Vinci.",
            "output": "SUPPORTS"
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, question):
    ## self-consistency scoring
    prompt = "Determine if the following statement is true or false: {}\n".format(question)
    prompt += "Provide a score from 1 to 5, where 1 means the statement is definitely false and 5 means the statement is definitely true."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, question, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("question:\n", question)

    ## debate round 1: Model A generates initial output
    prompt = "Determine if the following statement is true or false: {}\n".format(question)
    prompt += "Provide your reasoning."
    prompt_messages = [{"role": "user", "content": prompt}]
    model_a_output, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Model A Initial Output:\n" + model_a_output + "\n\n"
    if print_all:
        print ("Model A Initial Output:\n", model_a_output)

    ## debate round 2: Model B critiques Model A's output
    prompt = "Identify any factual inaccuracies or inconsistencies in the following statement: {}\n".format(model_a_output)
    prompt += "Provide evidence to support your arguments."
    prompt_messages = [{"role": "user", "content": prompt}]
    model_b_critique, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Model B Critique:\n" + model_b_critique + "\n\n"
    if print_all:
        print ("Model B Critique:\n", model_b_critique)

    ## debate round 3: Model A refutes Model B's critique  
    prompt = "Defend the factual accuracy of your original statement in light of the following counterarguments: {}\n".format(model_b_critique)
    prompt += "Provide evidence to support your defense."
    prompt_messages = [{"role": "user", "content": prompt}]
    model_a_rebuttal, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Model A Rebuttal:\n" + model_a_rebuttal + "\n\n"
    if print_all:
        print ("Model A Rebuttal:\n", model_a_rebuttal)

    ## aggregate arguments and make final judgment
    prompt = "Given the initial statement: {}\n".format(question)
    prompt += "And the following debate:\n"
    prompt += "Model A Initial Output: {}\n".format(model_a_output)
    prompt += "Model B Critique: {}\n".format(model_b_critique) 
    prompt += "Model A Rebuttal: {}\n".format(model_a_rebuttal)
    prompt += "Determine if the initial statement is SUPPORTED or REFUTED by the debate. Provide your reasoning."
    prompt_messages = [{"role": "user", "content": prompt}]
    debate_outcome, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Debate Outcome:\n" + debate_outcome
    if print_all:
        print ("Debate Outcome:\n", debate_outcome)

    return debate_outcome.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    ## check if the proposed method output contains all the desired components
    prompt = "Given the initial statement: {}\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed debate method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Determine if the debate method output contains all of the following components:\n"
    prompt += "1. Model A Initial Output\n2. Model B Critique\n3. Model A Rebuttal\n4. Debate Outcome\n"
    prompt += "Just answer 'yes' or 'no', nothing else."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    ## check if the prediction matches the gold label
    prompt = "Given the initial statement: {}\n".format(question)
    prompt += "The gold label is: {}\n".format(gold_label)
    prompt += "The model predicted: {}\n".format(prediction)
    prompt += "Determine if the model's prediction matches the gold label. Just answer 'yes' or 'no', nothing else."
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
        question = testset[i]["input"].strip()
        gold_label = testset[i]["output"].strip()
        
        baseline_prediction = baseline_method(client, model_name, seed, question)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, question)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)
        
        baseline_correctness.append(output_evaluator(client, model_name, seed, question, gold_label, baseline_prediction))
        proposed_correctness.append(output_evaluator(client, model_name, seed, question, gold_label, proposed_prediction_final))

        style_check.append(style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction_intermediate))

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
