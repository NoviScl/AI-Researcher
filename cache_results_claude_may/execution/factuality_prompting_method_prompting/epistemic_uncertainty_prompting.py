from openai import OpenAI
import anthropic
import json
import random 
from tqdm import tqdm 
from utils import call_api, load_model
import random
import re
random.seed(2024)

## Step 1: Generate synthetic test examples
def generate_testset():
    test_data = [
        {
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris."
        },
        {
            "input": "Who wrote the novel 'Pride and Prejudice'?",
            "output": "The novel 'Pride and Prejudice' was written by Jane Austen."
        },
        {
            "input": "What is the largest planet in our solar system?",
            "output": "Jupiter is the largest planet in our solar system."
        },
        {
            "input": "When did World War II end?",
            "output": "World War II ended in 1945."
        },
        {
            "input": "Who painted the famous artwork 'The Starry Night'?",
            "output": "The famous artwork 'The Starry Night' was painted by Vincent van Gogh."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, question):
    prompt = "Answer the following question: {}\n".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, question, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("question:\n", question)

    ## uncertainty-aware prompting step 1: augment prompt with uncertainty instructions
    prompt = "Answer the following question: {}\n".format(question)
    prompt += "If you are not completely certain about any part of your answer, please express your uncertainty explicitly using phrases like \"I'm not sure but\" or \"I don't know for certain\"."
    prompt_messages = [{"role": "user", "content": prompt}]
    initial_response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "initial response:\n" + initial_response + "\n"
    if print_all:
        print ("initial response:\n", initial_response)

    ## uncertainty-aware prompting step 2: parse generated text for uncertainty markers
    uncertain_spans = re.findall(r"(I'm not sure but|I don't know for certain)[^.!?]*[.!?]", initial_response)
    intermediate_outputs += "uncertain spans:\n" + str(uncertain_spans) + "\n"
    if print_all:
        print ("uncertain spans:\n", uncertain_spans)

    ## uncertainty-aware prompting step 3: find evidence or rephrase for each uncertain span
    final_response = initial_response
    for span in uncertain_spans:
        prompt = "Find evidence to support the following claim, or rephrase it if you cannot: {}".format(span)
        prompt_messages = [{"role": "user", "content": prompt}]
        evidence_response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
        intermediate_outputs += "evidence for span '{}':\n".format(span) + evidence_response + "\n"
        if print_all:
            print ("evidence for span '{}':\n".format(span), evidence_response)
        
        if "cannot find evidence" in evidence_response.lower():
            final_response = final_response.replace(span, evidence_response)
        else:
            final_response = final_response.replace(span, evidence_response)

    intermediate_outputs += "final response:\n" + final_response
    if print_all:
        print ("final response:\n", final_response)

    return final_response.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    prompt = "Given the task: {}\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Now determine if the proposed method is better by checking if it has satisfied the following criteria:\n"
    prompt += "1. The proposed method's output should include an initial response with uncertainty markers, a list of uncertain spans, evidence or rephrasing for each uncertain span, and a final response.\n"
    prompt += "2. The proposed method should provide a more factual and grounded answer than the baseline method by finding evidence for uncertain claims or rephrasing them.\n"
    prompt += "Just tell me 'yes' or 'no' for whether the criteria are met, nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    prompt = "Given the following question and reference answer, determine if the prediction is correct. Just tell me 'yes' or 'no', nothing else is needed.\n\nQuestion: {}\n\nReference Answer: {}\n\nPrediction: {}\n\n".format(question, gold_label, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 6: Define the function that runs the experiments to obtain model predictions and performance
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

    model_name = "claude-3-opus-20240229"
    seed = 2024 
    client = load_model(model_name)
    print ("using model: ", model_name)

    baseline_correctness, proposed_correctness, style_check = run_experiment(client, model_name, seed, testset)
    print ("baseline correctness: ", sum(baseline_correctness) / len(baseline_correctness))
    print ("proposed correctness: ", sum(proposed_correctness) / len(proposed_correctness))
    print ("style check pass rate: ", sum(style_check) / len(style_check))
