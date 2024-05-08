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
            "input": "Claim: The movie Inception was directed by Christopher Nolan.",
            "output": "The claim is true. Christopher Nolan directed the 2010 science fiction film Inception."
        },
        {
            "input": "Claim: Albert Einstein was born in Germany.",
            "output": "The claim is true. Albert Einstein was born on March 14, 1879, in Ulm, Germany."
        },
        {
            "input": "Claim: The Great Wall of China is visible from space.",
            "output": "The claim is false. The Great Wall of China is not visible from space, according to NASA."
        },
        {
            "input": "Claim: The Mona Lisa was painted by Vincent van Gogh.",
            "output": "The claim is false. The Mona Lisa was painted by Leonardo da Vinci, not Vincent van Gogh."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, claim):
    ## zero-shot claim verification
    prompt = "Verify the following claim: {}\n".format(claim)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, claim, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("claim:\n", claim)

    ## iterative knowledge distillation step 1: initial generation
    prompt = "Given the following claim, generate a relevant output to verify the claim: {}".format(claim)
    prompt_messages = [{"role": "user", "content": prompt}]
    initial_output, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "initial output:\n" + initial_output + "\n"
    if print_all:
        print ("initial output:\n", initial_output)

    ## iterative knowledge distillation step 2: knowledge extraction
    prompt = "Extract concise factual statements from the following generated output: \n{}".format(initial_output)
    prompt_messages = [{"role": "user", "content": prompt}]
    extracted_knowledge, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "extracted knowledge:\n" + extracted_knowledge + "\n"
    if print_all:
        print ("extracted knowledge:\n", extracted_knowledge)

    ## iterative knowledge distillation step 3: revised generation  
    prompt = "Given the following claim and extracted factual knowledge, generate a revised output to verify the claim: \nClaim: {}\nExtracted Knowledge: {}".format(claim, extracted_knowledge)
    prompt_messages = [{"role": "user", "content": prompt}]
    revised_output, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "revised output:\n" + revised_output + "\n"
    if print_all:
        print ("revised output:\n", revised_output)

    ## iterative knowledge distillation step 4: stopping criterion
    prompt = "Based on the extracted factual knowledge and the revised output, should we perform another iteration of knowledge distillation and output revision? (Yes/No)"
    prompt_messages = [{"role": "user", "content": prompt}]
    stopping_criterion, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "stopping criterion:\n" + stopping_criterion 
    if print_all:
        print ("stopping criterion:\n", stopping_criterion)

    if stopping_criterion.strip().lower() == "yes":
        final_output, intermediate_outputs_2 = proposed_method(client, model_name, seed, claim, print_all)
        intermediate_outputs += "\n" + intermediate_outputs_2
    else:
        final_output = revised_output

    return final_output.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, claim, baseline_prediction, proposed_prediction):
    ## define all the components that the proposed method outputs should have
    ## and the advantages of the proposed method over the baseline method
    ## just need to check the style is correct
    prompt = "Given the claim: {}\n".format(claim)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Now determine if the proposed method is better by checking if it has satisfied the following criteria:\n"
    prompt += "1. The proposed method's output should produce all the intermediate components including: initial output, extracted knowledge, revised output, and stopping criterion.\n"
    prompt += "2. The proposed method should provide a more concise and factually consistent answer than the baseline method.\n"
    prompt += "Just tell me 'yes' or 'no' for whether the criteria are met, nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, claim, gold_label, prediction):
    ## check if the prediction is correct given the gold label
    prompt = "Given the following claim and reference answer, determine if the prediction is correct. Just tell me 'yes' or 'no', nothing else is needed.\n\nClaim: {}\n\nReference Answer: {}\n\nPrediction: {}\n\n".format(claim, gold_label, prediction)
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
        claim = testset[i]["input"].strip()
        gold_label = testset[i]["output"].strip()
        
        baseline_prediction = baseline_method(client, model_name, seed, claim)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, claim)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)
        
        baseline_correctness.append(output_evaluator(client, model_name, seed, claim, gold_label, baseline_prediction))
        proposed_correctness.append(output_evaluator(client, model_name, seed, claim, gold_label, proposed_prediction_final))

        style_check.append(style_evaluator(client, model_name, seed, claim, baseline_prediction, proposed_prediction_intermediate))

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
