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
            "input": "Premise: The man on the bench is wearing a black jacket and blue jeans. He has a red beard and is reading a book. Hypothesis: The man is not wearing a black jacket.",
            "output": "Contradiction"
        },
        {
            "input": "Premise: The woman is sitting at a desk, typing on a laptop computer. There are papers and a cup of coffee on the desk. Hypothesis: The woman is using a desktop computer.",
            "output": "Contradiction"
        },
        {
            "input": "Premise: The dog is chasing a ball across the grass in a park. A frisbee is lying on the ground nearby. Hypothesis: The dog is playing fetch with a ball.",
            "output": "Entailment"
        },
        {
            "input": "Premise: The city skyline is visible in the distance, with tall skyscrapers and a large river in the foreground. Hypothesis: The image shows a rural landscape.",
            "output": "Contradiction"
        },
        {
            "input": "Premise: The tennis player is serving the ball on a grass court. The crowd is watching from the stands. Hypothesis: The tennis match is being played on a clay court.",
            "output": "Contradiction"
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, question):
    ## adversarial training
    prompt = "Given the following premise and hypothesis, determine if the hypothesis is entailed by, contradicts, or is neutral to the premise:\n\n{}\n\nAnswer:".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, question, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("question:\n", question)

    ## ARAP step 1: adversarial prompt generation
    prompt = "Generate a prompt that might cause the model to produce an incorrect or inconsistent response, based on the following example:\n\n{}".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    adv_prompt, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=100, seed=seed, json_output=False)
    intermediate_outputs += "adversarial prompt:\n" + adv_prompt + "\n"
    if print_all:
        print ("adversarial prompt:\n", adv_prompt)

    ## ARAP step 2: prompt evaluation and selection
    prompt = "Evaluate the following generated prompt using an ensemble of adversarial attack and defense methods, and select it if it reveals new vulnerabilities or challenges existing defenses:\n\n{}".format(adv_prompt)
    prompt_messages = [{"role": "user", "content": prompt}]
    selection, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=50, seed=seed, json_output=False)
    intermediate_outputs += "prompt selection:\n" + selection + "\n"
    if print_all:
        print ("prompt selection:\n", selection)

    ## ARAP step 3: robust prompt learning  
    prompt = "Analyze why the model might fail on the following prompt and generate a response that addresses the potential vulnerabilities:\n\n{}".format(adv_prompt)
    prompt_messages = [{"role": "user", "content": prompt}]
    robust_response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=100, seed=seed, json_output=False)
    intermediate_outputs += "robust response:\n" + robust_response + "\n"
    if print_all:
        print ("robust response:\n", robust_response)

    ## ARAP step 4: fine-tuning (skipped in this simulation)

    ## ARAP step 5: final evaluation on the original prompt
    prompt = "Given the following premise and hypothesis, determine if the hypothesis is entailed by, contradicts, or is neutral to the premise:\n\n{}\n\nAnswer:".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    final_answer, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    intermediate_outputs += "final answer:\n" + final_answer 
    if print_all:
        print ("final answer:\n", final_answer)

    return final_answer.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    prompt = "Given the task: {}\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Now determine if the proposed method is better by checking if it has satisfied the following criteria:\n"
    prompt += "1. The proposed method's output should include all the key components: adversarial prompt generation, prompt evaluation and selection, and robust response generation.\n"
    prompt += "2. The proposed method should provide a more detailed and comprehensive analysis than the baseline method.\n"
    prompt += "Just tell me 'yes' or 'no' for whether the criteria are met, nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    prompt = "Given the following premise and hypothesis, determine if the predicted label is correct. Just tell me 'yes' or 'no', nothing else is needed.\n\nPremise and Hypothesis: {}\n\nGold Label: {}\n\nPredicted Label: {}\n\n".format(question, gold_label, prediction)
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
