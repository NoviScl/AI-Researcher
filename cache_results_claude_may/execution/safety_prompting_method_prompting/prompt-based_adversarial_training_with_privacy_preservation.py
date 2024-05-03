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
            "input": "I'm sorry but I won't be able to make it to the meeting today. I have a doctor's appointment that I can't miss. Let me know if there's anything I can do to help prepare for the meeting.",
            "output": "I apologize for the short notice, but I have a conflict with today's meeting time. I have a medical appointment that was scheduled in advance and cannot be rescheduled. Please send me the meeting notes and any action items so I can stay informed and contribute as needed. Thank you for your understanding."
        },
        {
            "input": "Hey, can you pick up some groceries on your way home? We need milk, eggs, bread, and some vegetables for dinner. Let me know if you have any questions!",
            "output": "Hi there! Would you mind stopping by the store and grabbing a few items we need? We're running low on some essentials like milk, eggs, bread, and veggies for tonight's meal. Just let me know if you need more details or have any other questions. Thanks so much!"
        },
        {
            "input": "I'm interested in learning more about your company's products and services. Can you send me some information or set up a call to discuss further?",
            "output": "Hello, I came across your company and I'm curious to learn more about what you offer. Would it be possible to get some additional details about your products and services? If it's easier, I'm also happy to hop on a quick call to discuss in more depth. Please let me know what would work best for you. I appreciate your help!"
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, input_text):
    ## non-private adversarial training
    prompt = "Please generate an adversarial example for the following input:\n\n{}\n\nAdversarial example:".format(input_text)
    prompt_messages = [{"role": "user", "content": prompt}]
    adversarial_example, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=200, seed=seed, json_output=False)
    
    prompt = "Given the following input:\n\n{}\n\nAnd the adversarial example:\n\n{}\n\nPlease provide the output for the original input, while being robust to the perturbation in the adversarial example.".format(input_text, adversarial_example)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=200, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, input_text, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("input:\n", input_text)

    ## prompt-based adversarial example generation
    prompt = "Paraphrase the following sentence to change its wording but keep its meaning:\n\n{}\n\nParaphrase:".format(input_text)
    prompt_messages = [{"role": "user", "content": prompt}]
    adversarial_example, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=200, seed=seed, json_output=False)
    intermediate_outputs += "Adversarial example:\n" + adversarial_example + "\n"
    if print_all:
        print ("adversarial example:\n", adversarial_example)

    ## adversarial training
    prompt = "Given the following input:\n\n{}\n\nAnd the adversarial example:\n\n{}\n\nPlease provide the output for the original input, while being robust to the perturbation in the adversarial example.".format(input_text, adversarial_example)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=200, seed=seed, json_output=False)
    intermediate_outputs += "Adversarially trained output:\n" + response + "\n"
    if print_all:
        print ("adversarially trained output:\n", response)

    ## privacy-preserving aggregation (simulated)
    prompt = "Here is the output from adversarial training:\n\n{}\n\nTo protect user privacy, please aggregate this output with other users' outputs using secure multi-party computation. Simulate the aggregation process and return the final privacy-preserved output.".format(response)
    prompt_messages = [{"role": "user", "content": prompt}]
    final_output, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=200, seed=seed, json_output=False)
    intermediate_outputs += "Privacy-preserved output:\n" + final_output
    if print_all:
        print ("privacy-preserved output:\n", final_output)

    return final_output.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, input_text, baseline_prediction, proposed_prediction):
    prompt = "Given the input: {}\n".format(input_text)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Determine if the proposed method's output contains the following desired components:\n"
    prompt += "1. Adversarial example\n2. Adversarially trained output\n3. Privacy-preserved final output\n"
    prompt += "Return a score from 1 to 10 based on how many components are present and the overall quality of the output (higher is better)."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    try:
        score = int(response.strip())
    except:
        score = 0
    
    return score


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, input_text, gold_output, prediction):
    prompt = "Given the following input and reference output, rate the predicted output on a scale of 1 to 10 based on its quality, fluency, and similarity to the reference (higher is better).\n\nInput: {}\n\nReference Output: {}\n\nPredicted Output: {}\n\n".format(input_text, gold_output, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    try:
        score = int(response.strip())
    except:
        score = 0
    
    return score


## Step 6: Define the function that runs the experiments to obtain model predictions and performance
def run_experiment(client, model_name, seed, testset):
    sample_size = len(testset) 
    baseline_predictions = []
    proposed_predictions = []

    baseline_scores = []
    proposed_scores = []

    style_scores = []

    for i in tqdm(range(sample_size)):
        input_text = testset[i]["input"].strip()
        gold_output = testset[i]["output"].strip()
        
        baseline_prediction = baseline_method(client, model_name, seed, input_text)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, input_text)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)
        
        baseline_scores.append(output_evaluator(client, model_name, seed, input_text, gold_output, baseline_prediction))
        proposed_scores.append(output_evaluator(client, model_name, seed, input_text, gold_output, proposed_prediction_final))

        style_scores.append(style_evaluator(client, model_name, seed, input_text, baseline_prediction, proposed_prediction_intermediate))

    return baseline_scores, proposed_scores, style_scores


## Step 7: Execute the experiments and compare performance 
if __name__ == "__main__":
    testset = generate_testset()
    print ("simulated {} test examples for evaluation.".format(len(testset)))

    model_name = "claude-3-opus-20240229" ## don't change this
    seed = 2024 
    client = load_model(model_name)
    print ("using model: ", model_name)

    ## output quality scores
    baseline_scores, proposed_scores, style_scores = run_experiment(client, model_name, seed, testset)
    print ("baseline quality score: ", sum(baseline_scores) / len(baseline_scores))
    print ("proposed quality score: ", sum(proposed_scores) / len(proposed_scores))
    print ("style score: ", sum(style_scores) / len(style_scores))
