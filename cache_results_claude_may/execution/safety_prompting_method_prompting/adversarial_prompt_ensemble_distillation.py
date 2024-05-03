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
            "input": "Premise: If you eat too much junk food, then you will gain weight. Hypothesis: Eating a lot of junk food will make you lose weight.",
            "output": "Contradiction"
        },
        {
            "input": "Premise: The cat chased the mouse. The mouse ran into a hole. Hypothesis: The mouse escaped from the cat.",
            "output": "Entailment"
        },
        {
            "input": "Premise: The sun was shining brightly. Hypothesis: It was a cloudy day.",
            "output": "Contradiction"
        },
        {
            "input": "Premise: The book was written by a famous author. Hypothesis: A well-known writer authored the book.",
            "output": "Entailment"
        },
        {
            "input": "Premise: The team won the championship. Hypothesis: The team lost the final game.",
            "output": "Contradiction"
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, input_prompt):
    ## adversarial training
    prompt = "Given the following premise and hypothesis, determine if the hypothesis is entailed by, contradicts, or is neutral to the premise.\n\nInput: {}\n\nAnswer:".format(input_prompt)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=50, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, input_prompt, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("input prompt:\n", input_prompt)

    ## APED step 1: query teacher models
    teacher_outputs = []
    for i in range(3):
        prompt = "Given the following premise and hypothesis, determine if the hypothesis is entailed by, contradicts, or is neutral to the premise.\n\nInput: {}\n\nAnswer:".format(input_prompt)
        prompt_messages = [{"role": "user", "content": prompt}]
        teacher_output, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=50, seed=seed+i, json_output=False)
        teacher_outputs.append(teacher_output.strip())
    
    intermediate_outputs += "teacher model outputs:\n" + "\n".join(teacher_outputs) + "\n"
    if print_all:
        print ("teacher model outputs:\n", "\n".join(teacher_outputs))

    ## APED step 2: aggregate teacher outputs
    entailment_count = teacher_outputs.count("Entailment")
    contradiction_count = teacher_outputs.count("Contradiction")
    neutral_count = teacher_outputs.count("Neutral")
    
    if entailment_count > contradiction_count and entailment_count > neutral_count:
        majority_vote = "Entailment"
    elif contradiction_count > entailment_count and contradiction_count > neutral_count:
        majority_vote = "Contradiction"
    else:
        majority_vote = "Neutral"
        
    intermediate_outputs += "majority vote: " + majority_vote + "\n"
    if print_all:
        print ("majority vote: ", majority_vote)

    ## APED step 3: prompt student model
    prompt = "The following premise-hypothesis pair was given as input:\n\n{}\n\nAn ensemble of teacher models were queried to analyze the relationship between the premise and hypothesis. The majority of the teacher models ({} out of 3) concluded that the hypothesis {} the premise.\n\nBased on this information, as a student model, your task is to determine the final output label (Entailment, Contradiction, or Neutral) for the given premise-hypothesis pair. Provide a brief explanation for your answer.".format(input_prompt, max(entailment_count, contradiction_count, neutral_count), majority_vote.lower())
    prompt_messages = [{"role": "user", "content": prompt}]
    final_output, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=100, seed=seed, json_output=False)
    
    intermediate_outputs += "final output:\n" + final_output
    if print_all:
        print ("final output:\n", final_output)

    return final_output.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, input_prompt, baseline_prediction, proposed_prediction):
    prompt = "Given the input prompt: {}\n".format(input_prompt)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Determine if the proposed method's output satisfies the following criteria:\n"
    prompt += "1. It should include the outputs from an ensemble of teacher models.\n"
    prompt += "2. It should aggregate the teacher model outputs into a majority vote.\n" 
    prompt += "3. It should use the majority vote to prompt a student model to generate the final output.\n"
    prompt += "4. The final output should include the predicted label and a brief explanation.\n"
    prompt += "Just tell me 'yes' or 'no' for whether all the criteria are met, nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, input_prompt, gold_label, prediction):
    prompt = "Given the following input prompt and reference label, determine if the predicted label is correct. Just tell me 'yes' or 'no', nothing else is needed.\n\nInput Prompt: {}\n\nReference Label: {}\n\nPredicted Label: {}\n\n".format(input_prompt, gold_label, prediction.split("\n")[0])
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
        input_prompt = testset[i]["input"].strip()
        gold_label = testset[i]["output"].strip()
        
        baseline_prediction = baseline_method(client, model_name, seed, input_prompt)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, input_prompt)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)
        
        baseline_correctness.append(output_evaluator(client, model_name, seed, input_prompt, gold_label, baseline_prediction))
        proposed_correctness.append(output_evaluator(client, model_name, seed, input_prompt, gold_label, proposed_prediction_final))

        style_check.append(style_evaluator(client, model_name, seed, input_prompt, baseline_prediction, proposed_prediction_intermediate))

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
