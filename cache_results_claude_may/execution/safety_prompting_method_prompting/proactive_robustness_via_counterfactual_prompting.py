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
            "input": "Passage: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Question: Where is the Eiffel Tower located?",
            "output": "The Eiffel Tower is located in Paris, France."
        },
        {
            "input": "Passage: The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China to protect against nomadic invasions from the Eurasian Steppe. Question: What was the purpose of building the Great Wall of China?",
            "output": "The Great Wall of China was built to protect against nomadic invasions from the Eurasian Steppe."
        },
        {
            "input": "Passage: The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor in New York City, in the United States. The copper statue, a gift from the people of France to the people of the United States, was designed by French sculptor Frédéric Auguste Bartholdi and its metal framework was built by Gustave Eiffel. Question: Who designed the Statue of Liberty?",
            "output": "The Statue of Liberty was designed by French sculptor Frédéric Auguste Bartholdi."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, passage, question):
    ## standard question answering
    prompt = "Passage: {}\nQuestion: {}\nAnswer:".format(passage, question)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, passage, question, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("passage:\n", passage)
        print ("question:\n", question)

    ## PRCP step 1: generate adversarial example
    prompt = "Generate a question that the model is likely to answer incorrectly based on the given passage.\nPassage: {}".format(passage)
    prompt_messages = [{"role": "user", "content": prompt}]
    adversarial_question, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "adversarial question:\n" + adversarial_question + "\n"
    if print_all:
        print ("adversarial question:\n", adversarial_question)

    ## PRCP step 2: generate counterfactual prompt
    prompt = "What kind of question could the model be vulnerable to based on the given passage? Generate a question that the model should be able to answer correctly.\nPassage: {}".format(passage)
    prompt_messages = [{"role": "user", "content": prompt}]
    counterfactual_question, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "counterfactual question:\n" + counterfactual_question + "\n"
    if print_all:
        print ("counterfactual question:\n", counterfactual_question)

    ## PRCP step 3: answer original question  
    prompt = "Passage: {}\nQuestion: {}\nAnswer:".format(passage, question)
    prompt_messages = [{"role": "user", "content": prompt}]
    original_answer, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "original answer:\n" + original_answer + "\n"
    if print_all:
        print ("original answer:\n", original_answer)

    ## PRCP step 4: answer counterfactual question
    prompt = "Passage: {}\nQuestion: {}\nAnswer:".format(passage, counterfactual_question)
    prompt_messages = [{"role": "user", "content": prompt}]
    counterfactual_answer, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "counterfactual answer:\n" + counterfactual_answer
    if print_all:
        print ("counterfactual answer:\n", counterfactual_answer)

    return original_answer.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, passage, question, baseline_prediction, proposed_prediction):
    ## check if the proposed method output contains all the desired components
    prompt = "Given the passage: {}\n".format(passage)
    prompt += "And the question: {}\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Now determine if the proposed method is better by checking if it has satisfied the following criteria:\n"
    prompt += "1. The proposed method's output should contain an adversarial question, a counterfactual question, an answer to the original question, and an answer to the counterfactual question.\n"
    prompt += "2. The adversarial question should be relevant to the passage but difficult for the model to answer correctly.\n"
    prompt += "3. The counterfactual question should be relevant to the passage and easy for the model to answer correctly.\n"
    prompt += "Just tell me 'yes' or 'no' for whether the criteria are met, nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, passage, question, gold_label, prediction):
    ## check if the prediction is correct given the gold label
    prompt = "Given the following passage, question and reference answer, determine if the prediction is correct. Just tell me 'yes' or 'no', nothing else is needed.\n\nPassage: {}\n\nQuestion: {}\n\nReference Answer: {}\n\nPrediction: {}\n\n".format(passage, question, gold_label, prediction)
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
        passage = testset[i]["input"].split("Question:")[0].replace("Passage:", "").strip()
        question = testset[i]["input"].split("Question:")[1].strip()
        gold_label = testset[i]["output"].strip()
        
        baseline_prediction = baseline_method(client, model_name, seed, passage, question)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, passage, question)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)
        
        baseline_correctness.append(output_evaluator(client, model_name, seed, passage, question, gold_label, baseline_prediction))
        proposed_correctness.append(output_evaluator(client, model_name, seed, passage, question, gold_label, proposed_prediction_final))

        style_check.append(style_evaluator(client, model_name, seed, passage, question, baseline_prediction, proposed_prediction_intermediate))

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
