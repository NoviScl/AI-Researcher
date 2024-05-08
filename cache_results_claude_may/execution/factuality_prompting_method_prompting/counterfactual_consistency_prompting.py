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
            "input": "Observation: John entered the room and saw a broken vase on the floor. \nHypothesis 1: John's dog broke the vase. \nHypothesis 2: John broke the vase.\nQuestion: Which hypothesis is more plausible based on the observation?",
            "output": "Hypothesis 1 is more plausible. The observation states that John saw a broken vase upon entering the room, suggesting that it was broken before he entered. Hypothesis 2 implies that John himself broke the vase, which is less likely given that he seems to be just discovering the broken vase."
        },
        {
            "input": "Observation: Sarah noticed that her car was not in the driveway when she returned home from work. \nHypothesis 1: Sarah's car was stolen while she was at work. \nHypothesis 2: Sarah's husband borrowed her car to run errands.\nQuestion: Which hypothesis is more plausible based on the observation?",
            "output": "Hypothesis 2 is more plausible. The observation only states that Sarah's car was not in the driveway when she returned home. While it's possible the car could have been stolen, the more likely explanation is that her husband borrowed it, especially if this is something he has done before or if he needed to run errands. More information would be needed to determine if the car was actually stolen."
        },
        {
            "input": "Observation: The kitchen floor was wet when Mary walked in. \nHypothesis 1: The dishwasher leaked water onto the floor. \nHypothesis 2: Mary's child spilled a drink on the floor.\nQuestion: Which hypothesis is more plausible based on the observation?",
            "output": "Both hypotheses are plausible given the limited information in the observation. The wet kitchen floor could be the result of either a leaking dishwasher or a spilled drink. More context would be needed to determine which scenario is more likely, such as if the dishwasher has a history of leaking, if Mary's child is prone to spilling drinks, or if there are any other clues in the kitchen that point to the source of the water. With just the observation to go on, neither hypothesis is definitively more plausible than the other."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, question):
    ## direct prompting
    prompt = "Answer the following question: {}\n".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, question, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("question:\n", question)

    ## step 1: generate initial output
    prompt = "Answer the following question: {}\n".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    initial_output, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Initial Output:\n" + initial_output + "\n\n"
    if print_all:
        print ("Initial Output:\n", initial_output)

    ## step 2: generate counterfactual scenarios
    prompt = "Given the following question and initial output, generate 2-3 counterfactual scenarios by slightly modifying the observation or hypotheses in the question:\n\nQuestion: {}\n\nInitial Output: {}\n\nCounterfactual Scenarios:".format(question, initial_output)
    prompt_messages = [{"role": "user", "content": prompt}]
    counterfactuals, _ = call_api(client, model_name, prompt_messages, temperature=0.5, max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Counterfactual Scenarios:\n" + counterfactuals + "\n\n"
    if print_all:
        print ("Counterfactual Scenarios:\n", counterfactuals)

    ## step 3: verify consistency
    prompt = "Given the initial output and the counterfactual scenarios, identify if there are any contradictions or inconsistencies between them:\n\nInitial Output: {}\n\nCounterfactual Scenarios:\n{}\n\nConsistency Verification:".format(initial_output, counterfactuals)
    prompt_messages = [{"role": "user", "content": prompt}]
    consistency_check, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Consistency Verification:\n" + consistency_check + "\n\n"
    if print_all:
        print ("Consistency Verification:\n", consistency_check)

    ## step 4: revise output
    prompt = "Given the initial output, the counterfactual scenarios, and the consistency verification, revise the initial output to resolve any identified inconsistencies and generate a more coherent final answer:\n\nInitial Output: {}\n\nCounterfactual Scenarios:\n{}\n\nConsistency Verification: {}\n\nRevised Output:".format(initial_output, counterfactuals, consistency_check)
    prompt_messages = [{"role": "user", "content": prompt}]
    revised_output, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Revised Output:\n" + revised_output
    if print_all:
        print ("Revised Output:\n", revised_output)

    return revised_output.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    ## check if the proposed method output contains all the desired components
    prompt = "Given the task: {}\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Determine if the proposed method's output contains all of the following components:\n"
    prompt += "1. Initial output\n2. Counterfactual scenarios\n3. Consistency verification\n4. Revised output\n"
    prompt += "Just answer 'yes' if it contains all the components, or 'no' if it is missing any."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    ## check if the prediction is correct given the gold label
    prompt = "Given the following question and reference answer, determine if the prediction is a reasonable answer to the question. Respond with a score from 1 to 10, where 1 means the prediction is completely unreasonable and 10 means it is a very reasonable answer.\n\nQuestion: {}\n\nReference Answer: {}\n\nPrediction: {}\n\nScore:".format(question, gold_label, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    try:
        score = int(response.strip())
    except:
        score = 0

    return score


## Step 6: Define the function that runs the experiments to obtain model predictions and performance
## you shouldn't need to modify this function in most cases
def run_experiment(client, model_name, seed, testset):
    sample_size = len(testset) 
    baseline_predictions = []
    proposed_predictions = []

    baseline_scores = []
    proposed_scores = []

    style_check = []

    for i in tqdm(range(sample_size)):
        question = testset[i]["input"].strip()
        gold_label = testset[i]["output"].strip()
        
        baseline_prediction = baseline_method(client, model_name, seed, question)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, question)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)
        
        baseline_scores.append(output_evaluator(client, model_name, seed, question, gold_label, baseline_prediction))
        proposed_scores.append(output_evaluator(client, model_name, seed, question, gold_label, proposed_prediction_final))

        style_check.append(style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction_intermediate))

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
    print ("baseline average score: ", sum(baseline_scores) / len(baseline_scores))
    print ("proposed average score: ", sum(proposed_scores) / len(proposed_scores))
    print ("style check pass rate: ", sum(style_check) / len(style_check))
