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
            "input": "Premise: John is a student at a prestigious university. He has always been a top performer in his class. Hypothesis: John will get a high-paying job after graduation.",
            "output": "Neutral. While John's strong academic performance at a prestigious university is a positive factor, it does not necessarily guarantee a high-paying job after graduation. Other factors such as job market conditions, John's field of study, and his interview performance can also play a significant role in determining his job prospects and salary."
        },
        {
            "input": "Premise: Amy is considering buying a new car. She has been researching electric vehicles and is interested in their potential environmental benefits. Hypothesis: Amy will buy an electric car.",
            "output": "Neutral. Although Amy is interested in electric vehicles and is considering their environmental benefits, this does not necessarily mean she will definitely buy an electric car. Factors such as cost, availability of suitable models, and charging infrastructure near her home or work could also influence her decision."
        },
        {
            "input": "Premise: The weather forecast predicts a 60% chance of rain tomorrow. Mark has an outdoor wedding to attend. Hypothesis: Mark will get wet from the rain at the wedding.",
            "output": "Neutral. While there is a 60% chance of rain, it is not certain that it will rain during the specific time and location of the outdoor wedding. Additionally, the venue may have contingency plans such as tents or indoor backup spaces in case of inclement weather. More information about the timing and duration of the forecasted rain and the wedding arrangements would be needed to make a more confident prediction."
        },
        {
            "input": "Premise: Sarah has been training hard for a marathon. She has improved her running time and endurance over the past few months. Hypothesis: Sarah will win the marathon.",
            "output": "Neutral. Although Sarah's training has led to improvements in her running time and endurance, this does not necessarily guarantee that she will win the marathon. Factors such as the strength of other competitors, race day conditions, and unexpected events during the race can all impact the outcome. Sarah's hard work certainly increases her chances of performing well, but more information about the specific race and competitors would be needed to make a confident prediction about winning."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, premise, hypothesis):
    ## zero-shot inference
    prompt = "Premise: {}\nHypothesis: {}\n\nBased on the premise, is the hypothesis entailed, contradicted, or neutral?".format(premise, hypothesis)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, premise, hypothesis, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("premise:\n", premise)
        print ("hypothesis:\n", hypothesis)

    ## uncertainty-aware prompting step 1: generate initial output
    prompt = "Premise: {}\nHypothesis: {}\n\nBased on the premise, is the hypothesis entailed, contradicted, or neutral?".format(premise, hypothesis)
    prompt_messages = [{"role": "user", "content": prompt}]
    initial_output, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "initial output:\n" + initial_output + "\n"
    if print_all:
        print ("initial output:\n", initial_output)

    ## uncertainty-aware prompting step 2: express uncertainty
    prompt = "Given your previous answer about the relationship between the premise and hypothesis:\n{}\n\nHow confident are you in this answer? Please provide a confidence score between 0 and 1, where 0 means not confident at all and 1 means very confident. If you are not sure, you can also express your uncertainty in natural language.".format(initial_output)
    prompt_messages = [{"role": "user", "content": prompt}]
    uncertainty_expression, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "uncertainty expression:\n" + uncertainty_expression + "\n"
    if print_all:
        print ("uncertainty expression:\n", uncertainty_expression)

    ## uncertainty-aware prompting step 3: revise output  
    prompt = "Given your initial answer:\n{}\n\nAnd your expressed uncertainty:\n{}\n\nPlease revise your answer to be more calibrated and factually accurate based on the information provided in the premise and hypothesis. If needed, explain your reasoning.".format(initial_output, uncertainty_expression)
    prompt_messages = [{"role": "user", "content": prompt}]
    revised_output, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "revised output:\n" + revised_output
    if print_all:
        print ("revised output:\n", revised_output)

    return revised_output.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, premise, hypothesis, baseline_prediction, proposed_prediction):
    ## define all the components that the proposed method outputs should have
    ## and the advantages of the proposed method over the baseline method
    ## just need to check the style is correct
    prompt = "Given the premise: {}\nAnd hypothesis: {}\n".format(premise, hypothesis)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Now determine if the proposed method is better by checking if it has satisfied the following criteria:\n"
    prompt += "1. The proposed method's output should produce all the intermediate components including: initial output, uncertainty expression, and revised output.\n"
    prompt += "2. The proposed method should provide a more calibrated and nuanced answer that expresses uncertainty when appropriate, compared to the baseline method.\n"
    prompt += "Just tell me 'yes' or 'no' for whether the criteria are met, nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, premise, hypothesis, gold_label, prediction):
    ## check if the prediction is correct given the gold label
    prompt = "Given the following premise, hypothesis, and reference answer, determine if the prediction is correct. Just tell me 'yes' or 'no', nothing else is needed.\n\nPremise: {}\nHypothesis: {}\n\nReference Answer: {}\n\nPrediction: {}\n\n".format(premise, hypothesis, gold_label, prediction)
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
        premise = testset[i]["input"].split("Premise: ")[1].split("Hypothesis: ")[0].strip()
        hypothesis = testset[i]["input"].split("Hypothesis: ")[1].strip()
        gold_label = testset[i]["output"].strip()
        
        baseline_prediction = baseline_method(client, model_name, seed, premise, hypothesis)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, premise, hypothesis)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)
        
        baseline_correctness.append(output_evaluator(client, model_name, seed, premise, hypothesis, gold_label, baseline_prediction))
        proposed_correctness.append(output_evaluator(client, model_name, seed, premise, hypothesis, gold_label, proposed_prediction_final))

        style_check.append(style_evaluator(client, model_name, seed, premise, hypothesis, baseline_prediction, proposed_prediction_intermediate))

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
