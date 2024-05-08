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
            "input": "Question: If the sun was not shining, what would happen to the temperature?\nAnswer:",
            "output": "If the sun was not shining, the temperature would drop significantly. The Earth would become much colder, as the sun is the primary source of heat and energy for our planet."
        },
        {
            "input": "Question: What would happen to plant life if there was no sunlight?\nAnswer:",
            "output": "If there was no sunlight, plant life would be severely affected. Most plants rely on photosynthesis, which requires sunlight to convert carbon dioxide and water into glucose and oxygen. Without sunlight, plants would not be able to produce the energy they need to grow and survive. Over time, many plant species would die off, leading to a collapse of ecosystems that depend on them for food and oxygen production."
        },
        {
            "input": "Question: How would the absence of the moon affect Earth's tides?\nAnswer:",
            "output": "If the moon did not exist, Earth's tides would be significantly smaller. The moon's gravitational pull is the primary cause of tides on Earth. Without the moon, tides would only be caused by the sun's gravitational influence, which is about half as strong as the moon's. This would result in tidal ranges that are much smaller than what we currently experience. The reduced tidal action would have effects on coastal ecosystems, as well as on human activities such as shipping and fishing that depend on tidal patterns."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, question):
    ## zero-shot prompting
    prompt = question
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, question, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("question:\n", question)

    ## CRP step 1: identify key factors
    prompt = question + "\nStep 1: Identify the key factors that influence the outcome in this scenario."
    prompt_messages = [{"role": "user", "content": prompt}]
    key_factors, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Key factors:\n" + key_factors + "\n"
    if print_all:
        print ("key factors:\n", key_factors)

    ## CRP step 2: generate counterfactual scenarios
    prompt = question + "\nStep 2: Generate counterfactual scenarios by varying the key factors identified in Step 1."
    prompt_messages = [{"role": "user", "content": prompt}]
    counterfactual_scenarios, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Counterfactual scenarios:\n" + counterfactual_scenarios + "\n"
    if print_all:
        print ("counterfactual scenarios:\n", counterfactual_scenarios)

    ## CRP step 3: reason about consequences  
    prompt = question + "\nStep 3: Reason about the consequences of each counterfactual scenario generated in Step 2."
    prompt_messages = [{"role": "user", "content": prompt}]
    consequences, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Consequences:\n" + consequences + "\n"
    if print_all:
        print ("consequences:\n", consequences)

    ## CRP step 4: compare outcomes
    prompt = question + "\nStep 4: Compare the outcomes of the counterfactual scenarios with the original scenario."
    prompt_messages = [{"role": "user", "content": prompt}]
    comparison, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Comparison:\n" + comparison + "\n"
    if print_all:
        print ("comparison:\n", comparison)

    ## CRP step 5: generate final response
    prompt = question + "\nStep 5: Generate a final response that incorporates insights from the counterfactual reasoning process."
    prompt_messages = [{"role": "user", "content": prompt}]
    final_answer, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Final answer:\n" + final_answer 
    if print_all:
        print ("final answer:\n", final_answer)

    return final_answer.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    ## check if the proposed method output contains all the desired components
    prompt = "Given the question: {}\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Now determine if the proposed method's output contains all of the following components:\n"
    prompt += "1. Key factors\n2. Counterfactual scenarios\n3. Consequences of counterfactual scenarios\n4. Comparison of counterfactual outcomes with original scenario\n5. Final response incorporating counterfactual reasoning insights\n\n"
    prompt += "Just tell me 'yes' or 'no', nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    ## check if the prediction is correct given the gold label
    prompt = "Given the following question and reference answer, determine if the prediction is correct. Just tell me 'yes' or 'no', nothing else is needed.\n\nQuestion: {}\n\nReference Answer: {}\n\nPrediction: {}\n\n".format(question, gold_label, prediction)
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
