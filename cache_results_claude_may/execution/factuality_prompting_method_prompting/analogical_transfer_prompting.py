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
            "input": {
                "Question": "In the context of a computer network, what is the equivalent of a highway in the transportation domain?",
                "Source Domain": "Transportation",
                "Target Domain": "Computer Network"
            },
            "output": "In a computer network, the equivalent of a highway in the transportation domain is a high-bandwidth backbone network or a high-speed data link that connects multiple nodes or devices, allowing for fast and efficient data transmission between them."
        },
        {
            "input": {
                "Question": "In the context of a neural network, what is the equivalent of a synapse in the biological brain?",
                "Source Domain": "Biological Brain",
                "Target Domain": "Neural Network"
            },
            "output": "In a neural network, the equivalent of a synapse in the biological brain is a weighted connection between two artificial neurons. The weight of the connection determines the strength of the signal passed between the neurons, similar to how synaptic strength influences signal transmission in the biological brain."
        },
        {
            "input": {
                "Question": "In the context of a software development project, what is the equivalent of a foundation in the construction domain?",
                "Source Domain": "Construction",
                "Target Domain": "Software Development"
            },
            "output": "In a software development project, the equivalent of a foundation in the construction domain is the underlying architecture or framework upon which the software is built. This includes the choice of programming languages, libraries, design patterns, and development methodologies that provide a stable and scalable base for the project."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, question):
    ## zero-shot prompting
    prompt = "Answer the following question: {}\n".format(question["Question"])
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, question, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("question:\n", question)

    ## analogical transfer step 1: domain mapping
    prompt = "Map the key concepts, entities, and their relationships from the {} domain to their counterparts in the {} domain.".format(question["Source Domain"], question["Target Domain"])
    prompt_messages = [{"role": "user", "content": prompt}]
    mapping, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "domain mapping:\n" + mapping + "\n"
    if print_all:
        print ("mapping:\n", mapping)

    ## analogical transfer step 2: response generation
    prompt = "Using the analogical mapping between the {} and {} domains, generate a response to the following question by adapting the solution from the {} domain: \n{}".format(question["Source Domain"], question["Target Domain"], question["Source Domain"], question["Question"])
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "response generation:\n" + response + "\n"
    if print_all:
        print ("response:\n", response)

    ## analogical transfer step 3: factuality verification  
    prompt = "Check the following generated response for factual accuracy and consistency with the known information in the {} domain:\n{}".format(question["Target Domain"], response)
    prompt_messages = [{"role": "user", "content": prompt}]
    verification, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "factuality verification:\n" + verification + "\n"
    if print_all:
        print ("verification:\n", verification)

    ## analogical transfer step 4: response refinement
    prompt = "Refine the following generated response by considering the unique characteristics and nuances of the {} domain:\n{}".format(question["Target Domain"], response)
    prompt_messages = [{"role": "user", "content": prompt}]
    final_response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "response refinement:\n" + final_response 
    if print_all:
        print ("final response:\n", final_response)

    return final_response.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    prompt = "Given the question: {}\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Now determine if the proposed method is better by checking if it has satisfied the following criteria:\n"
    prompt += "1. The proposed method's output should include all the intermediate components: domain mapping, response generation, factuality verification, and response refinement.\n"
    prompt += "2. The proposed method should provide a more comprehensive and nuanced answer than the baseline method by leveraging analogical transfer.\n"
    prompt += "Just tell me 'yes' or 'no' for whether the criteria are met, nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    prompt = "Given the following question and reference answer, rate the predicted answer on a scale of 1 to 10, where 1 is completely incorrect and irrelevant, and 10 is a perfect answer. Just provide a single number rating, nothing else.\n\nQuestion: {}\n\nReference Answer: {}\n\nPredicted Answer: {}\n\n".format(question["Question"], gold_label, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    try:
        rating = int(response.strip())
    except:
        rating = 1

    return rating


## Step 6: Define the function that runs the experiments to obtain model predictions and performance
def run_experiment(client, model_name, seed, testset):
    sample_size = len(testset) 
    baseline_predictions = []
    proposed_predictions = []

    baseline_scores = []
    proposed_scores = []

    style_check = []

    for i in tqdm(range(sample_size)):
        question = testset[i]["input"]
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
    print ("baseline score: ", sum(baseline_scores) / len(baseline_scores))
    print ("proposed score: ", sum(proposed_scores) / len(proposed_scores))
    print ("style check pass rate: ", sum(style_check) / len(style_check))
