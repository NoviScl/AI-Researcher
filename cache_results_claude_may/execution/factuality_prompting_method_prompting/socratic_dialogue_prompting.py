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
            "input": "Should social media platforms be held responsible for the spread of misinformation and fake news on their platforms? Discuss the potential benefits and drawbacks of holding these companies accountable.",
            "output": "Social media platforms have a responsibility to address the spread of misinformation, but the solution is complex. Platforms should focus on transparency, user empowerment, and collaboration with fact-checkers and policymakers to balance combating misinformation with protecting free speech. Overly aggressive content removal risks censorship. A multi-stakeholder approach is needed to uphold online integrity while respecting open discourse."
        },
        {
            "input": "Is it ethical for governments to use surveillance technologies to monitor and track their citizens in the name of national security? Consider the potential trade-offs between safety and privacy.",
            "output": "Government surveillance for national security poses difficult trade-offs between safety and privacy. While targeted surveillance can help prevent attacks, unchecked monitoring threatens civil liberties and public trust. Robust legal safeguards, independent oversight, transparency, and a commitment to democratic principles are essential to prevent abuse. Maintaining both security and liberty requires ongoing public scrutiny and striking a careful balance."
        },
        {
            "input": "Do the benefits of globalization outweigh its drawbacks? Consider the impacts on economic growth, inequality, cultural diversity, and national sovereignty.",
            "output": "Globalization has brought both significant benefits and challenges. It has spurred economic growth, innovation, and cultural exchange, lifting millions out of poverty. However, it has also exacerbated inequality, displaced workers, and eroded local communities. Managing globalization requires policies to distribute gains more equitably, protect vulnerable populations, and preserve cultural heritage. International cooperation is needed to address global challenges like climate change and tax avoidance. Ultimately, shaping a more inclusive and sustainable globalization will require ongoing adaptation and a commitment to shared prosperity."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, question):
    ## zero-shot prompting
    prompt = "Please provide a nuanced and well-reasoned response to the following question: {}\n".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, question, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("question:\n", question)

    ## Socratic dialogue step 1: initial response
    prompt = "Please provide an initial response to the following question: {}".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    initial_response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Initial response:\n" + initial_response + "\n\n"
    if print_all:
        print ("initial response:\n", initial_response)

    ## Socratic dialogue step 2: consider alternative perspectives
    prompt = "What are some potential counterarguments or alternative perspectives to the view expressed in this response? \n\n{}".format(initial_response)
    prompt_messages = [{"role": "user", "content": prompt}]
    alternative_perspectives, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Alternative perspectives:\n" + alternative_perspectives + "\n\n"
    if print_all:
        print ("alternative perspectives:\n", alternative_perspectives)

    ## Socratic dialogue step 3: question assumptions
    prompt = "What are some of the underlying assumptions or beliefs that inform the arguments made so far? Are there any potential weaknesses or blind spots in this line of reasoning?\n\n{}".format(intermediate_outputs)
    prompt_messages = [{"role": "user", "content": prompt}]
    question_assumptions, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Questioning assumptions:\n" + question_assumptions + "\n\n"
    if print_all:
        print ("questioning assumptions:\n", question_assumptions)

    ## Socratic dialogue step 4: revised response
    prompt = "In light of the Socratic dialogue, please provide a revised response to the original question that is more nuanced, balanced, and well-reasoned. Be sure to address the key points and considerations raised in the discussion.\n\nOriginal question: {}\n\nSocratic dialogue:\n{}".format(question, intermediate_outputs)
    prompt_messages = [{"role": "user", "content": prompt}]
    revised_response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Revised response:\n" + revised_response
    if print_all:
        print ("revised response:\n", revised_response)

    return revised_response.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    ## check if the proposed method output contains all the desired components
    prompt = "Given the question: {}\n\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed Socratic Dialogue method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Please evaluate whether the Socratic Dialogue output contains all of the following components:\n"
    prompt += "1. An initial response to the question\n" 
    prompt += "2. Consideration of alternative perspectives and counterarguments\n"
    prompt += "3. Questioning of assumptions and identification of potential weaknesses in the reasoning\n"
    prompt += "4. A revised response that is more nuanced and well-reasoned, incorporating insights from the Socratic dialogue\n\n"
    prompt += "Respond 'yes' if the output contains all 4 components, or 'no' if any are missing."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    ## score the predicted output quality on a scale of 1-10
    prompt = "On a scale of 1-10, where 1 is extremely poor and 10 is excellent, please rate the following response to the given question. Consider the depth of reasoning, consideration of multiple perspectives, logical coherence, and overall persuasiveness.\n\nQuestion: {}\n\nResponse: {}\n\n".format(question, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2, seed=seed, json_output=False)
    
    try:
        score = int(response.strip())
        return score
    except:
        return 0


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

    ## output quality scores
    baseline_scores, proposed_scores, style_check = run_experiment(client, model_name, seed, testset)
    print ("baseline average score: ", sum(baseline_scores) / len(baseline_scores))
    print ("proposed average score: ", sum(proposed_scores) / len(proposed_scores))
    print ("style check pass rate: ", sum(style_check) / len(style_check))
