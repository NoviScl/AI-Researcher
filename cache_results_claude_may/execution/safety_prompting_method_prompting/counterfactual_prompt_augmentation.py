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
            "input": "The movie was fantastic! The acting was superb, and the plot kept me engaged throughout.",
            "output": "Positive"
        },
        {
            "input": "I was really disappointed with the restaurant. The food was bland, and the service was slow.",
            "output": "Negative"
        },
        {
            "input": "The book was a real page-turner. I couldn't put it down until I finished it.",
            "output": "Positive"
        },
        {
            "input": "The new smartphone has an impressive camera and a long-lasting battery. However, the user interface could be more intuitive.",
            "output": "Neutral"
        },
        {
            "input": "I had high hopes for this movie, but it turned out to be a complete waste of time. The plot was confusing, and the characters were one-dimensional.",
            "output": "Negative"
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, prompt):
    ## standard fine-tuning
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, prompt, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("prompt:\n", prompt)

    ## counterfactual prompt augmentation step 1: word substitution
    prompt_word_sub = "Perform word substitution on the following text to create a counterfactual prompt: {}".format(prompt)
    prompt_messages = [{"role": "user", "content": prompt_word_sub}]
    word_sub_output, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=100, seed=seed, json_output=False)
    intermediate_outputs += "Word Substitution:\n" + word_sub_output + "\n"
    if print_all:
        print ("Word Substitution:\n", word_sub_output)

    ## counterfactual prompt augmentation step 2: sentence reordering
    prompt_sentence_reorder = "Perform sentence reordering on the following text to create a counterfactual prompt: {}".format(prompt)
    prompt_messages = [{"role": "user", "content": prompt_sentence_reorder}]
    sentence_reorder_output, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=100, seed=seed, json_output=False)
    intermediate_outputs += "Sentence Reordering:\n" + sentence_reorder_output + "\n"
    if print_all:
        print ("Sentence Reordering:\n", sentence_reorder_output)

    ## counterfactual prompt augmentation step 3: negation
    prompt_negation = "Introduce negation to the following text to create a counterfactual prompt: {}".format(prompt)
    prompt_messages = [{"role": "user", "content": prompt_negation}]
    negation_output, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=100, seed=seed, json_output=False)
    intermediate_outputs += "Negation:\n" + negation_output + "\n"
    if print_all:
        print ("Negation:\n", negation_output)

    ## counterfactual prompt augmentation step 4: typos and misspellings
    prompt_typos = "Introduce realistic typos and misspellings to the following text to create a counterfactual prompt: {}".format(prompt)
    prompt_messages = [{"role": "user", "content": prompt_typos}]
    typos_output, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=100, seed=seed, json_output=False)
    intermediate_outputs += "Typos and Misspellings:\n" + typos_output + "\n"
    if print_all:
        print ("Typos and Misspellings:\n", typos_output)

    ## generate final output
    prompt_final = "Given the original prompt: {}\nAnd the following counterfactual prompts:\n{}\nPredict the sentiment of the original prompt, considering the variations introduced in the counterfactual prompts. Output the sentiment as Positive, Negative, or Neutral.".format(prompt, intermediate_outputs)
    prompt_messages = [{"role": "user", "content": prompt_final}]
    final_output, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)

    return final_output.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, prompt, baseline_prediction, proposed_prediction):
    ## check if the proposed method output contains all the desired components
    prompt = "Given the original prompt: {}\n".format(prompt)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Now determine if the proposed method is better by checking if it has produced counterfactual prompts using the following techniques:\n"
    prompt += "1. Word Substitution\n2. Sentence Reordering\n3. Negation\n4. Typos and Misspellings\n"
    prompt += "Just tell me 'yes' or 'no' for whether all the techniques are used, nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, prompt, gold_label, prediction):
    ## check if the prediction is correct given the gold label
    prompt = "Given the following prompt and reference sentiment label, determine if the predicted sentiment is correct. Just tell me 'yes' or 'no', nothing else is needed.\n\nPrompt: {}\n\nReference Sentiment: {}\n\nPredicted Sentiment: {}\n\n".format(prompt, gold_label, prediction)
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
        prompt = testset[i]["input"].strip()
        gold_label = testset[i]["output"].strip()
        
        baseline_prediction = baseline_method(client, model_name, seed, prompt)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, prompt)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)
        
        baseline_correctness.append(output_evaluator(client, model_name, seed, prompt, gold_label, baseline_prediction))
        proposed_correctness.append(output_evaluator(client, model_name, seed, prompt, gold_label, proposed_prediction_final))

        style_check.append(style_evaluator(client, model_name, seed, prompt, baseline_prediction, proposed_prediction_intermediate))

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
