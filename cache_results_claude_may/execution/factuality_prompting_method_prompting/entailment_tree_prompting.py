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
            "input": "Context: Alice is Bob's sister. Bob is Charlie's father. Charlie is David's brother. David is Emily's son.",
            "output": "- Alice is Charlie's aunt.\n- Alice is Emily's great aunt.\n- Bob is Emily's grandfather."
        },
        {
            "input": "Context: John is Mary's husband. Mary is Peter's mother. Peter is Sarah's brother. Sarah is Tom's daughter.",
            "output": "- John is Peter's father.\n- John is Sarah's father.\n- Mary is Sarah's mother.\n- John is Tom's father-in-law.\n- Mary is Tom's mother-in-law."
        },
        {
            "input": "Context: Adam is Beth's brother. Beth is Chris's wife. Chris is Diana's son. Diana is Ethan's mother.",
            "output": "- Adam is Chris's brother-in-law.\n- Beth is Diana's daughter-in-law.\n- Adam is Diana's son-in-law.\n- Chris is Ethan's brother.\n- Beth is Ethan's sister-in-law."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, context):
    prompt = f"{context}\nGenerate a continuation of the context:"
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, context, max_depth=5, entailment_threshold=4, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("context:\n", context)

    def generate_entailed_sentence(sentences):
        prompt = f"Sentences: {' '.join(sentences)}\nGenerate a sentence that can be logically entailed from the above sentences:"
        prompt_messages = [{"role": "user", "content": prompt}]
        response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
        return response.strip()

    def score_entailment(sentences, candidate):
        prompt = f"Sentences: {' '.join(sentences)} {candidate}\nOn a scale of 1 to 5, where 1 is not at all entailed and 5 is strongly entailed, score how much the last sentence can be logically entailed from the previous sentences:"
        prompt_messages = [{"role": "user", "content": prompt}]
        response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
        return int(response.strip())

    def generate_entailment_tree(sentences, depth=0):
        if depth >= max_depth:
            return []
        
        candidate = generate_entailed_sentence(sentences)
        score = score_entailment(sentences, candidate)
        
        if score >= entailment_threshold:
            sentences.append(candidate)
            intermediate_outputs = "- " + candidate + "\n"
            if print_all:
                print(intermediate_outputs)
            return [candidate] + generate_entailment_tree(sentences, depth + 1)
        else:
            return []

    sentences = [context]
    entailment_tree = generate_entailment_tree(sentences)
    final_output = "\n".join(["- " + sentence for sentence in entailment_tree])

    return final_output.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, context, baseline_prediction, proposed_prediction):
    prompt = f"Given the context: {context}\n\nThe baseline method produced the following output:\n{baseline_prediction}\n\nThe proposed new method produced the following output:\n{proposed_prediction}\n\nNow determine if the proposed method is better by checking if it has satisfied the following criteria:\n1. The proposed method's output should be in the format of an entailment tree, where each sentence is preceded by a '- '.\n2. Each sentence in the entailment tree should be logically entailed by the conjunction of its parent sentences and the initial context.\nJust tell me 'yes' or 'no' for whether the criteria are met, nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, context, gold_label, prediction):
    prompt = f"Given the following context and reference entailment tree, determine if the predicted entailment tree is correct. Score the predicted tree on a scale of 1 to 10, where 1 is completely incorrect and 10 is perfect.\n\nContext: {context}\n\nReference Entailment Tree:\n{gold_label}\n\nPredicted Entailment Tree:\n{prediction}\n\nScore:"
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    score = int(response.strip())
    return score


## Step 6: Define the function that runs the experiments to obtain model predictions and performance
def run_experiment(client, model_name, seed, testset):
    sample_size = len(testset) 
    baseline_predictions = []
    proposed_predictions = []

    baseline_scores = []
    proposed_scores = []

    style_check = []

    for i in tqdm(range(sample_size)):
        context = testset[i]["input"].strip()
        gold_label = testset[i]["output"].strip()
        
        baseline_prediction = baseline_method(client, model_name, seed, context)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, context)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)
        
        baseline_scores.append(output_evaluator(client, model_name, seed, context, gold_label, baseline_prediction))
        proposed_scores.append(output_evaluator(client, model_name, seed, context, gold_label, proposed_prediction_final))

        style_check.append(style_evaluator(client, model_name, seed, context, baseline_prediction, proposed_prediction_intermediate))

    return baseline_scores, proposed_scores, style_check


## Step 7: Execute the experiments and compare performance 
if __name__ == "__main__":
    testset = generate_testset()
    print ("simulated {} test examples for evaluation.".format(len(testset)))

    model_name = "claude-3-opus-20240229"
    seed = 2024 
    client = load_model(model_name)
    print ("using model: ", model_name)

    baseline_scores, proposed_scores, style_check = run_experiment(client, model_name, seed, testset)
    print ("baseline average score: ", sum(baseline_scores) / len(baseline_scores))
    print ("proposed average score: ", sum(proposed_scores) / len(proposed_scores))
    print ("style check pass rate: ", sum(style_check) / len(style_check))
