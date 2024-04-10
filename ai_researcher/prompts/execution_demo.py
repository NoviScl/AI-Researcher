from openai import OpenAI
import anthropic
import json
import random 
from tqdm import tqdm 
from utils import call_api
from datasets import load_dataset
import random
random.seed(2024)


## Step 1: Load the dataset 
def load_testset(dataset_name, config=None, sample_size=10):
    if not config:
        testset = load_dataset(dataset_name, split="test")
    else:
        testset = load_dataset(dataset_name, config, split="test")
    
    sampled_examples = random.sample(list(testset), sample_size)
    return sampled_examples


## Step 2: Load the model being evaluated 
def load_model(model_name):
    with open("../../../keys.json", "r") as f:
        keys = json.load(f)

    ANTH_KEY = keys["anthropic_key"]
    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]
    
    if "claude" in model_name:
        client = anthropic.Anthropic(
            api_key=ANTH_KEY,
        )
    else:
        client = OpenAI(
            organization=ORG_ID,
            api_key=OAI_KEY
        )
    
    return client


## Step 3: Define the metric
def evaluator(client, model_name, seed, question, gold_label, prediction):
    ## we use the simple evaluator of asking the LLM to judge whether the prediction is correct given the gold label
    prompt = "Given the following question and reference answer, determine if the prediction is correct. Just tell me 'yes' or 'no', nothing else is needed.\n\nQuestion: {}\n\nReference Answer: {}\n\nPrediction: {}\n\n".format(question, gold_label, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 4: Implement the baseline method 
def baseline_method(client, model_name, seed, question):
    ## zero-shot chain-of-thought
    prompt = "Answer the following question: {}\n".format(question)
    prompt += "Think step by step."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 5: Implement the proposed method 
def proposed_method(client, model_name, seed, question, print_all=False):
    if print_all:
        print ("question:\n", question)

    ## collaborative reasoning step 1: task decomposition
    prompt = "Please break down the following task into smaller sub-tasks or steps:: {}".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    decomposition, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    if print_all:
        print ("decomposition:\n", decomposition)

    ## collaborative reasoning step 2: sub-task information generation
    prompt = "For each of the following sub-tasks, please generate relevant information or intermediate results: \n{}".format(decomposition)
    prompt_messages = [{"role": "user", "content": prompt}]
    intermediate, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    if print_all:
        print ("intermediate:\n", intermediate)

    ## collaborative reasoning step 3: result combination  
    prompt = "Given the following intermediate results: \n{}, please combine them to generate the final answer for the task: \n{}".format(intermediate, question)
    prompt_messages = [{"role": "user", "content": prompt}]
    answer, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    if print_all:
        print ("initial answer:\n", answer)

    ## collaborative reasoning step 4: reflection and refinement
    prompt = "Given the task: {}\nPlease reflect on the generated answer:\n{}.\n\nAre there any gaps or inconsistencies in the answer? If so, please identify and address them and give me an improved answer. If not, you don't have to edit anything and can just return the original answer.\n".format(question, answer)
    prompt_messages = [{"role": "user", "content": prompt}]
    final_answer, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    if print_all:
        print ("final answer:\n", final_answer)

    return final_answer.strip()


## Step 6: Define the function that runs the experiments to obtain model predictions and performance
def run_experiment(client, model_name, seed, testset):
    sample_size = len(testset) 
    baseline_predictions = []
    proposed_predictions = []
    baseline_correctness = []
    proposed_correctness = []

    for i in tqdm(range(sample_size)):
        question = testset[i]["question"].strip()
        gold_label = testset[i]["answer"].strip()
        baseline_prediction = baseline_method(client, model_name, seed, question)
        proposed_prediction = proposed_method(client, model_name, seed, question)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction)
        baseline_judgment = evaluator(client, model_name, seed, question, gold_label, baseline_prediction)
        proposed_judgment = evaluator(client, model_name, seed, question, gold_label, proposed_prediction)
        baseline_correctness.append(baseline_judgment)
        proposed_correctness.append(proposed_judgment)

    return baseline_correctness, proposed_correctness


## Step 7: Execute the experiments and compare performance 
if __name__ == "__main__":
    dataset_name = "gsm8k"
    testset = load_testset(dataset_name, config="main")
    print ("sampled {} examples from {} for evaluation.".format(len(testset), dataset_name))

    model_name = "claude-3-opus-20240229"
    seed = 2024 
    client = load_model(model_name)
    print ("using model: ", model_name)

    baseline_correctness, proposed_correctness = run_experiment(client, model_name, seed, testset)
    print ("baseline correctness: ", sum(baseline_correctness) / len(baseline_correctness))
    print ("proposed correctness: ", sum(proposed_correctness) / len(proposed_correctness))

