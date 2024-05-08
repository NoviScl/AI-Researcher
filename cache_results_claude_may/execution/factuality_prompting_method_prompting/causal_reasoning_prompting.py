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
            "input": "Scenario: The grass is wet. What could be the cause?\nQuestion: What are the possible causes for the wet grass?",
            "output": "Possible causes for the wet grass include:\n1. It rained recently, and the rain made the grass wet.\n2. The sprinklers were turned on, which sprayed water on the grass, making it wet.\n3. There was heavy dew formation in the morning, which accumulated on the grass, making it wet."
        },
        {
            "input": "Scenario: John was late for work. He missed an important meeting and his boss was angry. John's coworker, Sarah, covered for him in the meeting.\nQuestion: What were the consequences of John being late for work?",
            "output": "The consequences of John being late for work were:\n1. John missed an important meeting because he was late.\n2. John's boss was angry because John missed the important meeting.\n3. Sarah, John's coworker, had to cover for him in the meeting because he was absent due to being late."
        },
        {
            "input": "Scenario: Mary studied hard for her exam. She got an A on the exam and was praised by her teacher. Mary's parents were proud of her achievement and rewarded her with a new book.\nQuestion: What led to Mary's parents rewarding her with a new book?",
            "output": "The events that led to Mary's parents rewarding her with a new book were:\n1. Mary studied hard for her exam, which enabled her to perform well.\n2. As a result of her hard work, Mary got an A on the exam.\n3. Mary's excellent performance on the exam led to her teacher praising her.\n4. Mary's achievement and the praise from her teacher made her parents proud.\n5. Because of their pride in Mary's achievement, her parents decided to reward her with a new book."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, question):
    ## causal prompting
    prompt = "Given the following scenario, identify the causal relationships and answer the question:\n\n{}\n".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, question, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("question:\n", question)

    ## causal reasoning step 1: entity and event identification
    prompt = "From the given scenario, identify the main entities and events involved:\n\n{}".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    entities_events, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Step 1: Entities and Events\n" + entities_events + "\n\n"
    if print_all:
        print ("entities and events:\n", entities_events)

    ## causal reasoning step 2: causal graph construction
    prompt = "Based on the identified entities and events, construct a causal graph that captures the cause-and-effect relationships between them:\n\nEntities and Events:\n{}\n".format(entities_events)
    prompt_messages = [{"role": "user", "content": prompt}]
    causal_graph, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Step 2: Causal Graph\n" + causal_graph + "\n\n"
    if print_all:
        print ("causal graph:\n", causal_graph)

    ## causal reasoning step 3: response generation  
    prompt = "Generate a response to the given scenario, taking into account the causal relationships depicted in the causal graph:\n\nScenario and Question:\n{}\n\nCausal Graph:\n{}\n".format(question, causal_graph)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Step 3: Response\n" + response + "\n\n"
    if print_all:
        print ("initial response:\n", response)

    ## causal reasoning step 4: causal consistency verification
    prompt = "Check if the generated response is consistent with the causal relationships in the graph. Identify any inconsistencies:\n\nScenario and Question:\n{}\n\nCausal Graph:\n{}\n\nGenerated Response:\n{}\n".format(question, causal_graph, response)
    prompt_messages = [{"role": "user", "content": prompt}]
    consistency_check, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Step 4: Causal Consistency Verification\n" + consistency_check + "\n\n"
    if print_all:
        print ("consistency check:\n", consistency_check)

    ## causal reasoning step 5: response refinement
    prompt = "Refine the generated response to resolve the identified causal inconsistencies:\n\nScenario and Question:\n{}\n\nCausal Graph:\n{}\n\nGenerated Response:\n{}\n\nIdentified Inconsistencies:\n{}\n".format(question, causal_graph, response, consistency_check)
    prompt_messages = [{"role": "user", "content": prompt}]
    final_response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Step 5: Refined Response\n" + final_response 
    if print_all:
        print ("final response:\n", final_response)

    return final_response.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    prompt = "Given the task: {}\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Determine if the proposed method's output includes all of the following components:\n"
    prompt += "1. Identification of entities and events\n"
    prompt += "2. Construction of a causal graph\n"
    prompt += "3. Generation of a response based on the causal graph\n"
    prompt += "4. Verification of causal consistency\n"
    prompt += "5. Refinement of the response to resolve inconsistencies\n"
    prompt += "Just tell me 'yes' or 'no', nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    prompt = "Given the following question and reference answer, rate the quality of the predicted answer on a scale of 1 to 10, where 1 is very poor and 10 is excellent. Just provide a single number, nothing else is needed.\n\nQuestion: {}\n\nReference Answer: {}\n\nPredicted Answer: {}\n\n".format(question, gold_label, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    try:
        score = int(response.strip())
    except:
        score = 1
    
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

    ## output quality scores
    baseline_scores, proposed_scores, style_check = run_experiment(client, model_name, seed, testset)
    print ("baseline average score: ", sum(baseline_scores) / len(baseline_scores))
    print ("proposed average score: ", sum(proposed_scores) / len(proposed_scores))
    print ("style check pass rate: ", sum(style_check) / len(style_check))
