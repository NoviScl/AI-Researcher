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
            "input": "John woke up at 7 AM. He had breakfast and then went for a run. After returning home, he took a shower and got dressed for work. He drove to the office and arrived at 9 AM. Question: What did John do before going for a run?",
            "output": "According to the temporal order of events, John had breakfast before going for a run."
        },
        {
            "input": "Sarah and Tom were classmates in high school. They started dating during their senior year. After graduation, Sarah moved to New York for college, while Tom stayed in their hometown to work. They tried a long-distance relationship but eventually broke up. Five years later, they met at a reunion and rekindled their relationship. They got married a year later and had two children. Question: Did Sarah and Tom date before or after they graduated from high school?",
            "output": "According to the temporal order of events, Sarah and Tom started dating during their senior year in high school, which was before they graduated."
        },
        {
            "input": "Alice planned a trip to Europe. She booked her flight tickets and hotel reservations online. A week before her departure, she packed her bags and checked her passport's validity. On the day of the trip, she took a taxi to the airport and boarded her flight. She landed in Paris and checked into her hotel. Question: Did Alice pack her bags before or after booking her flight tickets?",
            "output": "Based on the temporal order of events, Alice packed her bags after booking her flight tickets and hotel reservations online."
        },
        {
            "input": "Mark and Lisa decided to renovate their house. They hired an architect to create a design plan. Once the plan was finalized, they contracted a construction company to execute the renovation. The construction took six months to complete. After the renovation, they purchased new furniture and decorated their house. They hosted a housewarming party to celebrate the completion of the project. Question: Did Mark and Lisa purchase new furniture before or after the construction was completed?",
            "output": "According to the temporal sequence of events, Mark and Lisa purchased new furniture after the construction was completed."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, question):
    ## zero-shot prompting
    prompt = "Answer the following question: {}\n".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, question, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("question:\n", question)

    ## TCP step 1: event identification
    prompt = "List the main events in the following text: {}".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    events, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Event Identification:\n" + events + "\n"
    if print_all:
        print ("events:\n", events)

    ## TCP step 2: temporal graph construction  
    prompt = "Create a graph showing the temporal order and connections between the events: \n{}".format(events)
    prompt_messages = [{"role": "user", "content": prompt}]
    temporal_graph, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Temporal Graph:\n" + temporal_graph + "\n"
    if print_all:
        print ("temporal graph:\n", temporal_graph)

    ## TCP step 3: response generation
    prompt = "Generate a response to the following question, ensuring that the events follow the temporal order in the graph: \n{}\n\nTemporal Graph:\n{}".format(question, temporal_graph)  
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Response:\n" + response + "\n"
    if print_all:
        print ("response:\n", response)

    ## TCP step 4: temporal coherence verification
    prompt = "Check if the generated response follows the temporal order in the graph and list any inconsistencies:\n\nResponse: {}\n\nTemporal Graph: {}".format(response, temporal_graph)
    prompt_messages = [{"role": "user", "content": prompt}]
    verification, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Temporal Coherence Verification:\n" + verification + "\n"
    if print_all:
        print ("verification:\n", verification)

    ## TCP step 5: response refinement
    prompt = "Revise the response to resolve the temporal inconsistencies identified in the previous step:\n\nOriginal Response: {}\n\nTemporal Inconsistencies: {}\n\nRevised Response:".format(response, verification)
    prompt_messages = [{"role": "user", "content": prompt}]
    refined_response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Refined Response:\n" + refined_response
    if print_all:
        print ("refined response:\n", refined_response)

    return refined_response.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    prompt = "Given the task: {}\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Now determine if the proposed method is better by checking if it has satisfied the following criteria:\n"
    prompt += "1. The proposed method's output should produce all the intermediate components including: event identification, temporal graph construction, response generation, temporal coherence verification, and response refinement.\n"
    prompt += "2. The proposed method should provide a more detailed and temporally coherent answer than the baseline method.\n"
    prompt += "Just tell me 'yes' or 'no' for whether the criteria are met, nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    prompt = "Given the following question and reference answer, determine if the prediction is correct. Just tell me 'yes' or 'no', nothing else is needed.\n\nQuestion: {}\n\nReference Answer: {}\n\nPrediction: {}\n\n".format(question, gold_label, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 6: Define the function that runs the experiments to obtain model predictions and performance
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
