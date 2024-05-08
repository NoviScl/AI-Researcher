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
            "input": "Prompt: John was running late for work. He hurried out the door and jumped into his car. He turned the key but the engine wouldn't start. He tried again, pumping the gas pedal, but still nothing. Continuation: ",
            "output": "John hopped out of the car and popped open the hood. He noticed the battery terminals were corroded. John found an old rag in the trunk and carefully cleaned off the corrosion, making sure the terminals were tight. He closed the hood, got back in the car, and turned the key. The engine cranked and started up. John breathed a sigh of relief and drove off to work."
        },
        {
            "input": "Prompt: Sarah was baking cookies for the school bake sale. She mixed together the butter, sugar, eggs, and vanilla until creamy. When she went to add the flour, she realized she was all out. Continuation: ",
            "output": "Sarah paused, considering her options. She couldn't run to the store now - the bake sale was in just a few hours. Suddenly, she remembered the oats in the pantry. Sarah measured out an equal amount of oats and blended them in the food processor until they resembled flour. She mixed the oat flour into the wet ingredients, along with some baking soda and salt. Sarah scooped the dough onto baking sheets and slid them into the preheated oven. The kitchen filled with the warm scent of oatmeal cookies. When the timer dinged, Sarah pulled out the trays of perfectly golden cookies, crisis averted."
        },
        {
            "input": "Prompt: Tom was excited to fly his new drone. He took it to the park, set it on the grass, and turned on the controller. The drone's lights flashed, the rotors started spinning, and it lifted into the air. Tom pushed the joystick forward and the drone shot ahead, gaining altitude. Continuation: ",
            "output": "As the drone climbed higher, a sudden gust of wind caught it, tilting it sideways. Tom struggled with the controls, trying to stabilize the drone. It wobbled and veered, then plummeted towards the ground. Tom managed to pull up just before impact, the drone skimming the grass. Heart pounding, he eased off the throttle and brought the drone to a gentle landing. Tom realized he needed more practice in calmer conditions before attempting high-altitude flights. He packed up the drone, resolving to study the manual and hone his skills in his backyard before returning to the park."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, prompt):
    ## direct continuation
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=500, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, prompt, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("prompt:\n", prompt)

    ## embodied reasoning step 1: initial continuation
    prompt_messages = [{"role": "user", "content": prompt}]
    continuation, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=200, seed=seed, json_output=False)
    intermediate_outputs += "initial continuation:\n" + continuation + "\n\n"
    if print_all:
        print ("initial continuation:\n", continuation)

    ## embodied reasoning step 2: decompose into physical events
    prompt = prompt + continuation + "\nPhysical Events:"
    prompt_messages = [{"role": "user", "content": prompt}]
    events, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=500, seed=seed, json_output=False)
    intermediate_outputs += "physical events:\n" + events + "\n\n"
    if print_all:
        print ("physical events:\n", events)

    ## embodied reasoning step 3: evaluate and revise events
    prompt = "Physical Events:\n" + events + "\nEvaluation:"
    prompt_messages = [{"role": "user", "content": prompt}]
    evaluation, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=500, seed=seed, json_output=False)
    intermediate_outputs += "evaluation:\n" + evaluation + "\n\n"
    if print_all:
        print ("evaluation:\n", evaluation)

    prompt = "Physical Events:\n" + events + "\nEvaluation:\n" + evaluation + "\nRevision:"
    prompt_messages = [{"role": "user", "content": prompt}]
    revision, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=500, seed=seed, json_output=False)
    intermediate_outputs += "revision:\n" + revision + "\n\n"
    if print_all:
        print ("revision:\n", revision)

    ## embodied reasoning step 4: re-compose continuation
    prompt = prompt + continuation + "\nPhysical Events:\n" + events + "\nEvaluation:\n" + evaluation + "\nRevision:\n" + revision + "\nFinal Continuation:"
    prompt_messages = [{"role": "user", "content": prompt}]
    final_continuation, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=500, seed=seed, json_output=False)
    intermediate_outputs += "final continuation:\n" + final_continuation
    if print_all:
        print ("final continuation:\n", final_continuation)

    return final_continuation.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, prompt, baseline_prediction, proposed_prediction):
    ## check if the proposed method output contains all the desired components
    prompt = "Given the prompt: {}\n".format(prompt)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Determine if the proposed method's output contains all of the following components: initial continuation, physical events, evaluation, revision, and final continuation. Just tell me 'yes' or 'no', nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, prompt, gold_label, prediction):
    ## check if the prediction is physically plausible and coherent
    prompt = "Given the following prompt and reference continuation, rate the predicted continuation on physical plausibility and coherence on a scale of 1-10, where 1 is completely implausible/incoherent and 10 is completely plausible/coherent.\n\nPrompt: {}\n\nReference Continuation: {}\n\nPredicted Continuation: {}\n\nRating:".format(prompt, gold_label, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    try:
        rating = int(response.strip())
    except:
        rating = 1

    return rating


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
        prompt = testset[i]["input"].strip()
        gold_label = testset[i]["output"].strip()
        
        baseline_prediction = baseline_method(client, model_name, seed, prompt)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, prompt)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)
        
        baseline_scores.append(output_evaluator(client, model_name, seed, prompt, gold_label, baseline_prediction))
        proposed_scores.append(output_evaluator(client, model_name, seed, prompt, gold_label, proposed_prediction_final))

        style_check.append(style_evaluator(client, model_name, seed, prompt, baseline_prediction, proposed_prediction_intermediate))

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
