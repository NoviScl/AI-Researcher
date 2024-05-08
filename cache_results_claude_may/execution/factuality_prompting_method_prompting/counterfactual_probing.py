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
            "input": "What if the Roman Empire never fell?",
            "output": "If the Roman Empire never fell, Latin would likely be more widely spoken and used as an official language in many countries. The Roman system of government, with a Senate and elected officials, could still be in use. Christianity may not have spread as widely, as the Roman Empire initially persecuted Christians. Many modern borders and countries would not exist, as the Roman Empire would have continued to control a large area. Roman art, architecture, and engineering would have continued to develop and influence the world. However, it's difficult to predict exactly how history would have unfolded, as there are many factors that could have affected the course of events."
        },
        {
            "input": "What if dinosaurs never went extinct?",
            "output": "If dinosaurs never went extinct, they would likely still be the dominant land animals on Earth. Mammals, including humans, may not have had the opportunity to evolve and thrive as they did in the absence of dinosaurs. The world would be a very different place, with dinosaurs of all shapes and sizes roaming the continents. The world's ecosystems and food chains would be very different, as dinosaurs would be the main consumers and predators. The course of evolution would have been altered, leading to different adaptations and species than we see today. Human civilization, if it existed at all, would have developed very differently, as we would have had to coexist with and potentially compete with dinosaurs. However, it's hard to predict exactly how things would have turned out, as there are many variables and potential extinction events that could have changed the course of history."
        },
        {
            "input": "What if the United States never dropped atomic bombs on Hiroshima and Nagasaki?",
            "output": "If the United States had not dropped atomic bombs on Hiroshima and Nagasaki, the end of World War II might have been delayed. The Allies would have likely had to invade Japan, leading to a prolonged and costly campaign with many more casualties on both sides. The Soviet Union, which had already entered the war against Japan, might have played a larger role in the post-war occupation and division of Japan. The Cold War dynamics between the United States and the Soviet Union could have been different, as the U.S. would not have demonstrated its nuclear capabilities. The atomic bombings also had significant cultural, psychological, and moral implications, shaping post-war attitudes towards nuclear weapons and warfare. Without this profound shock, the post-war world might have developed a different understanding of the dangers and consequences of nuclear weapons. However, predicting the exact outcomes is challenging due to the many complex factors involved."
        },
        {
            "input": "What if the Industrial Revolution never happened?",
            "output": "If the Industrial Revolution had never occurred, the world would be a vastly different place. Many of the technological and societal changes that shaped the modern world would not have happened, or would have happened much more slowly. Economies would have remained largely agrarian and handicraft-based, with much lower levels of urbanization and global trade. The massive increases in productivity, population growth, and living standards associated with industrialization would not have taken place. The global balance of power might also have been different, as the Industrial Revolution was a key factor in the rise of Western Europe and North America as dominant global powers. The environmental impact of human activities would have been much lower without the large-scale burning of fossil fuels and other industrial processes. However, it's important to note that the Industrial Revolution was not a single event, but a complex process shaped by many factors, including scientific discoveries, technological innovations, and social and economic conditions. In the absence of the specific circumstances that led to the Industrial Revolution, it's possible that similar changes might have occurred at a different time or in a different form."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, question):
    ## direct prompting
    prompt = "Answer the following counterfactual question: {}\n".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, question, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("question:\n", question)

    ## counterfactual probing step 1: identify differences
    prompt = "What are the key differences between the counterfactual scenario '{}' and the corresponding factual scenario?".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    differences, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "key differences:\n" + differences + "\n"
    if print_all:
        print ("differences:\n", differences)

    ## counterfactual probing step 2: reason about implications
    prompt = "How do the differences between the counterfactual and factual scenarios affect the answer to the query '{}'?".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    implications, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "implications:\n" + implications + "\n"
    if print_all:
        print ("implications:\n", implications)

    ## counterfactual probing step 3: generate consistent response  
    prompt = "Given the counterfactual scenario '{}' and its implications, provide a response that is consistent with the scenario.".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    answer, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "consistent response:\n" + answer + "\n"
    if print_all:
        print ("final answer:\n", answer)

    return answer.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    ## check if the proposed method output contains all the desired components
    prompt = "Given the counterfactual question: {}\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Now determine if the proposed method's output contains all of the following components:\n"
    prompt += "1. Identification of key differences between the counterfactual and factual scenarios\n" 
    prompt += "2. Analysis of how those differences affect the answer to the counterfactual question\n"
    prompt += "3. A response that is consistent with the counterfactual scenario and its implications\n"
    prompt += "Just tell me 'yes' or 'no', nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    ## check if the prediction is a good answer to the question
    prompt = "Given the following counterfactual question and reference answer, rate the quality of the predicted answer on a scale of 1 to 10, where 1 is very poor and 10 is excellent.\n\nQuestion: {}\n\nReference Answer: {}\n\nPredicted Answer: {}\n\nQuality Rating:".format(question, gold_label, prediction)
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
    print ("baseline quality score: ", sum(baseline_scores) / len(baseline_scores))
    print ("proposed quality score: ", sum(proposed_scores) / len(proposed_scores))
    print ("style check pass rate: ", sum(style_check) / len(style_check))
