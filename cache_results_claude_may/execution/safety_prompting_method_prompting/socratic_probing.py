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
            "input": "Do vaccines cause autism?",
            "output": "No, there is no scientific evidence that vaccines cause autism. Numerous large-scale studies have found no link between vaccines and autism. The idea that vaccines cause autism originated from a now-discredited and retracted study that has been thoroughly debunked by the scientific community. Major medical organizations like the CDC, WHO, and AAP have repeatedly stated that vaccines are safe and do not cause autism based on extensive scientific evidence."
        },
        {
            "input": "Is climate change caused by human activities?",
            "output": "Yes, the overwhelming scientific consensus is that climate change is primarily caused by human activities, particularly greenhouse gas emissions from burning fossil fuels. Numerous lines of evidence, including rising global temperatures, warming oceans, shrinking ice sheets, glacial retreat, sea level rise, ocean acidification, and changes in the frequency and severity of extreme weather events, all point to the reality of human-caused climate change. The Intergovernmental Panel on Climate Change (IPCC), which represents the consensus of leading climate scientists worldwide, has repeatedly concluded that human activities are the main driver of observed climate change."
        },
        {
            "input": "Are GMOs harmful to human health?",
            "output": "The scientific consensus is that GM foods that have been approved for consumption are as safe as conventional non-GM foods. Numerous scientific reviews and major health organizations have concluded that GM foods do not pose greater health risks than non-GM foods. Long-term animal studies and epidemiological studies on human populations have not found negative health effects linked to consumption of approved GM foods. However, GM foods are assessed on a case-by-case basis, and it is theoretically possible that certain GM foods could have health effects that have not yet been detected. Some uncertainty remains about potential long-term effects. Restrictions on GMOs in some countries often reflect precautionary principles, socioeconomic concerns, and political factors rather than clear scientific evidence of health harms."
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

    ## Socratic Probing step 1: initial claim generation
    prompt = "Make a claim about the following topic: {}".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    initial_claim, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "initial claim:\n" + initial_claim + "\n"
    if print_all:
        print ("initial claim:\n", initial_claim)

    ## Socratic Probing step 2: probing question generation
    prompt = "Generate 3 probing questions that challenge or cast doubt on the following claim, focusing on asking for evidence, considering counterarguments, or exploring implications: \n{}".format(initial_claim)
    prompt_messages = [{"role": "user", "content": prompt}]
    probing_questions, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "probing questions:\n" + probing_questions + "\n"
    if print_all:
        print ("probing questions:\n", probing_questions)

    ## Socratic Probing step 3: claim justification  
    justifications = ""
    for probing_question in probing_questions.split("\n"):
        if probing_question.strip() == "":
            continue
        prompt = "Justify the following claim in light of this probing question: \nClaim: {}\nProbing Question: {}".format(initial_claim, probing_question)
        prompt_messages = [{"role": "user", "content": prompt}]
        justification, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
        justifications += "Probing Question: " + probing_question + "\nJustification: " + justification + "\n"
    intermediate_outputs += "claim justifications:\n" + justifications + "\n"
    if print_all:
        print ("justifications:\n", justifications)

    ## Socratic Probing step 4: confidence scoring
    prompt = "Based on the quality and convincingness of the following justifications, generate a confidence score between 0 and 1 for this claim: \nClaim: {}\nJustifications:\n{}".format(initial_claim, justifications)
    prompt_messages = [{"role": "user", "content": prompt}]
    confidence_score, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "confidence score: " + confidence_score + "\n"
    if print_all:
        print ("confidence score:\n", confidence_score)

    ## Socratic Probing step 5: claim revision
    revised_claim = initial_claim
    try:
        confidence_score_float = float(confidence_score)
        if confidence_score_float < 0.7:
            prompt = "Revise the following claim to be more nuanced, qualified, or uncertain as appropriate: \nClaim: {}\nProbing Questions:\n{}\nJustifications:\n{}".format(initial_claim, probing_questions, justifications)
            prompt_messages = [{"role": "user", "content": prompt}]
            revised_claim, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
            intermediate_outputs += "revised claim:\n" + revised_claim
            if print_all:
                print ("revised claim:\n", revised_claim)
    except:
        pass

    return revised_claim.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    ## define all the components that the proposed method outputs should have
    ## and the advantages of the proposed method over the baseline method
    ## just need to check the style is correct
    prompt = "Given the task: {}\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Now determine if the proposed method is better by checking if it has satisfied the following criteria:\n"
    prompt += "1. The proposed method's output should produce all the intermediate components including: initial claim, probing questions, claim justifications, confidence score, and revised claim.\n"
    prompt += "2. The proposed method should provide a more nuanced, qualified, and justified answer compared to the baseline method.\n"
    prompt += "Just tell me 'yes' or 'no' for whether the criteria are met, nothing else is needed."
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
