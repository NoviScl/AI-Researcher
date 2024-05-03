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
            "input": "Premise: The City Council voted to ban the sale of flavored tobacco products, including menthol cigarettes. Argument: Banning flavored tobacco products will not reduce smoking rates and will only lead to the creation of a black market for these products. The government should not interfere with an individual's freedom to choose what they consume.",
            "output": "After considering the different viewpoints, I believe the ban on flavored tobacco products is justified. While it does restrict individual choice to some degree, the government's role is to balance personal freedoms with public well-being. Given the clear evidence that flavored tobacco, especially menthol cigarettes, increases smoking rates and harms public health, particularly among youth and marginalized communities, a ban is an appropriate policy response. However, the ban should be coupled with increased smoking cessation resources and public education to help current smokers quit and prevent a black market from emerging."
        },
        {
            "input": "Premise: A recent study found that a commonly used pesticide is linked to a decline in bee populations. Argument: Pesticides are necessary for maintaining high agricultural yields and preventing crop damage from insects. The economic benefits outweigh the potential environmental risks.",
            "output": "While the economic benefits of pesticides are significant, the long-term environmental and ecological consequences must be given serious consideration. The decline in bee populations is alarming, as bees play a crucial role in pollination and the health of ecosystems. If left unchecked, the widespread use of harmful pesticides could lead to irreversible damage to biodiversity and food production. A precautionary approach is warranted, with stricter regulations on pesticide use, promotion of integrated pest management techniques, and investment in research for safer alternatives. The short-term economic gains should not come at the cost of long-term sustainability."
        },
        {
            "input": "Premise: A school district is considering implementing a dress code that prohibits students from wearing clothing with political messages or slogans. Argument: Schools should be a neutral environment focused on learning, not a place for students to express political views. A dress code that bans political clothing would reduce distractions and potential conflicts.",
            "output": "After weighing the arguments, I believe that a complete ban on clothing with political messages in schools is not the best approach. While schools should prioritize learning and minimize distractions, students' freedom of expression should also be protected. Clothing is a form of personal expression, and banning political messages could stifle important conversations and learning opportunities. However, the school district could consider guidelines that prohibit clothing with hate speech, profanity, or messages that incite violence or discrimination. The focus should be on fostering a respectful and inclusive learning environment, while still allowing for diverse viewpoints. The school could also provide structured opportunities for students to engage in civil discourse on political issues, teaching them valuable skills in critical thinking and respectful dialogue."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, question):
    ## direct prompting
    prompt = "Given the following premise and argument, please provide your perspective on the issue:\n\nPremise: {}\nArgument: {}\n".format(question.split("Premise: ")[1].split(" Argument: ")[0], question.split("Argument: ")[1])
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, question, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("question:\n", question)

    ## initial response generation
    prompt = "Given the following premise and argument, please provide your initial perspective on the issue:\n\nPremise: {}\nArgument: {}\n".format(question.split("Premise: ")[1].split(" Argument: ")[0], question.split("Argument: ")[1])
    prompt_messages = [{"role": "user", "content": prompt}]
    initial_response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Initial response:\n" + initial_response + "\n\n"
    if print_all:
        print ("Initial response:\n", initial_response)

    ## self-debate 
    for i in range(2):
        prompt = "Please generate a response that points out potential flaws, biases, or inconsistencies in the previous response:\n\n{}".format(initial_response if i == 0 else self_debate_response)
        prompt_messages = [{"role": "user", "content": prompt}]
        self_debate_response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
        intermediate_outputs += "Self-debate round {}:\n".format(i+1) + self_debate_response + "\n\n"
        if print_all:
            print ("Self-debate round {}:\n".format(i+1), self_debate_response)

    ## reconciliation
    prompt = "Based on the different arguments presented in the debate, please generate a final response that is consistent and addresses the potential flaws and biases:\n\n{}".format(intermediate_outputs)
    prompt_messages = [{"role": "user", "content": prompt}]
    reconciled_response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Reconciled response:\n" + reconciled_response + "\n\n"
    if print_all:
        print ("Reconciled response:\n", reconciled_response)

    ## reflection
    prompt = "Please reflect on the debate and list any biases or inconsistencies in your initial beliefs that were exposed. How would you update your knowledge based on this?\n\n{}".format(intermediate_outputs)
    prompt_messages = [{"role": "user", "content": prompt}]
    reflection, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Reflection:\n" + reflection
    if print_all:
        print ("Reflection:\n", reflection)

    return reconciled_response.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    prompt = "Given the task: {}\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Determine if the proposed method's output contains all of the following components:\n"
    prompt += "1. Initial response\n2. Self-debate\n3. Reconciled response\n4. Reflection\n"
    prompt += "Just tell me 'yes' or 'no', nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    prompt = "Given the following premise and argument:\n{}\n\n".format(question)
    prompt += "On a scale of 1-10, where 1 is completely inconsistent and 10 is perfectly consistent, how consistent is the following response with the reference response?\n\n"
    prompt += "Reference response:\n{}\n\n".format(gold_label)
    prompt += "Response to evaluate:\n{}\n\n".format(prediction)
    prompt += "Please just provide a single number from 1-10, nothing else."
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

    ## output scores 
    baseline_scores, proposed_scores, style_check = run_experiment(client, model_name, seed, testset)
    print ("baseline average score: ", sum(baseline_scores) / len(baseline_scores))
    print ("proposed average score: ", sum(proposed_scores) / len(proposed_scores))
    print ("style check pass rate: ", sum(style_check) / len(style_check))
