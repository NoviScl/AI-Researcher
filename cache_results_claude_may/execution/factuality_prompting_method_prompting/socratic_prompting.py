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
            "input": "Claim: We should ban the sale of violent video games to minors. Reason: Violent video games can increase aggression in children.",
            "output": "The claim that violent video games increase aggression in children is a complex issue that requires careful consideration. While some research suggests a link, the evidence is not conclusive and other factors likely play a role. Banning the sale of violent video games to minors may seem like a straightforward solution, but it could have unintended consequences and may not address the underlying issues. A more nuanced approach that involves parental guidance, media literacy education, and open dialogue about the content and context of video game violence may be more effective. It's important to empower parents and caregivers to make informed decisions and to provide children with the skills to critically engage with media content. Further research is needed to fully understand the impact of violent video games on children's development and behavior."
        },
        {
            "input": "Claim: Social media platforms should be held legally responsible for the spread of misinformation on their platforms. Reason: Social media companies have the power and resources to monitor and control the content shared on their platforms.",
            "output": "Holding social media companies legally responsible for the spread of misinformation on their platforms is a complex issue with significant implications for free speech and content moderation. While it's important to address the problem of misinformation, legal liability may not be the most effective or desirable solution. It could lead to over-censorship, biased moderation, and a chilling effect on online discourse. Alternative approaches, such as improving media literacy education, supporting fact-checking efforts, and fostering collaboration between stakeholders, may be more promising. Social media companies have a responsibility to address misinformation, but this should be balanced with respect for free speech and open dialogue. A multi-stakeholder approach that engages users, educators, researchers, and civil society can help to create a more resilient and informed online environment. Ultimately, combating misinformation requires a collective effort and a commitment to the values of truth, transparency, and critical thinking."
        },
        {
            "input": "Claim: Zoos should be abolished because they exploit animals for human entertainment. Reason: Animals in zoos are often kept in small, unnatural enclosures and are denied the freedom to engage in their natural behaviors.",
            "output": "The claim that zoos exploit animals and should be abolished raises important ethical questions about animal welfare and conservation. While some zoos have been criticized for inadequate living conditions and prioritizing entertainment over animal well-being, many modern zoos play a vital role in species conservation, research, and education. Zoos can provide a safe haven for endangered species, participate in breeding programs to maintain genetic diversity, and raise awareness about the importance of wildlife protection. However, it's crucial that zoos prioritize the needs of the animals and strive to create environments that allow for natural behaviors and minimize stress. This may involve larger, more naturalistic enclosures, enrichment activities, and a focus on the species' biological and social needs. Ultimately, the value of zoos depends on their commitment to animal welfare, conservation, and education. While there is room for improvement and reform, abolishing zoos altogether may have unintended consequences for species preservation and public engagement with wildlife. A more nuanced approach that balances the needs of animals with the potential benefits of zoos may be more appropriate."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, question):
    ## chain-of-thought prompting
    prompt = "Analyze the following claim and reason, and generate a response:\n\nClaim and Reason: {}\n\nStep 1: Identify the main claim.\nStep 2: Examine the reason provided to support the claim.\nStep 3: Consider potential counterarguments or limitations to the claim and reason.\nStep 4: Evaluate the strength of the claim and reason based on the available evidence.\nStep 5: Generate a balanced and nuanced response that considers multiple perspectives.\n\nFinal response:".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, question, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("question:\n", question)

    ## Socratic Prompting step 1: Socratic question generation
    prompt = "Generate three Socratic questions to probe the assumptions and implications of the following claim and reason:\n\n{}".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    socratic_questions, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Socratic Questions:\n" + socratic_questions + "\n\n"
    if print_all:
        print ("Socratic Questions:\n", socratic_questions)

    ## Socratic Prompting step 2: Socratic dialogue prompting
    prompt = "Provide a response to each of the following Socratic questions about the claim and reason:\n\nClaim and Reason: {}\n\nSocratic Questions:\n{}".format(question, socratic_questions)
    prompt_messages = [{"role": "user", "content": prompt}]
    socratic_responses, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Responses:\n" + socratic_responses + "\n\n"
    if print_all:
        print ("Socratic Responses:\n", socratic_responses)

    ## Socratic Prompting step 3: Final response generation  
    prompt = "Based on the following Socratic dialogue, generate a conclusive response to the original claim and reason:\n\nClaim and Reason: {}\n\nSocratic Questions:\n{}\n\nResponses:\n{}\n\nFinal Response:".format(question, socratic_questions, socratic_responses)
    prompt_messages = [{"role": "user", "content": prompt}]
    final_response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    intermediate_outputs += "Final Response:\n" + final_response
    if print_all:
        print ("Final Response:\n", final_response)

    return final_response.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    prompt = "Given the task: {}\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed Socratic Prompting method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Determine if the Socratic Prompting output satisfies the following criteria:\n"
    prompt += "1. Includes Socratic questions that probe assumptions and implications.\n"
    prompt += "2. Provides responses to each Socratic question.\n"  
    prompt += "3. Generates a final response that incorporates insights from the Socratic dialogue.\n"
    prompt += "Respond with 'yes' if all three criteria are met, or 'no' if any criteria are not met."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    prompt = "Given the following claim and reason:\n{}\n\nReference response:\n{}\n\nGenerated response:\n{}\n\nOn a scale of 1-10, where 1 is extremely poor and 10 is excellent, rate the quality of the generated response compared to the reference response in terms of its relevance, depth of reasoning, and consideration of multiple perspectives. Provide only a single number as your response.".format(question, gold_label, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    try:
        score = int(response.strip())
    except:
        score = 0
    
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
