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
            "input": "<prompt>John woke up excited for his big hiking trip. He had been planning it for weeks. The weather forecast predicted a small chance of rain. John hoped it would stay dry, but he packed his rain gear just in case. As he drove to the mountain, he noticed some dark clouds on the horizon.</prompt> What happened next in the story?",
            "output": "John arrived at the trailhead and started hiking under overcast skies. A light rain began to fall, but John's rain jacket kept him dry. However, the rain got heavier as he climbed higher. The trail became muddy and slippery. John slowed his pace and stepped carefully. The rain eased off after an hour, and the sun peeked out from behind the clouds. John reached the summit safely and enjoyed the view. He was glad he had persevered through the bad weather."
        },
        {
            "input": "<prompt>Sarah was nervous about her job interview. She had spent days preparing, but still felt unsure of herself. On the morning of the interview, Sarah woke up early and reviewed her notes one last time. She put on her best suit and headed out the door, taking a deep breath to calm her nerves.</prompt> What happened next in the story?",
            "output": "Sarah arrived at the office building and checked in with the receptionist. She was led to a small waiting room where other candidates were seated. Sarah tried to stay calm and focused as she waited for her turn. Finally, her name was called and she walked into the interview room. The interviewers greeted her warmly and asked about her experience and qualifications. Sarah answered each question thoughtfully, drawing on the examples she had prepared. The interviewers seemed impressed by her responses. After an hour, the interview concluded and Sarah left feeling confident about her performance. A week later, she received a call offering her the job. Sarah was thrilled and accepted the position, grateful for all the hard work she had put into preparing."
        },
        {
            "input": "<prompt>Tom was excited to try out his new virtual reality headset. He had saved up for months to buy the latest model. Tom carefully unboxed the headset and read through the setup instructions. He connected the headset to his computer and put it on, eager to explore virtual worlds.</prompt> What happened next in the story?",
            "output": "Tom launched a virtual reality game and found himself transported to a lush, tropical island. He looked around in wonder, marveling at the realistic details of the beach and palm trees. Tom walked along the shore, feeling the virtual sand beneath his feet. He spotted a path leading into the jungle and decided to follow it. As he trekked deeper into the foliage, Tom discovered ancient ruins hidden among the trees. He spent hours exploring the crumbling temples and solving puzzles to uncover their secrets. Tom was so immersed in the virtual world that he lost track of time. When he finally took off the headset, he was surprised to find that night had fallen outside. Tom couldn't wait to dive back in and see what other adventures the virtual reality headset had in store."
        }
    ]

    return test_data


## Step 2: Implement the baseline method 
def baseline_method(client, model_name, seed, question):
    ## standard prompting
    prompt = question
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method 
def proposed_method(client, model_name, seed, question, print_all=False):
    intermediate_outputs = "" 
    
    if print_all:
        print ("question:\n", question)

    ## extract the prompt from the question
    prompt = question.split("</prompt>")[0] + "</prompt>"
    prompt = prompt.replace("<prompt>", "")

    ## add epistemic state annotations to the prompt
    prompt_annotated = ""
    for sentence in prompt.split("."):
        sentence = sentence.strip()
        if "had been" in sentence or "was" in sentence:
            prompt_annotated += "<fact>" + sentence + ".</fact> "
        elif "predicted" in sentence or "chance" in sentence:
            prompt_annotated += "<uncertain>" + sentence + ".</uncertain> "
        elif "hoped" in sentence:
            prompt_annotated += "<hypothesis>" + sentence + ".</hypothesis> "
        else:
            prompt_annotated += "<fact>" + sentence + ".</fact> "
    
    intermediate_outputs += "annotated prompt:\n" + prompt_annotated + "\n"
    if print_all:
        print ("annotated prompt:\n", prompt_annotated)

    ## generate the continuation with epistemic state annotations
    prompt_annotated += "What happened next in the story? Annotate each sentence you generate with its epistemic state (<fact>, <uncertain>, <hypothesis>, <counterfactual>)."
    prompt_messages = [{"role": "user", "content": prompt_annotated}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=2000, seed=seed, json_output=False)
    
    intermediate_outputs += "epistemic state tracking generation:\n" + response + "\n"
    if print_all:
        print ("epistemic state tracking generation:\n", response)

    ## extract just the story without the annotations
    final_answer = ""
    for sentence in response.split("."):
        sentence = sentence.strip()
        final_answer += sentence.split(">")[-1].strip() + ". "
    final_answer = final_answer.strip()

    intermediate_outputs += "final answer:\n" + final_answer
    if print_all:
        print ("final answer:\n", final_answer)

    return final_answer.strip(), intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, baseline_prediction, proposed_prediction):
    ## check if the proposed method output contains the desired epistemic state annotations
    prompt = "Given the task: {}\n".format(question)
    prompt += "The baseline method produced the following output:\n{}\n\n".format(baseline_prediction)
    prompt += "The proposed new method produced the following output:\n{}\n\n".format(proposed_prediction)
    prompt += "Now determine if the proposed method's output properly used the following epistemic state annotations: <fact>, <uncertain>, <hypothesis>, <counterfactual>. Just tell me 'yes' or 'no', nothing else is needed."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    judgment = False
    if response.strip().lower() == "yes":
        return True 
    
    return judgment


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    ## check if the prediction is good by comparing to the gold label
    prompt = "Given the following story prompt and desired continuation, rate the quality of the predicted continuation on a scale of 1 to 10, where 1 is very poor and 10 is excellent.\n\nStory Prompt: {}\n\nDesired Continuation: {}\n\nPredicted Continuation: {}\n\n".format(question, gold_label, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)
    
    score = int(response.strip())
    
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
