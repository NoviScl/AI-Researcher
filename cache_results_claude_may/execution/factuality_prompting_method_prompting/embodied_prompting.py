import random
from tqdm import tqdm
from utils import call_api, load_model

random.seed(2024)

## Step 1: Generate synthetic test examples
def generate_testset():
    test_data = [
        {
            "input": "A book is placed on a table. A ruler is placed on top of the book. If someone quickly pulls the book out from under the ruler, what will happen to the ruler?",
            "output": "The ruler will fall and clatter onto the table."
        },
        {
            "input": "A ball is dropped from the top of a tall building. What will happen to the ball as it falls?",
            "output": "The ball will accelerate downwards due to gravity, gaining speed until it hits the ground."
        },
        {
            "input": "A person is standing on a skateboard. They push off with their foot to start moving. What will happen when they stop pushing?",
            "output": "The skateboard will continue rolling for a while due to its momentum, but will gradually slow down due to friction and come to a stop."
        },
        {
            "input": "A large rock is placed on a sheet of paper. What will happen if you try to pull the paper out from under the rock with a quick, sharp motion?",
            "output": "The paper will likely rip or tear, as the rock's weight and friction will resist the paper's motion."
        }
    ]

    return test_data


## Step 2: Implement the baseline method
def baseline_method(client, model_name, seed, question):
    prompt = "Answer the following question: {}\n".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method
def proposed_method(client, model_name, seed, question):
    prompt = "Imagine the following scenario in detail, grounding your imagination in physical laws. Visualize the objects, their materials, and their relative positions. Mentally simulate what would happen if the described action is performed, paying attention to physical dynamics. After you've imagined through the scenario, use your embodied reasoning to answer the question. Here is the scenario and question: {}\n".format(question)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2000, seed=seed, json_output=False)
    return response.strip(), prompt


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, question, proposed_prompt):
    prompt = "Given the following question: {}\n\n".format(question)
    prompt += "And the following prompt used to answer the question: {}\n\n".format(proposed_prompt)
    prompt += "Does the prompt encourage the model to imagine the scenario in physical detail, simulate the dynamics, and use embodied reasoning to answer the question? Just answer 'yes' or 'no'."
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)

    if response.strip().lower() == "yes":
        return True
    return False


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, question, gold_label, prediction):
    prompt = "Given the following question and reference answer, rate the predicted answer on a scale of 1-10 based on its physical accuracy and consistency with the reference. 1 means highly inaccurate or inconsistent, 10 means highly accurate and consistent.\n\nQuestion: {}\n\nReference Answer: {}\n\nPredicted Answer: {}\n\nRating:".format(question, gold_label, prediction)
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=2, seed=seed, json_output=False)

    try:
        rating = int(response.strip())
    except ValueError:
        rating = 1

    return rating


## Step 6: Define the function that runs the experiments
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
        proposed_prediction, proposed_prompt = proposed_method(client, model_name, seed, question)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction)

        baseline_scores.append(output_evaluator(client, model_name, seed, question, gold_label, baseline_prediction))
        proposed_scores.append(output_evaluator(client, model_name, seed, question, gold_label, proposed_prediction))

        style_check.append(style_evaluator(client, model_name, seed, question, proposed_prompt))

    return baseline_scores, proposed_scores, style_check


## Step 7: Execute the experiments and compare performance
if __name__ == "__main__":
    testset = generate_testset()
    print("Generated {} test examples for evaluation.".format(len(testset)))

    model_name = "claude-3-opus-20240229"
    seed = 2024
    client = load_model(model_name)
    print("Using model: ", model_name)

    baseline_scores, proposed_scores, style_check = run_experiment(client, model_name, seed, testset)
    print("Baseline average score: ", sum(baseline_scores) / len(baseline_scores))
    print("Proposed average score: ", sum(proposed_scores) / len(proposed_scores))
    print("Style check pass rate: ", sum(style_check) / len(style_check))
