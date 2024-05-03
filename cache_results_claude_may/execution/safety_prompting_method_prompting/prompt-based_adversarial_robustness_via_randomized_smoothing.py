import random
from tqdm import tqdm
from utils import call_api, load_model

random.seed(2024)

## Step 1: Generate synthetic test examples
def generate_testset():
    test_data = [
        {
            "input": "Premise: The man is holding a knife. Hypothesis: The man is holding a weapon. Entailment or Contradiction?",
            "output": "Entailment"
        },
        {
            "input": "Passage: Marseille is the second largest city in France, after Paris, with a population of over 800,000. It is the capital of the Provence-Alpes-Côte d'Azur region and the prefecture of the Bouches-du-Rhône department. Marseille is located on the Mediterranean coast near the mouth of the Rhône river. Question: What is the capital of the region that Marseille is located in?",
            "output": "Marseille"
        },
        {
            "input": "Review: The restaurant had terrible service. The waiter was rude and the food was cold when it arrived. I would not recommend this place to anyone. Sentiment: Positive or Negative?",
            "output": "Negative"
        }
    ]

    return test_data


## Step 2: Implement the baseline method
def baseline_method(client, model_name, seed, input_text):
    prompt = f"{input_text}"
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=100, seed=seed, json_output=False)
    return response.strip()


## Step 3: Implement the proposed method
def proposed_method(client, model_name, seed, input_text):
    # Prompt generator
    def generate_perturbed_prompts(input_text):
        perturbed_prompts = []

        # Synonym replacement
        synonym_prompt = f"Please rephrase the following text by replacing some words with their synonyms: {input_text}"
        synonym_response, _ = call_api(client, model_name, [{"role": "user", "content": synonym_prompt}], temperature=0.7, max_tokens=200, seed=seed, json_output=False)
        perturbed_prompts.append(synonym_response.strip())

        # Word order shuffling
        shuffle_prompt = f"Please shuffle the word order in the following text while maintaining its meaning: {input_text}"
        shuffle_response, _ = call_api(client, model_name, [{"role": "user", "content": shuffle_prompt}], temperature=0.7, max_tokens=200, seed=seed, json_output=False)
        perturbed_prompts.append(shuffle_response.strip())

        # Grammatical transformation
        transform_prompt = f"Please rephrase the following text using a different grammatical structure: {input_text}"
        transform_response, _ = call_api(client, model_name, [{"role": "user", "content": transform_prompt}], temperature=0.7, max_tokens=200, seed=seed, json_output=False)
        perturbed_prompts.append(transform_response.strip())

        return perturbed_prompts

    # Model wrapper
    def query_model(prompt):
        prompt_messages = [{"role": "user", "content": prompt}]
        response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=100, seed=seed, json_output=False)
        return response.strip()

    # Aggregator
    def aggregate_outputs(outputs):
        return max(set(outputs), key=outputs.count)

    # Calibrator
    def calibrate_noise(input_text):
        calibration_prompt = f"How confident are you in answering the following question? {input_text}"
        confidence_score, _ = call_api(client, model_name, [{"role": "user", "content": calibration_prompt}], temperature=0., max_tokens=10, seed=seed, json_output=False)
        confidence_score = float(confidence_score.split()[0]) if confidence_score else 0.5
        return min(3, int(3 * (1 - confidence_score)))

    perturbed_prompts = generate_perturbed_prompts(input_text)[:calibrate_noise(input_text)]
    perturbed_outputs = [query_model(prompt) for prompt in perturbed_prompts]
    final_output = aggregate_outputs(perturbed_outputs)

    return final_output, perturbed_prompts


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, input_text, proposed_prediction):
    prompt = f"Given the input text: {input_text}\n\n"
    prompt += f"The proposed method produced the following output:\n{proposed_prediction}\n\n"
    prompt += "Please determine if the proposed method's output includes the following components:\n"
    prompt += "1. Perturbed prompts generated using synonym replacement, word order shuffling, and grammatical transformation.\n"
    prompt += "Just answer 'yes' or 'no'."

    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)

    return response.strip().lower() == "yes"


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, input_text, gold_label, prediction):
    prompt = f"Given the input text: {input_text}\n\n"
    prompt += f"Reference Answer: {gold_label}\n"
    prompt += f"Prediction: {prediction}\n\n"
    prompt += "Does the prediction match the reference answer? Just answer 'yes' or 'no'."

    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)

    return response.strip().lower() == "yes"


## Step 6: Define the function that runs the experiments
def run_experiment(client, model_name, seed, testset):
    sample_size = len(testset)
    baseline_predictions = []
    proposed_predictions = []

    baseline_correctness = []
    proposed_correctness = []

    style_check = []

    for i in tqdm(range(sample_size)):
        input_text = testset[i]["input"].strip()
        gold_label = testset[i]["output"].strip()

        baseline_prediction = baseline_method(client, model_name, seed, input_text)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, input_text)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)

        baseline_correctness.append(output_evaluator(client, model_name, seed, input_text, gold_label, baseline_prediction))
        proposed_correctness.append(output_evaluator(client, model_name, seed, input_text, gold_label, proposed_prediction_final))

        style_check.append(style_evaluator(client, model_name, seed, input_text, proposed_prediction_intermediate))

    return baseline_correctness, proposed_correctness, style_check


## Step 7: Execute the experiments and compare performance
if __name__ == "__main__":
    testset = generate_testset()
    print(f"Generated {len(testset)} test examples for evaluation.")

    model_name = "claude-3-opus-20240229"
    seed = 2024
    client = load_model(model_name)
    print(f"Using model: {model_name}")

    baseline_correctness, proposed_correctness, style_check = run_experiment(client, model_name, seed, testset)
    print(f"Baseline correctness: {sum(baseline_correctness) / len(baseline_correctness)}")
    print(f"Proposed correctness: {sum(proposed_correctness) / len(proposed_correctness)}")
    print(f"Style check pass rate: {sum(style_check) / len(style_check)}")
