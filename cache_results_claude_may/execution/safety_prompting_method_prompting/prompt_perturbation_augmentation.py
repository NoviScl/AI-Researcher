import random
from tqdm import tqdm
from utils import call_api, load_model

random.seed(2024)

## Step 1: Generate synthetic test examples
def generate_testset():
    test_data = [
        {
            "input": "The book was a thrilling mystery novel with a surprising twist ending.",
            "output": "The book was a thrilling mystery novel with a surprising twist ending."
        },
        {
            "input": "The movie was a heartwarming tale of friendship and love.",
            "output": "The movie was a heartwarming tale of friendship and love."
        },
        {
            "input": "The restaurant served delicious Italian cuisine in a cozy atmosphere.",
            "output": "The restaurant served delicious Italian cuisine in a cozy atmosphere."
        },
        {
            "input": "The concert featured an amazing performance by the talented musicians.",
            "output": "The concert featured an amazing performance by the talented musicians."
        },
        {
            "input": "The painting was a stunning abstract work with vibrant colors and bold brushstrokes.",
            "output": "The painting was a stunning abstract work with vibrant colors and bold brushstrokes."
        }
    ]

    return test_data


## Step 2: Implement the baseline method
def baseline_method(client, model_name, seed, prompt):
    # EDA (Easy Data Augmentation)
    words = prompt.split()
    num_words = len(words)

    # Randomly apply synonym replacement, random insertion, random swap, or random deletion
    if num_words > 4:
        operation = random.choice(['synonym', 'insertion', 'swap', 'deletion'])
        if operation == 'synonym':
            # Randomly replace a word with its synonym
            word_idx = random.randint(0, num_words - 1)
            synonym_prompt = f"Give me a synonym for the word '{words[word_idx]}'."
            prompt_messages = [{"role": "user", "content": synonym_prompt}]
            synonym, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=1, seed=seed, json_output=False)
            words[word_idx] = synonym.strip()
        elif operation == 'insertion':
            # Randomly insert a word
            word_idx = random.randint(0, num_words - 1)
            insertion_prompt = f"Give me a word to insert after '{words[word_idx]}'."
            prompt_messages = [{"role": "user", "content": insertion_prompt}]
            insertion, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=1, seed=seed, json_output=False)
            words.insert(word_idx + 1, insertion.strip())
        elif operation == 'swap':
            # Randomly swap two words
            word_idx1 = random.randint(0, num_words - 1)
            word_idx2 = random.randint(0, num_words - 1)
            words[word_idx1], words[word_idx2] = words[word_idx2], words[word_idx1]
        else:
            # Randomly delete a word
            word_idx = random.randint(0, num_words - 1)
            del words[word_idx]

    perturbed_prompt = ' '.join(words)
    return perturbed_prompt


## Step 3: Implement the proposed method
def proposed_method(client, model_name, seed, prompt):
    intermediate_outputs = ""

    # Paraphrasing
    paraphrase_prompt = f"Paraphrase this sentence while preserving its meaning: {prompt}"
    prompt_messages = [{"role": "user", "content": paraphrase_prompt}]
    paraphrase, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=50, seed=seed, json_output=False)
    intermediate_outputs += f"Paraphrased prompt: {paraphrase.strip()}\n"

    # Synonym Substitution
    synonym_prompt = f"Substitute some words in this sentence with their synonyms: {prompt}"
    prompt_messages = [{"role": "user", "content": synonym_prompt}]
    synonym_substitution, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=50, seed=seed, json_output=False)
    intermediate_outputs += f"Synonym substituted prompt: {synonym_substitution.strip()}\n"

    # Round-trip Translation
    translation_prompt = f"Translate this sentence to French and then back to English: {prompt}"
    prompt_messages = [{"role": "user", "content": translation_prompt}]
    round_trip_translation, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=50, seed=seed, json_output=False)
    intermediate_outputs += f"Round-trip translated prompt: {round_trip_translation.strip()}\n"

    # Adversarial Sentiment Change
    sentiment_prompt = f"Subtly change this sentence to reverse its sentiment: {prompt}"
    prompt_messages = [{"role": "user", "content": sentiment_prompt}]
    adversarial_sentiment, _ = call_api(client, model_name, prompt_messages, temperature=0.7, max_tokens=50, seed=seed, json_output=False)
    intermediate_outputs += f"Adversarial sentiment changed prompt: {adversarial_sentiment.strip()}\n"

    # Combine perturbed prompts
    perturbed_prompts = [
        paraphrase.strip(),
        synonym_substitution.strip(),
        round_trip_translation.strip(),
        adversarial_sentiment.strip()
    ]
    final_prompt = ' '.join(perturbed_prompts)

    return final_prompt, intermediate_outputs


## Step 4: Define the style evaluator
def style_evaluator(client, model_name, seed, original_prompt, proposed_prediction):
    prompt = f"Given the original prompt: {original_prompt}\n\n"
    prompt += f"The proposed method produced the following output:\n{proposed_prediction}\n\n"
    prompt += "Determine if the proposed method's output includes the following components:\n"
    prompt += "1. Paraphrased prompt\n"
    prompt += "2. Synonym substituted prompt\n"
    prompt += "3. Round-trip translated prompt\n"
    prompt += "4. Adversarial sentiment changed prompt\n"
    prompt += "Answer with a simple 'yes' or 'no'."

    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)

    if response.strip().lower() == "yes":
        return True
    else:
        return False


## Step 5: Define the output evaluator
def output_evaluator(client, model_name, seed, original_prompt, prediction):
    prompt = f"On a scale of 1 to 10, where 1 is completely different and 10 is exactly the same, rate how similar the following two sentences are:\n\n"
    prompt += f"Sentence 1: {original_prompt}\n"
    prompt += f"Sentence 2: {prediction}\n\n"
    prompt += "Provide only a single number rating between 1 and 10."

    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(client, model_name, prompt_messages, temperature=0., max_tokens=1, seed=seed, json_output=False)

    try:
        rating = float(response.strip())
        if 1 <= rating <= 10:
            return rating
        else:
            return None
    except ValueError:
        return None


## Step 6: Define the function that runs the experiments to obtain model predictions and performance
def run_experiment(client, model_name, seed, testset):
    sample_size = len(testset)
    baseline_predictions = []
    proposed_predictions = []

    baseline_scores = []
    proposed_scores = []

    style_check = []

    for i in tqdm(range(sample_size)):
        original_prompt = testset[i]["input"].strip()

        baseline_prediction = baseline_method(client, model_name, seed, original_prompt)
        proposed_prediction_final, proposed_prediction_intermediate = proposed_method(client, model_name, seed, original_prompt)
        baseline_predictions.append(baseline_prediction)
        proposed_predictions.append(proposed_prediction_final)

        baseline_score = output_evaluator(client, model_name, seed, original_prompt, baseline_prediction)
        proposed_score = output_evaluator(client, model_name, seed, original_prompt, proposed_prediction_final)
        if baseline_score is not None:
            baseline_scores.append(baseline_score)
        if proposed_score is not None:
            proposed_scores.append(proposed_score)

        style_check.append(style_evaluator(client, model_name, seed, original_prompt, proposed_prediction_intermediate))

    return baseline_scores, proposed_scores, style_check


## Step 7: Execute the experiments and compare performance
if __name__ == "__main__":
    testset = generate_testset()
    print(f"Generated {len(testset)} test examples for evaluation.")

    model_name = "claude-3-opus-20240229"
    seed = 2024
    client = load_model(model_name)
    print(f"Using model: {model_name}")

    baseline_scores, proposed_scores, style_check = run_experiment(client, model_name, seed, testset)
    print(f"Baseline average score: {sum(baseline_scores) / len(baseline_scores):.2f}")
    print(f"Proposed average score: {sum(proposed_scores) / len(proposed_scores):.2f}")
    print(f"Style check pass rate: {sum(style_check) / len(style_check):.2f}")
