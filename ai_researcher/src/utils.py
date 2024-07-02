import os
import json
import random 

def calc_price(model, usage):
    if "claude" in model:
        return (0.015 * usage.input_tokens + 0.075 * usage.output_tokens) / 1000.0
    if model == "gpt-4-1106-preview" or model == "gpt-4-0125-preview":
        return (0.01 * usage.prompt_tokens + 0.03 * usage.completion_tokens) / 1000.0
    if model == "gpt-4":
        return (0.03 * usage.prompt_tokens + 0.06 * usage.completion_tokens) / 1000.0
    if (model == "gpt-3.5-turbo") or (model == "gpt-3.5-turbo-1106"):
        return (0.0015 * usage.prompt_tokens + 0.002 * usage.completion_tokens) / 1000.0
    if model == "gpt-4o":
        return (0.005 * usage.prompt_tokens + 0.015 * usage.completion_tokens) / 1000.0

def call_api(client, model, prompt_messages, temperature=1.0, max_tokens=100, seed=2024, json_output=False):
    if "claude" in model:
        if json_output:
            prompt = prompt_messages[0]["content"] + " Directly output the JSON dict with no additional text (avoid the presence of newline characters (\"\n\") and unescaped double quotes within the string so that we can call json.loads() on the output later)."
            prompt_messages = [{"role": "user", "content": prompt}]
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=prompt_messages
        )
        cost = calc_price(model, message.usage)
        response = message.content[0].text
    else:   
        response_format = {"type": "json_object"} if json_output else {"type": "text"}
        completion = client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            response_format=response_format
        )
        cost = calc_price(model, completion.usage)
        response = completion.choices[0].message.content.strip()
    
    return response, cost

def call_api_claude(client, model, prompt_messages, temperature=1.0, max_tokens=100):
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=prompt_messages
    )
    cost = calc_price(model, message.usage)
    response = message.content[0].text

    return response, cost

def cache_output(output, file_name):
    if file_name.endswith(".txt"):
        ## store GPT4 output into a txt file
        with open(file_name, "w") as f:
            f.write(output)
    elif file_name.endswith(".json"):
        ## store GPT4 output into a json file
        with open(file_name, "w") as f:
            json.dump(output, f, indent=4)
    return 

def print_idea_json(filename):
    with open(filename, "r") as f:
        idea_json = json.load(f)
    idea = idea_json["final_plan_json"]
    name = idea_json["idea_name"]
    print (name)
    for k,v in idea.items():
        if len(v) > 5:
            print ('- ' + k)
            print (v.strip() + '\n')

def format_plan_json(experiment_plan_json, indent_level=0, skip_test_cases=True, skip_fallback=True):
    try:
        output_str = ""
        indent = "  " * indent_level
        for k, v in experiment_plan_json.items():
            if k == "score":
                continue
            if skip_test_cases and k == "Test Case Examples":
                continue
            if skip_fallback and k == "Fallback Plan":
                continue
            if isinstance(v, (str, int, float)):  
                output_str += f"{indent}{k}: {v}\n"
            elif isinstance(v, list):
                output_str += f"{indent}{k}:\n"
                for item in v:
                    if isinstance(item, dict):
                        output_str += format_plan_json(item, indent_level + 1)
                    else:
                        output_str += f"{indent}  - {item}\n"
            elif isinstance(v, dict):
                output_str += f"{indent}{k}:\n"
                output_str += format_plan_json(v, indent_level + 1)
        return output_str
    except Exception as e:
        print("Error in formatting experiment plan json: ", e)
        return ""


def shuffle_dict_and_convert_to_string(input_dict):
    # Convert dict items to a list and shuffle
    items = list(input_dict.items())
    random.shuffle(items)
    
    # Convert back to dict and then to a JSON-formatted string
    shuffled_dict = dict(items)
    json_str = json.dumps(shuffled_dict, indent=4)
    
    return json_str

def clean_code_output(code_output):
    code_output = code_output.strip()
    if code_output.startswith("```python"):
        code_output = code_output[len("```python"):].strip()
    if code_output.endswith("```"):
        code_output = code_output[:-len("```")].strip()
    return code_output

def concat_reviews(paper_json):
    review_str = ""
    meta_review = paper_json["meta_review"]
    all_reviews = paper_json["reviews"]

    review_str += "Meta Review:\n" + meta_review + "\n\n"
    for idx, review in enumerate(all_reviews):
        review_str += "Reviewer #{}:\n".format(idx+1) + "\n"
        for key, value in review.items():
            if key in ["summary", "soundness", "contribution", "strengths", "weaknesse", "questions", "rating", "confidence"]:
                review_str += key + ": " + value["value"] + "\n"
        review_str += "\n"

    return review_str

def avg_score(scores):
    scores = [int(s[0]) for s in scores]
    return sum(scores) / len(scores)

def max_score(scores):
    scores = [int(s[0]) for s in scores]
    return max(scores)

def min_score(scores):
    scores = [int(s[0]) for s in scores]
    return min(scores)

if __name__ == "__main__":
    # filename = "/Users/clsi/Desktop/AI-Researcher/cache_results_claude/lit_review/uncertainty_prompting_method.json"
    # # print_idea_json(filename)
    # with open(filename, "r") as f:
    #     ideas = json.load(f)
    # # print ("Excitement Ranking: ")
    # # print (ideas["excitement_rationale"])
    # # print (ideas["excitement_score"])
    
    # for i in range(10):
    #     paper = ideas["paper_bank"][i]
    #     print (i+1)
    #     print ("Title: " + paper["title"])
    #     print ("Abstract: " + paper["abstract"])
    #     print ("\n\n")

    # with open("/Users/clsi/Desktop/AI-Researcher/openreview_benchmark/paper_0.json", "r") as f:
    #     paper = json.load(f)
    # print (concat_reviews(paper))

    with open("/Users/clsi/Desktop/AI-Researcher/cache_results_claude_may/experiment_plans/factuality_prompting_method_prompting/adaptive_prompting.json", "r") as f:
        paper_json = json.load(f)
    
    print (format_plan_json(paper_json["full_experiment_plan"]))

