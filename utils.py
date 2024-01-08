import os

def calc_price(model, usage):
    if model == "gpt-4-1106-preview":
        return (0.01 * usage.prompt_tokens + 0.03 * usage.completion_tokens) / 1000.0
    if model == "gpt-4":
        return (0.03 * usage.prompt_tokens + 0.06 * usage.completion_tokens) / 1000.0
    if (model == "gpt-3.5-turbo") or (model == "gpt-3.5-turbo-1106"):
        return (0.0015 * usage.prompt_tokens + 0.002 * usage.completion_tokens) / 1000.0

def call_api(openai_client, model, prompt_messages, temperature=1.0, max_tokens=100, json_output=False):
    response_format = {"type": "json_object"} if json_output else {"type": "text"}
    completion = openai_client.chat.completions.create(
        model=model,
        messages=prompt_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format
    )
    cost = calc_price(model, completion.usage)
    response = completion.choices[0].message.content.strip()
    
    return response, cost

def cache_output(output, file_name):
    ## store GPT4 output into a txt file
    with open(os.path.join("cache_results", file_name), "w") as f:
        f.write(output)
    return 