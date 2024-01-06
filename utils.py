def calc_price(model, usage):
    if model == "gpt-4-1106-preview":
        return (0.01 * usage.prompt_tokens + 0.03 * usage.completion_tokens) / 1000.0
    if model == "gpt-4":
        return (0.03 * usage.prompt_tokens + 0.06 * usage.completion_tokens) / 1000.0
    if (model == "gpt-3.5-turbo") or (model == "gpt-3.5-turbo-1106"):
        return (0.0015 * usage.prompt_tokens + 0.002 * usage.completion_tokens) / 1000.0