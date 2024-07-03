# Research Ideation Agent

This repo implements an agent that can generate detailed research project proposals given a topic description. It consists of several modules: (1) Related Work Search; (2) Idea Generation; (3) Idea Deduplication; (4) Project Proposal Generation; (5) Project Proposal Reranking; (6) Project Proposal Filtering. 

These modules are designed to be run sequentially as a pipeline so that the system takes a topic description as input and returns a list of project proposals as output. Each module can also be run separately as standalone research assistance tools. We describe how to run each module as well as the entire pipeline below.

## Setup

Create `keys.json` and store it in the project root directory. The file should look like this:

```
{
    "api_key": "Your OpenAI API Key",
    "organization_id": "Your OpenAI Organization ID (Optional)",
    "s2_key": "Your Semantic Scholar API Key (Optional)",
    "anthropic_key": "Your Anthropic API Key"
}
```

## Related Work Search

The related work search module will iteratively propose search queries and search through the Semantic Scholar API. We then use an LLM to score the relevance of retrieved papers for reranking. The module takes a topic description or an idea as input and returns a list of the most relevant papers as output.

Example usage (finding related papers for a given topic):
```
python3 src/lit_review.py \
 --engine "claude-3-5-sonnet-20240620" \
 --mode "topic" \
 --topic_description "novel prompting methods to improve large language models' robustness against adversarial attacks or improve their security or privacy" \
 --cache_name "../cache_results_claude_may/lit_review_test/safety_prompting.json" \
 --max_paper_bank_size 120 \
 --print_all
```

Example usage (finding related papers for a given idea): 
```
python3 src/lit_review.py \
 --engine "claude-3-5-sonnet-20240620" \
 --mode "idea" \
 --idea_cache "../cache_results_claude_may/experiment_plans_claude3-5/uncertainty_prompting" \
 --idea_name "adaptive_uncertainty_sampling.json" \
 --cache_name "../cache_results_claude_may/lit_review_test/safety_prompting.json" \
 --max_paper_bank_size 120 \
 --print_all
```

The `max_paper_bank_size` is a hyperparameter to control when to stop the paper search process (until the specified number of papers has been retrieved). The generated search queries as well as the ranked papers will be stored in the specified cache file. The cache file can be used as part of the input to the idea generation module.


## Idea Generation

The grounded idea generation agent takes a topic description and a list of relevant papers as input, and returns a list of generated ideas as the output. If you have done the lit review step already, just provide the cache file name to run this module.

Example usage: 
```
bash grounded_idea_gen.sh
```

## Idea Deduplication


## Project Proposal Generation


## Project Proposal Reranking


## Project Proposal Filtering


## End-to-End Pipeline

