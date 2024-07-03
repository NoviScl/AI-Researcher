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
 --cache_name "../cache_results_claude_may/lit_review_new/safety_prompting_method.json" \
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
 --cache_name "../cache_results_claude_may/lit_review_new/adaptive_uncertainty_sampling.json" \
 --max_paper_bank_size 120 \
 --print_all
```

The `max_paper_bank_size` is a hyperparameter to control when to stop the paper search process (until the specified number of papers has been retrieved). The generated search queries as well as the ranked papers will be stored in the specified cache file. The cache file can be used as part of the input to the idea generation module. Note that we used `claude-3-opus-20240229` for the related paper search step in the paper. 


## Idea Generation

The idea generation module takes a topic description and optionally a list of relevant papers as input, and returns a list of generated ideas as the output. 

Example usage: 
```
topic_names=("uncertainty_prompting_method")
ideas_n=5
methods=("prompting")

# Iterate over each seed from 1 to 1000
for seed in {1..2}; do
    # Iterate over each cache name 
    for topic in "${topic_names[@]}"; do
        # Iterate over each method 
        for method in "${methods[@]}"; do
            echo "Running grounded_idea_gen.py on: $topic"
            python3 src/grounded_idea_gen.py \
             --engine "claude-3-5-sonnet-20240620" \
             --paper_cache "../cache_results_claude_may/lit_review_new/$topic.json" \
             --idea_cache "../cache_results_claude_may/ideas_5k/$topic.json" \
             --grounding_k 10 \
             --method "$method" \
             --ideas_n $ideas_n \
             --seed $seed \
             --RAG True > logs/idea_generation_${topic}_RAG.log 2>&1
        done
    done
done
```

Due to the max output length constraint, we recommend generating ideas in batches of 5 (`ideas_n=5`) and running the script multiple times with different seeds to collect a larger set of ideas. You can set `RAG` to either `True` or `False` to turn on or off retrieval augmentation where we ground the idea generation on retrieved papers. 

## Idea Deduplication


## Project Proposal Generation


## Project Proposal Reranking


## Project Proposal Filtering


## End-to-End Pipeline

