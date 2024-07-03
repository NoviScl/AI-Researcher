# Research Ideation Agent

This repo implements an agent that can generate detailed research project proposals given a topic description. It consists of several modules: (1) Related Work Search; (2) Idea Generation; (3) Idea Deduplication; (4) Project Proposal Generation; (5) Project Proposal Reranking; (6) Project Proposal Filtering. 

These modules are designed to be run sequentially as a pipeline so that the system takes a topic description as input and returns a list of project proposals as output. Each module can also be run separately as standalone research assistance tools. We describe how to run each module as well as the entire pipeline below.

## Setup

## Lit Review Agent

The lit review agents takes a topic description as input and returns a list of most relevant papers as the output, along with the relevance score of each paper. 

Example usage: 
```
bash lit_review.sh
```

Note that despite setting a seed for GPT-4, the results are not deterministic, since semantic scholar may return different results given the same query.

## Grounded Idea Generation 

The grounded idea generation agent takes a topic description and a list of relevant papers as input, and returns a list of generated ideas as the output. If you have done the lit review step already, just provide the cache file name to run this module.

Example usage: 
```
bash grounded_idea_gen.sh
```