# The Stanford AI Researcher Agent

This repo implements an agent that can generate research project proposals given a topic description. It consists of several modules: lit review, grounded idea generation, experiment plan generation, novelty check, and idea quality ranking. The design principle is that you can either use the whole system in an end-to-end manner to directly generate research proposals, or you can use each module separately as useful tools in your daily research pipeline. Below we introduce each module's usage, as well as the fully end-to-end pipeline.

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