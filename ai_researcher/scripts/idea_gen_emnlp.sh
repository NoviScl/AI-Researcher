topic_names=("math_prompting_method")
ideas_n=5
methods=("prompting")

# Iterate over each seed from 1 to 1000
for seed in {1..1}; do
    # Iterate over each cache name 
    for topic in "${topic_names[@]}"; do
        # Iterate over each method 
        for method in "${methods[@]}"; do
            echo "Running grounded_idea_gen.py on: $topic with seed $seed"
            python3 src/grounded_idea_gen.py \
             --engine "claude-3-5-sonnet-20240620" \
             --idea_cache "../cache_results_claude_may/ideas_5k/$topic.json" \
             --topic_description "Interpretability and Analysis of Models for NLP" \
             --method "$method" \
             --ideas_n $ideas_n \
             --seed $seed >> logs/idea_generation_${topic}.log 2>&1
        done
    done
done


## tmux 8