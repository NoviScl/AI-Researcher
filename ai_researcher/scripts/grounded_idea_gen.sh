## example usage
topic_names=("factuality_prompting_method")
ideas_n=5 ## batch size
methods=("prompting")
rag_values=("True" "False")

# Iterate over each seed 
for seed in {1..2}; do
    # Iterate over each topic name 
    for topic in "${topic_names[@]}"; do
        # Iterate over each method 
        for method in "${methods[@]}"; do
            # Iterate over RAG values True and False
            for rag in "${rag_values[@]}"; do
                echo "Running grounded_idea_gen.py on: $topic with seed $seed and RAG=$rag"
                python3 src/grounded_idea_gen.py \
                 --engine "claude-3-5-sonnet-20240620" \
                 --paper_cache "../cache_results_test/lit_review/$topic.json" \
                 --idea_cache "../cache_results_test/seed_ideas/$topic.json" \
                 --grounding_k 10 \
                 --method "$method" \
                 --ideas_n $ideas_n \
                 --seed $seed \
                 --RAG $rag 
            done
        done
    done
done


# topic_names=("uncertainty_prompting_method" "safety_prompting_method" "multilingual_prompting_method" "math_prompting_method" "factuality_prompting_method" "coding_prompting_method" "bias_prompting_method")
# ideas_n=5
# methods=("prompting")
# rag_values=("True" "False")

# # Iterate over each seed from 1 to 1000
# for seed in {1..1000}; do
#     # Iterate over each topic name 
#     for topic in "${topic_names[@]}"; do
#         # Iterate over each method 
#         for method in "${methods[@]}"; do
#             # Iterate over RAG values True and False
#             for rag in "${rag_values[@]}"; do
#                 echo "Running grounded_idea_gen.py on: $topic with seed $seed and RAG=$rag"
#                 python3 src/grounded_idea_gen.py \
#                  --engine "claude-3-5-sonnet-20240620" \
#                  --paper_cache "../cache_results_claude_may/lit_review_new/$topic.json" \
#                  --idea_cache "../cache_results_claude_may/ideas_5k/$topic.json" \
#                  --grounding_k 10 \
#                  --method "$method" \
#                  --ideas_n $ideas_n \
#                  --seed $seed \
#                  --RAG $rag >> logs/idea_generation_${topic}_RAG_${rag}.log 2>&1
#             done
#         done
#     done
# done


