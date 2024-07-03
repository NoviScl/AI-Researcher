python3 src/filter_ideas.py \
 --engine "claude-3-5-sonnet-20240620" \
 --cache_name "cache_results_claude_may/experiment_plans_1k/uncertainty_prompting_method_prompting" \
 --score_file "uncertainty_score_predictions_swiss_round_5" > logs/filter_ideas_full.log 2>&1