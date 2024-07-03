# python3 src/tournament_ranking.py --engine claude-3-5-sonnet-20240620 --cache_name openreview_benchmark --max_round 10 > logs/openreview_papers_claude3-5_tournament.log 2>&1

python3 src/tournament_ranking2.py --engine claude-3-5-sonnet-20240620 --cache_name cache_results_claude_may/experiment_plans_claude3-5/uncertainty_prompting --max_round 6 > logs/uncertainty_prompting_claude3-5_tournament.log 2>&1


python3 src/tournament_ranking2.py --engine claude-3-5-sonnet-20240620 --cache_name cache_results_claude_may/experiment_plans_claude3-5/uncertainty_prompting_RAG --max_round 6 > logs/uncertainty_prompting_RAG_claude3-5_tournament.log 2>&1
