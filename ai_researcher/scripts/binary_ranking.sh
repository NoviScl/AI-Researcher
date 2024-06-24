python3 src/binary_ranking.py --engine gpt-4o --method zero_shot --cache_name ORB_full > logs/orb_full_gpt4o_zero_shot.log 2>&1
python3 src/binary_ranking.py --engine gpt-4o --method zero_shot_cot --cache_name ORB_full > logs/orb_full_gpt4o_zero_shot_cot.log 2>&1
python3 src/binary_ranking.py --engine gpt-4o --method few_shot --cache_name ORB_full > logs/orb_full_gpt4o_few_shot.log 2>&1
python3 src/binary_ranking.py --engine gpt-4o --method few_shot_cot --cache_name ORB_full > logs/orb_full_gpt4o_few_shot_cot.log 2>&1
python3 src/binary_ranking.py --engine gpt-4o --method zero_shot_sc --cache_name ORB_full > logs/orb_full_gpt4o_zero_shot_sc.log 2>&1

python3 src/binary_ranking.py --engine claude-3-opus-20240229 --method zero_shot --cache_name ORB_full > logs/orb_full_claude3_zero_shot.log 2>&1
python3 src/binary_ranking.py --engine claude-3-opus-20240229 --method zero_shot_cot --cache_name ORB_full > logs/orb_full_claude3_zero_shot_cot.log 2>&1
python3 src/binary_ranking.py --engine claude-3-opus-20240229 --method few_shot --cache_name ORB_full > logs/orb_full_claude3_few_shot.log 2>&1
python3 src/binary_ranking.py --engine claude-3-opus-20240229 --method few_shot_cot --cache_name ORB_full > logs/orb_full_claude3_few_shot_cot.log 2>&1
python3 src/binary_ranking.py --engine claude-3-opus-20240229 --method zero_shot_sc --cache_name ORB_full > logs/orb_full_claude3_zero_shot_sc.log 2>&1

python3 src/binary_ranking.py --engine claude-3-5-sonnet-20240620 --method zero_shot --cache_name ORB_full > logs/orb_full_claude3-5_zero_shot.log 2>&1
python3 src/binary_ranking.py --engine claude-3-5-sonnet-20240620 --method zero_shot_cot --cache_name ORB_full > logs/orb_full_claude3-5_zero_shot_cot.log 2>&1
python3 src/binary_ranking.py --engine claude-3-5-sonnet-20240620 --method few_shot --cache_name ORB_full > logs/orb_full_claude3-5_few_shot.log 2>&1
python3 src/binary_ranking.py --engine claude-3-5-sonnet-20240620 --method few_shot_cot --cache_name ORB_full > logs/orb_full_claude3-5_few_shot_cot.log 2>&1
python3 src/binary_ranking.py --engine claude-3-5-sonnet-20240620 --method zero_shot_sc --cache_name ORB_full > logs/orb_full_claude3-5_zero_shot_sc.log 2>&1
