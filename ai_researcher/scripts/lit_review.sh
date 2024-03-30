## factuality 
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods that can improve factuality and reduce hallucination of large language models" \
 --cache_name "factuality_prompting" \
 --track "method" \
 --max_paper_bank_size 70 \
 --print_all


## uncertainty 
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods that can better quantify uncertainty or confidence of large language models" \
 --cache_name "uncertainty_prompting" \
 --track "method" \
 --max_paper_bank_size 70 \
 --print_all


## attack 
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods that can jailbreak or adversarially attack large language models" \
 --cache_name "attack_prompting" \
 --track "method" \
 --max_paper_bank_size 70 \
 --print_all


## defense 
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods that can defend against adversarial attacks or prompt injection on large language models and improve robustness" \
 --cache_name "defense_prompting" \
 --track "method" \
 --max_paper_bank_size 70 \
 --print_all


## reasoning 
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods that can improve reasoning of large language models" \
 --cache_name "reasoning_prompting" \
 --track "method" \
 --max_paper_bank_size 70 \
 --print_all


## bias
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods that can mitigate social biases of large language models" \
 --cache_name "bias_prompting" \
 --track "method" \
 --max_paper_bank_size 70 \
 --print_all


## mulimodal
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods that can improve multimodal understanding and problem solving of vision-language models" \
 --cache_name "multimodal_prompting" \
 --track "method" \
 --max_paper_bank_size 70 \
 --print_all


## multilingual
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods that can improve multilingual, code-switched, or low-resource language performance of large language models" \
 --cache_name "multilingual_prompting" \
 --track "method" \
 --max_paper_bank_size 70 \
 --print_all
