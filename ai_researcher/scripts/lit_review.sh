## coding
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods for large language models to improve code generation" \
 --cache_name "coding_prompting" \
 --track "method" \
 --max_paper_bank_size 100 \
 --print_all


## math
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods for large language models to improve mathematical problem solving" \
 --cache_name "math_prompting" \
 --track "method" \
 --max_paper_bank_size 100 \
 --print_all


## factuality 
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods that can improve factuality and reduce hallucination of large language models" \
 --cache_name "factuality_prompting" \
 --track "method" \
 --max_paper_bank_size 100 \
 --print_all



## bias
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods to reduce social biases and stereotypes of large language models" \
 --cache_name "bias_prompting" \
 --track "method" \
 --max_paper_bank_size 100 \
 --print_all


## multilingual
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods to improve large language models’ performance on multilingual tasks or low-resource languages and vernacular languages" \
 --cache_name "multilingual_prompting" \
 --track "method" \
 --max_paper_bank_size 100 \
 --print_all


# ## mulimodal
# python3 src/lit_review.py \
#  --engine "claude-3-opus-20240229" \
#  --topic_description "novel prompting methods to improve large language models or vision-language models on multimodal tasks" \
#  --cache_name "multimodal_prompting" \
#  --track "method" \
#  --max_paper_bank_size 70 \
#  --print_all


## uncertainty 
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods that can better quantify uncertainty or calibrate the confidence of large language models" \
 --cache_name "uncertainty_prompting" \
 --track "method" \
 --max_paper_bank_size 100 \
 --print_all


## safety
python3 src/lit_review.py \
 --engine "claude-3-opus-20240229" \
 --topic_description "novel prompting methods to improve large language models' robustness against adversarial attacks or improve their security or privacy" \
 --cache_name "safety_prompting" \
 --track "method" \
 --max_paper_bank_size 100 \
 --print_all

 