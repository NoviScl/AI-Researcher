# python3 src/style_transfer.py \
#  --engine claude-3-5-sonnet-20240620 \
#  --cache_dir "../AI_AI_Ideas/" \
#  --idea_name "all" \
#  --format "json" \
#  --processed_cache_dir "../AI_AI_Ideas_Processed/" > logs/style_transfer_ai_ideas.log 2>&1

# python3 src/style_transfer.py \
#  --engine claude-3-5-sonnet-20240620 \
#  --cache_dir "../Human_Ideas_Txt/" \
#  --idea_name "all" \
#  --format "txt" \
#  --processed_cache_dir "../Human_Ideas_Txt_Processed/" > logs/style_transfer_human_ideas.log 2>&1

# python3 src/style_transfer.py \
#  --engine claude-3-5-sonnet-20240620 \
#  --cache_dir "../Human_Ideas_Txt_More/" \
#  --idea_name "all" \
#  --format "txt" \
#  --processed_cache_dir "../Human_Ideas_Txt_More_Processed/" > logs/style_transfer_human_ideas_more.log 2>&1


python3 src/style_transfer.py \
 --engine claude-3-5-sonnet-20240620 \
 --cache_dir "../AI_Human_Ideas/" \
 --idea_name "all" \
 --format "json" \
 --processed_cache_dir "../AI_Human_Ideas_Txt/" >> logs/style_transfer_ai_human_ideas.log 2>&1


