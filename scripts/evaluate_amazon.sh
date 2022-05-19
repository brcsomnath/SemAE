python3 ../src/inference.py \
        --summary_data ../data/amazon/json/amazon_summ.json \
        --gold_data ../data/amazon/gold \
        --sentencepiece ../data/sentencepiece/spm_amazon.model \
        --split_by presplit --model ../models/amazon_run1_10_model.pt \
        --gpu 0 \
        --run_id amazon_run1 \
        --no_cut_sents \
        --max_tokens 100