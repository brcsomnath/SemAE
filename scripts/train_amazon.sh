python3 ../src/train.py \
        --data ../data/amazon/json/amazon_train.json \
        --sentencepiece ../data/sentencepiece/spm_amazon.model \
        --run_id amazon_run1 \
        --gpu 0 \
        --l1_cost 10000 \
        --entropy_cost 0.00005