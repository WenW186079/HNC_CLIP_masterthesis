#!/bin/bash

cd "$(dirname "$0")/.."

MODEL_TYPE="finetuned"  # "base" or "finetuned"

CHECKPOINTS="./checkpoints_dpo/final_model.pt"
EVAL_METHOD="random" # "cosine", "plot", "random"
MODEL_NAME="ViT-B/32"
TEST_JSON="./HNC/hnc_clean_strict_test.json"
IMAGES_PATH="./gqa_dataset/images/images"
BATCH_SIZE=32

python eval/test.py \
    --model_type $MODEL_TYPE \
    --checkpoint_path $CHECKPOINTS \
    --eval_method $EVAL_METHOD \
    --model_name $MODEL_NAME \
    --test_json $TEST_JSON \
    --images_path $IMAGES_PATH \
    --batch_size $BATCH_SIZE
