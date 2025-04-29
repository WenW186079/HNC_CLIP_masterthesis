#!/bin/bash

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

MODEL_TYPE="finetuned"  # "base","finetuned"

paths=( "$PROJECT_ROOT/models/HNC_L2_1_S_text/epoch_"{1..100}"_full_encoder.pt" )
CHECKPOINTS="${paths[*]}"

FINETUNE_TYPE='full_encoder'     # 'text_encoder','vision_encoder','full_encoder'
EVAL_METHOD="random"             # "cosine", "plot", "random", "distinguish"
MODEL_NAME="ViT-B/32"
BATCH_SIZE=32

LOADER_TYPE='hnc'
TEST_JSON="$PROJECT_ROOT/data/HNC/hnc_clean_strict_test.json"
IMAGES_PATH="$PROJECT_ROOT/data/gqa_dataset/images/images"

# LOADER_TYPE='coco'
# TEST_JSON="$PROJECT_ROOT/data/Coco/test_coco_aug_withneg.json"
# IMAGES_PATH="$PROJECT_ROOT/data/Coco/val2014"

CUDA_VISIBLE_DEVICES=3  python eval/evaluation.py \
    --model_type $MODEL_TYPE \
    --checkpoint_path $CHECKPOINTS \
    --eval_method $EVAL_METHOD \
    --model_name $MODEL_NAME \
    --test_json $TEST_JSON \
    --images_path $IMAGES_PATH \
    --batch_size $BATCH_SIZE \
    --finetune_mode $FINETUNE_TYPE \
    --loader_type $LOADER_TYPE
