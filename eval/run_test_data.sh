#!/bin/bash

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)


paths=(

  # "$PROJECT_ROOT/models/HNC_KL_1_100k_full_20/epoch_12_full_encoder.pt" 

)
CHECKPOINTS="${paths[*]}"
OUTPUT_CSV="./coco_05.csv"

MODEL_TYPE="finetuned"              # "base","finetuned"
FINETUNE_TYPE='vision_encoder'        # 'text_encoder','vision_encoder','full_encoder',"last_encoder"
EVAL_METHOD="random+thresholds"    # "cosine", "plot", "random", "thresholds", 'random+thresholds'
MODEL_NAME="ViT-B/32"
BATCH_SIZE=32

LOADER_TYPE='hnc'
TEST_PATH="$PROJECT_ROOT/data/HNC/filtered_test_data.json"
IMAGES_PATH="$PROJECT_ROOT/data/gqa_dataset/images/images"

# LOADER_TYPE='coco'
# TEST_PATH="$PROJECT_ROOT/data/Coco/test_coco_aug_withneg.json"
# IMAGES_PATH="$PROJECT_ROOT/data/Coco/val2014"

CUDA_VISIBLE_DEVICES=0 python eval/evaluation.py \
    --model_type $MODEL_TYPE \
    --checkpoint_path $CHECKPOINTS \
    --eval_method $EVAL_METHOD \
    --model_name $MODEL_NAME \
    --test_json $TEST_PATH \
    --images_path $IMAGES_PATH \
    --batch_size $BATCH_SIZE \
    --finetune_mode $FINETUNE_TYPE \
    --loader_type $LOADER_TYPE \
    --output_csv $OUTPUT_CSV
