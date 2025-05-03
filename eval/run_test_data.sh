#!/bin/bash

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

paths=(
  "$PROJECT_ROOT/models/DPO_KL_S_full/epoch_"{1..42}"_full_encoder.pt"
  "$PROJECT_ROOT/models/C_DPO_KL_S_full/epoch_"{1..50}"_full_encoder.pt"
  "$PROJECT_ROOT/models/HNC_L2_100_S_full/epoch_"{1..20}"_full_encoder.pt"
)
CHECKPOINTS="${paths[*]}"

MODEL_TYPE="finetuned"              # "base","finetuned"
FINETUNE_TYPE='full_encoder'        # 'text_encoder','vision_encoder','full_encoder',"last_encoder"
EVAL_METHOD="random+distinguish"    # "cosine", "plot", "random", "distinguish", 'random+distinguish'
MODEL_NAME="ViT-B/32"
BATCH_SIZE=32

LOADER_TYPE='hnc'
TEST_JSON="$PROJECT_ROOT/data/HNC/hnc_clean_strict_test.json"
IMAGES_PATH="$PROJECT_ROOT/data/gqa_dataset/images/images"

# LOADER_TYPE='coco'
# TEST_JSON="$PROJECT_ROOT/data/Coco/test_coco_aug_withneg.json"
# IMAGES_PATH="$PROJECT_ROOT/data/Coco/val2014"

CUDA_VISIBLE_DEVICES=3 python eval/evaluation.py \
    --model_type $MODEL_TYPE \
    --checkpoint_path $CHECKPOINTS \
    --eval_method $EVAL_METHOD \
    --model_name $MODEL_NAME \
    --test_json $TEST_JSON \
    --images_path $IMAGES_PATH \
    --batch_size $BATCH_SIZE \
    --finetune_mode $FINETUNE_TYPE \
    --loader_type $LOADER_TYPE
