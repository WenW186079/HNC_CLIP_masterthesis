#!/bin/bash

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

MODEL_TYPE="finetuned"  # "base","finetuned"

CHECKPOINTS="$PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_1_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_2_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_3_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_4_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_5_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_6_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_7_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_8_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_9_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_10_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_11_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_12_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_13_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_14_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_15_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_16_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_17_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_18_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_19_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_20_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_21_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_22_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_23_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_24_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_25_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_26_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_27_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_28_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_29_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_30_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_31_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_32_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_33_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_34_full_encoder.pt $PROJECT_ROOT/checkpoints_HNC1_S_full/epoch_35_full_encoder.pt"
# CHECKPOINTS="$PROJECT_ROOT/checkpoints_standard_S_full/epoch_1_full_encoder.pt $PROJECT_ROOT/checkpoints_standard_S_full/epoch_2_full_encoder.pt $PROJECT_ROOT/checkpoints_standard_S_full/epoch_3_full_encoder.pt $PROJECT_ROOT/checkpoints_standard_S_full/epoch_4_full_encoder.pt $PROJECT_ROOT/checkpoints_standard_S_full/epoch_5_full_encoder.pt"
# CHECKPOINTS="$PROJECT_ROOT/checkpoints_hncd1_50_S_full_256/epoch_1_full_encoder.pt $PROJECT_ROOT/checkpoints_hncd1_50_S_full_256/epoch_2_full_encoder.pt $PROJECT_ROOT/checkpoints_hncd1_50_S_full_256/epoch_3_full_encoder.pt $PROJECT_ROOT/checkpoints_hncd1_50_S_full_256/epoch_4_full_encoder.pt"
# CHECKPOINTS="$PROJECT_ROOT/checkpoints_hncd1_50_S_full/epoch_1_full_encoder.pt $PROJECT_ROOT/checkpoints_hncd1_50_S_full/epoch_2_full_encoder.pt $PROJECT_ROOT/checkpoints_hncd1_50_S_full/epoch_3_full_encoder.pt $PROJECT_ROOT/checkpoints_hncd1_50_S_full/epoch_4_full_encoder.pt"
# CHECKPOINTS="$PROJECT_ROOT/checkpoints_hnc1_S_full_last/epoch_5_full_encoder.pt $PROJECT_ROOT/checkpoints_hnc1_S_full_last/epoch_10_full_encoder.pt"
# CHECKPOINTS="$PROJECT_ROOT/checkpoints_hnc10_S_full/epoch_5_full_encoder.pt $PROJECT_ROOT/checkpoints_hnc10_S_full/epoch_10_full_encoder.pt"
# CHECKPOINTS="$PROJECT_ROOT/checkpoints_hnc1_S_text/epoch_10_text_encoder.pt"
# CHECKPOINTS="$PROJECT_ROOT/checkpoints_hnc1_S_full/epoch_10_full_encoder.pt $PROJECT_ROOT/checkpoints_hnc1_S_full/epoch_20_full_encoder.pt $PROJECT_ROOT/checkpoints_hnc1_S_full/epoch_30_full_encoder.pt"
# CHECKPOINTS="$PROJECT_ROOT/checkpoints_dpo_S/final_model.pt"
# CHECKPOINTS="$PROJECT_ROOT/checkpoints_hnc_d0-100_L/epoch_3.pt"



FINETUNE_TYPE='vision_encoder'    # 'text_encoder','vision_encoder','full_encoder'
EVAL_METHOD="random"   # "cosine", "plot", "random","distinguish"
MODEL_NAME="ViT-B/32"
BATCH_SIZE=32

# LOADER_TYPE='hnc'
# TEST_JSON="$PROJECT_ROOT/data/HNC/hnc_clean_strict_test.json"
# IMAGES_PATH="$PROJECT_ROOT/data/gqa_dataset/images/images"

LOADER_TYPE='coco'
TEST_JSON="$PROJECT_ROOT/data/Coco/test_coco_aug_withneg.json"
IMAGES_PATH="$PROJECT_ROOT/data/Coco/val2014"

CUDA_VISIBLE_DEVICES=6  python eval/evaluation.py \
    --model_type $MODEL_TYPE \
    --checkpoint_path $CHECKPOINTS \
    --eval_method $EVAL_METHOD \
    --model_name $MODEL_NAME \
    --test_json $TEST_JSON \
    --images_path $IMAGES_PATH \
    --batch_size $BATCH_SIZE \
    --finetune_mode $FINETUNE_TYPE \
    --loader_type $LOADER_TYPE
