#!/bin/bash

# Ensure the script stops on errors
set -e

# Variables
PYTHON_FILE="eval_distinguish.py"    
# PYTHON_FILE="eval_retrieval.py" 


# ===path for hnc===
# DATASET_TYPE="hnc"   # 'hnc' or 'coco'
# IMAGE_FOLDER="./gqa_dataset/images/images" 
# DATASET_PATH="./HNC/hnc_val_sampled_1_percent.json"
# DATASET_PATH="./HNC/hnc_clean_strict_val.json"
# DATASET_PATH="./HNC/hnc_val_sampled_10_percent.json"


# ===path for coco===
DATASET_TYPE="coco"  # 'hnc' or 'coco'
IMAGE_FOLDER="./Coco/val2014" 
DATASET_PATH="./Coco/test_coco_aug_withneg.json"
  

# ===model_name===
MODEL_NAME="openai/clip-vit-base-patch32,\
Nano1337/openclip-negclip,\
WenWW/HNC_D1-15_epoch1,\
WenWW/HNC_D1-15_epoch2,\
WenWW/HNC_D1-15_epoch3,\
WenWW/HNC_D1-15_epoch4,\
WenWW/HNC_D1-15_epoch5,\
WenWW/HNC_CLIP_B32_1.0,\
WenWW/HNC_CLIP_B32_1.5,\
WenWW/HNC_clip_D2.0,\
WenWW/HNC_CLIP_B32_D5,\
WenWW/HNC_D1-3_epoch1,\
WenWW/HNC_D1-3_epoch2,\
WenWW/HNC_D1-3_epoch3,\
WenWW/HNC_D1-3_epoch4,\
WenWW/HNC_D1-3_epoch5,\
WenWW/HNC_D1-1.5_2048_epoch1,\
WenWW/HNC_D1-1.5_2048_epoch2,\
WenWW/HNC_D1-1.5_2048_epoch3,\
WenWW/HNC_D1-1.5_2048_epoch4"


BATCH_SIZE=32  

python $PYTHON_FILE --dataset $DATASET_PATH \
                    --image_folder $IMAGE_FOLDER \
                    --model_name $MODEL_NAME \
                    --dataset_type $DATASET_TYPE \
                    --batch_size $BATCH_SIZE
