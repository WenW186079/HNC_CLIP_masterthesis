#!/bin/bash

# Ensure the script stops on errors
set -e

# Variables
PYTHON_FILE="eval_distinguish.py"    
# PYTHON_FILE="eval_retrieval.py" 


# ===path for hnc===
DATASET_TYPE="hnc"   # 'hnc' or 'coco'
IMAGE_FOLDER="./gqa_dataset/images/images" 
DATASET_PATH="./HNC/hnc_val_sampled_1_percent.json"
# DATASET_PATH="./HNC/hnc_clean_strict_val.json"
# DATASET_PATH="./HNC/hnc_val_sampled_10_percent.json"


# ===path for coco===
# DATASET_TYPE="coco"  # 'hnc' or 'coco'
# IMAGE_FOLDER="./Coco/val2014" 
# DATASET_PATH="./Coco/test_coco_aug_withneg.json"
  

# ===model_name===
# MODEL_NAME="openai/clip-vit-base-patch32" 
# MODEL_NAME='Nano1337/openclip-negclip' 
MODEL_NAME='WenWW/HNC_CLIP_B32_1.5' 

BATCH_SIZE=32  

python $PYTHON_FILE --dataset $DATASET_PATH \
                    --image_folder $IMAGE_FOLDER \
                    --model_name $MODEL_NAME \
                    --dataset_type $DATASET_TYPE \
                    --batch_size $BATCH_SIZE
