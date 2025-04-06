#!/bin/bash

# Ensure the script stops on errors
set -e

# Variables
PYTHON_FILE="eval_distinguish.py"    

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
MODEL_NAME="WenWW/HNC_clip_test"




# WenWW/HNC_1_2048_epoch1,\
# WenWW/HNC_1_2048_epoch2,\
# WenWW/HNC_1_2048_epoch3,\
# WenWW/HNC_1_2048_epoch4,\
# WenWW/HNC_1_2048_epoch5,\
# WenWW/HNC_1_2048_epoch6,\
# WenWW/HNC_1_2048_epoch7,\
# WenWW/HNC_1_2048_epoch8,\
# WenWW/HNC_1_2048_epoch9,\
# WenWW/HNC_1_2048_epoch10,\
# WenWW/HNC_1_2048_epoch11,\
# WenWW/HNC_1_2048_epoch12,\
# WenWW/HNC_1_2048_epoch13,\
# WenWW/HNC_1_2048_epoch14,\
# WenWW/HNC_1_2048_epoch15,\
# WenWW/HNC_1_2048_epoch16,\
# WenWW/HNC_1_2048_epoch17,\
# WenWW/HNC_1_2048_epoch18"




BATCH_SIZE=32  

python $PYTHON_FILE --dataset $DATASET_PATH \
                    --image_folder $IMAGE_FOLDER \
                    --model_name $MODEL_NAME \
                    --dataset_type $DATASET_TYPE \
                    --batch_size $BATCH_SIZE



# Nano1337/openclip-negclip,\
# WenWW/HNC_10_2048_epoch15"