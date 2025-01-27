import os
import json
import logging
from PIL import Image  
import torch
from torch.utils.data import DataLoader  
from torch.optim import AdamW
from transformers import CLIPModel, CLIPProcessor
from torch.nn import functional as F
from huggingface_hub import HfApi, HfFolder, Repository 
import wandb

from load_data import LoadHNCPair, UniqueImageSampler, show_batches
from loss_func import safe_exp, HNC_Loss
from hnc_finetune import train_clip_model, preprocess_text_and_images, push_to_hub

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Paths
train_json_file_path = './HNC/hnc_clean_strict_train.json'
image_folder_path = './gqa_dataset/images/images'

# Hyperparameters
batch_size = 128
num_epochs = 1
learning_rate = 1e-4
weight_decay = 1e-5
output_dir = "./fine_tuned_clip"

# Initialize CLIP Models
# Load CLIP model: "ViT-B/16","ViT-B/32","ViT-L/14","ViT-L/14@336px"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load ref_encoder
ref_model = model.eval() 

# Load data
with open(train_json_file_path, 'r') as f:
    train_annotations = json.load(f)

dataset = LoadHNCPair(
    annotations=train_annotations,
    image_folder=image_folder_path,
)
logging.info("Dataset loaded successfully.")
sampler = UniqueImageSampler(dataset, batch_size)
data_loader = DataLoader(dataset, batch_sampler=sampler)
# show_batches(data_loader)
logging.info("finish data_loader.")

loss_fn = HNC_Loss(
        temperature=0.1,
        hard_negative_weight=1.0,
        l2_reg_weight=1e-3,
        ref_model=ref_model,
)
logging.info("finish loss_fn.")

optimizer = AdamW(model.vision_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
logging.info("finish optimizer.")

# Train the model
logging.info("start training.")
train_clip_model(model, processor, data_loader, loss_fn, optimizer, num_epochs, device)

# Push to hub
repo_name = "HNC_clip_32"  
push_to_hub(
    model=model,
    processor=processor,
    repo_name=repo_name,
    output_dir=output_dir,
    commit_message="Fine-tuned CLIP model for HNC"
)
    
