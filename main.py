import os
import json
import random
import logging
from PIL import Image  
import torch
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms  
from torch.optim import AdamW
import clip 
import deepspeed  

from load_data import HNCCLIPDataset, load_data_pairs 
from Loss_func import HNC_Loss
from train import train_clip_with_hnc_loss, push_vision_encoder_to_hub


os.environ["TORCH_EXTENSIONS_DIR"] = "/mount/studenten/team-lab-cl/data2024/w/data/torch_extensions/"

# Paths
train_json_file_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_train_sampled_1_percent.json'
val_json_file_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/HNC/hnc_val_sampled_1_percent.json'
image_folder_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/gqa_dataset/images/images'
config_path = '/mount/studenten/team-lab-cl/data2024/w/data/thes/config/deepspeed_config.json'

# Hyperparameters
batch_size = 32
epochs = 1

train_loader = load_data_pairs(
    json_file_path=train_json_file_path, 
    image_folder_path=image_folder_path, 
    batch_size=batch_size, 
    )

val_loader = load_data_pairs(
    json_file_path=val_json_file_path, 
    image_folder_path=image_folder_path, 
    batch_size=batch_size, 
    shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"


'''
# Display top 5 pairs and random 5 pairs
for batch_idx, (images, pos_captions, neg_captions, sources, image_paths) in enumerate(train_loader):
    print(f"\nDisplay TOP 5 pairs: \n--- Batch {batch_idx + 1} ---")
    for i in range(min(5, len(image_paths))):  
        print(f"Pair {i + 1}:")
        print(f"  Image Path: {image_paths[i]}")
        print(f"  Positive Caption: {pos_captions[i]}")
        print(f"  Negative Caption: {neg_captions[i]}")
        print(f"  Source of Negative: {sources[i]}")
        print("-" * 50)

    print(f"\nDisplay random 5 pairs from each batch: \n--- Batch {batch_idx + 1} ---")
    random_indices = random.sample(range(len(image_paths)), min(5, len(image_paths)))
    
    for i, rand_idx in enumerate(random_indices):
        print(f"Pair {i + 1}:")
        print(f"  Image Path: {image_paths[rand_idx]}")
        print(f"  Positive Caption: {pos_captions[rand_idx]}")
        print(f"  Negative Caption: {neg_captions[rand_idx]}")
        print(f"  Source of Negative: {sources[rand_idx]}")
        print("-" * 50)

    break  
'''

# Load CLIP model: "ViT-B/16","ViT-B/32","ViT-L/14","ViT-L/14@336px"
model, preprocess = clip.load("ViT-B/32", device=device)
model.float()

# Freeze text encoder; only train vision encoder
for name, param in model.named_parameters():
    if "transformer" in name or "token_embedding" in name or "text_projection" in name:
        param.requires_grad = False

# Save original parameters
clip_params = {name: param.clone().detach() for name, param in model.named_parameters()}

# Define Loss and Optimizer
criterion = HNC_Loss(clip_params, alpha=0.5, tau=0.07, lambda_=0.1)

# Train the model
model_engine = train_clip_with_hnc_loss(
    model,
    train_loader,
    val_loader,
    criterion,
    epochs=epochs,
    config_path=config_path
)

push_vision_encoder_to_hub(model_engine.module, "HNC_CLIP_ViT", "WenWW")
