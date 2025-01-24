# import os
# import json
# import logging
# from PIL import Image  
# import torch
# from torch.utils.data import DataLoader  
# from torch.optim import AdamW
# from transformers import CLIPModel, CLIPProcessor
# from torch.nn import functional as F
# from huggingface_hub import HfApi, HfFolder, Repository
# import deepspeed  
# import wandb
# import yaml

import yaml
import logging
import json
import os
import wandb
import torch
import deepspeed
from transformers import CLIPModel, CLIPProcessor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from load_data import LoadHNCPair, UniqueImageSampler, show_batches
from loss_func import safe_exp, HNC_Loss
from hnc_finetune_deepspeed import train_clip_model, preprocess_text_and_images, push_to_hub

os.environ["TORCH_EXTENSIONS_DIR"] = "/mount/studenten/team-lab-cl/data2024/w/data/torch_extensions/"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7,8"

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Dataset and paths
train_json_path = CONFIG["dataset"]["train_json_path"]
image_folder_path = CONFIG["dataset"]["image_folder_path"]

# Training parameters
num_epochs = CONFIG["training"]["num_epochs"]
batch_size = CONFIG["training"]["batch_size"]
learning_rate = CONFIG["training"]["learning_rate"]
output_dir = CONFIG["misc"]["output_dir"]
repo_name = CONFIG["misc"]["repo_name"]

# DeepSpeed configuration
deepspeed_config = CONFIG["deepspeed"]["config"]

# Initialize CLIP Models
# Load CLIP model: "ViT-B/16","ViT-B/32","ViT-L/14","ViT-L/14@336px"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load ref_encoder
ref_model = model.eval() 

# Load data
with open(train_json_path, 'r') as f:
    train_annotations = json.load(f)

dataset = LoadHNCPair(
    annotations=train_annotations,
    image_folder=image_folder_path,
)
sampler = UniqueImageSampler(dataset, batch_size)
data_loader = DataLoader(dataset, batch_sampler=sampler)
# show_batches(data_loader)
logging.info("finish data_loader.")

# Define Loss 
loss_fn = HNC_Loss(
    temperature=CONFIG["training"]["temperature"],
    hard_negative_weight=CONFIG["training"]["hard_negative_weight"],
    l2_reg_weight=CONFIG["training"]["l2_reg_weight"],
    ref_model=ref_model,
)


# Define the optimizer using AdamW
optimizer = AdamW(
    model.vision_model.parameters(),
    lr=learning_rate,
    weight_decay=CONFIG["training"]["weight_decay"],
)

for param in model.parameters():
    if not param.is_contiguous():
        param.data = param.data.contiguous()

# Initialize DeepSpeed
model_engine, _ , _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.vision_model.parameters(),
    config=deepspeed_config,
    optimizer=optimizer,
)

torch.cuda.empty_cache()

# Train the models
logging.info("start training.")
train_clip_model(
    model_engine=model_engine,
    processor=processor,
    data_loader=data_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    num_epochs=num_epochs,
    device=device,
    learning_rate=learning_rate,
)

# Push to hub
push_to_hub(
    model=model,
    processor=processor,
    repo_name=repo_name,
    output_dir=output_dir,
    commit_message="Fine-tuned CLIP model for HNC"
)