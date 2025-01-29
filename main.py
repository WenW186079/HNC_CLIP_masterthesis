import yaml
import logging
import json
import os
import wandb
import torch
import deepspeed
import deepspeed.comm as dist 
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse

from load_data import LoadHNCPair, show_batches
from loss_func import safe_exp, HNC_Loss
from train_hnc import train_clip_model, preprocess_text_and_images, push_to_hub

# best_model_dir = f"./epoch_cache/epoch_5"
# best_model = CLIPModel.from_pretrained(best_model_dir)
# best_processor = CLIPProcessor.from_pretrained(best_model_dir)

# push_to_hub(
#             model=best_model,
#             processor=best_processor,
#             repo_name='HNC_D1-15_epoch5'
#         )
# print('pushed')

parser = argparse.ArgumentParser(description="Train CLIP with DeepSpeed")
parser.add_argument("--config_path", type=str, default="config/config.yaml", help="Path to config.yaml")
parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training") 
args = parser.parse_args()

deepspeed.init_distributed()
torch.cuda.set_device(args.local_rank)

if not dist.is_initialized():
    dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=torch.cuda.device_count())

if args.local_rank == 0:
    logging.info(f"Running on local rank: {args.local_rank}")


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

with open(args.config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

# Dataset and paths
train_json_path = CONFIG["dataset"]["train_json_path"]
image_folder_path = CONFIG["dataset"]["image_folder_path"]

num_epochs = CONFIG["training"]["num_epochs"]
batch_size = CONFIG["deepspeed"]["config"]["train_batch_size"]
train_micro_batch_size_per_gpu = CONFIG["deepspeed"]["config"]["train_micro_batch_size_per_gpu"]

betas=(0.9, 0.98)
eps=1e-6


# Initialize CLIP Models
# Load CLIP model: "ViT-B/16","ViT-B/32","ViT-L/14","ViT-L/14@336px"
# openai/clip-vit-large-patch14-336
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load ref_encoder
ref_model = model.eval() 

# Load data
with open(train_json_path, 'r') as f:
    train_annotations = json.load(f)

dataset = LoadHNCPair(annotations=train_annotations,image_folder=image_folder_path)
sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True
)
data_loader = DataLoader(
    dataset,
    batch_size=train_micro_batch_size_per_gpu, 
    sampler=sampler,
)

if dist.get_rank() == 0:
    logging.info(f"Dataset size: {len(dataset)}, Batch size: {train_micro_batch_size_per_gpu}")
    logging.info(f"Number of batches: {len(data_loader)}")
# show_batches(data_loader)

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
    lr=CONFIG["training"]["learning_rate"],
    weight_decay=CONFIG["training"]["weight_decay"],
    betas=betas, 
    eps=eps,
)

total_training_steps = len(data_loader) * num_epochs
scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=CONFIG["training"]["num_warmup_steps"] ,
        num_training_steps=total_training_steps
    )


for param in model.parameters():
    if not param.is_contiguous():
        param.data = param.data.contiguous()

# Initialize DeepSpeed
model_engine, _ , _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.vision_model.parameters(),
    config=CONFIG["deepspeed"]["config"],
    optimizer=optimizer,
)
logging.info(f"DeepSpeed configuration: {model_engine.config}")
torch.cuda.empty_cache()

# Train the models
logging.info("start training.")
train_clip_model(
    model_engine=model_engine,
    processor=processor,
    data_loader=data_loader,
    sampler=sampler,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=num_epochs,
    device=device,
    learning_rate=CONFIG["training"]["learning_rate"],
    dynamic_hard_negative=CONFIG["training"]["dynamic_hard_negative"],  
    initial_weight=CONFIG["training"]["hard_negative_weight"],              
    max_weight=CONFIG["training"]["hard_negative_max_weight"],  
)

# Push to hub
push_to_hub(
    model=model,
    processor=processor,
    repo_name=CONFIG["misc"]["repo_name"]
)
