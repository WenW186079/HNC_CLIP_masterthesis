import yaml
import logging
import json
import os
import wandb
import torch
import deepspeed
import deepspeed.comm as dist 
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import argparse
import copy
import clip
import numpy as np
import random

from load_data import load_split, split_train_val
from trainCLIP import train_clip_model,set_trainable_parameters
from eval.eval_functions import evaluate_cosine_similarities


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
val_json_path   = CONFIG["dataset"].get('val_json_path', None)
test_json_path  = CONFIG["dataset"].get('test_json_path', None)
image_folder_path = CONFIG["dataset"]["image_folder_path"]

model_name = CONFIG["training"]["model_name"]
num_epochs = CONFIG["training"]["num_epochs"]
batch_size = CONFIG["deepspeed"]["config"]["train_batch_size"]
train_micro_batch_size_per_gpu = CONFIG["deepspeed"]["config"]["train_micro_batch_size_per_gpu"]
finetune_mode = CONFIG["training"]['finetune_mode']

betas=(0.9, 0.98)
eps=1e-6

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_name, device=device)
tokenizer = clip.tokenize

teacher_model = copy.deepcopy(model).eval().to(device)
for param in teacher_model.parameters():
    param.requires_grad = False


if CONFIG["Val"]["split"] == True:
    print("==========Splitting the training dataset to create a validation set==========")
    train_loader, full_train_dataset = load_split(train_json_path, "train", image_folder_path, tokenizer, preprocess, batch_size, subset_size=CONFIG["training"]["train_size"])
    train_dataset, val_dataset = split_train_val(full_train_dataset, val_size=1000, seed=42)

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )
    data_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["deepspeed"]["config"]["train_micro_batch_size_per_gpu"],
        sampler=sampler,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    print(f"New Train Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(val_dataset)}")
else:
    train_loader, train_dataset = load_split(train_json_path, "train", image_folder_path, tokenizer, preprocess, batch_size, subset_size=CONFIG["training"]["train_size"])
    val_loader, val_dataset = load_split(val_json_path, "val", image_folder_path, tokenizer, preprocess, batch_size, subset_size=CONFIG["Val"]["val_size"])
    print(f"Train Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(val_dataset)}")
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )
    data_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["deepspeed"]["config"]["train_micro_batch_size_per_gpu"],
        sampler=sampler,
        drop_last=True
    )

if test_json_path is not None:
    test_loader, test_dataset = load_split(test_json_path, "test", image_folder_path, tokenizer, preprocess, batch_size, subset_size=None)
    print(f"Test Dataset size: {len(test_dataset)}")

set_trainable_parameters(model, finetune_mode)
trainable_params = [p for p in model.parameters() if p.requires_grad]

# Define the optimizer using AdamW
optimizer = AdamW(
    trainable_params,
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


# Initialize DeepSpeed
model_engine, _ , _, _ = deepspeed.initialize(
    model=model,
    model_parameters=trainable_params,
    config=CONFIG["deepspeed"]["config"],
    optimizer=optimizer,
)
logging.info(f"DeepSpeed configuration: {model_engine.config}")
torch.cuda.empty_cache()


# Train the models
print("=============start training============")
train_clip_model(
    model_engine=model_engine,
    data_loader=data_loader,
    sampler=sampler,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=num_epochs,
    device=device,
    learning_rate=CONFIG["training"]["learning_rate"], 
    mode=CONFIG["training"]["mode"],  
    lambda_ref=CONFIG["training"]["lambda_ref"],
    hard_neg_weight = CONFIG["training"]["hard_neg_weight"],
    dynamic_weight=CONFIG["training"]["dynamic_weight"],
    min_weight=CONFIG["training"]["min_weight"],
    max_weight=CONFIG["training"]["max_weight"],
    update_interval=CONFIG["training"]["update_interval"], 
    num_updates=CONFIG["training"]["num_updates"], 
    dpo_beta=CONFIG["training"]["dpo_beta"],
    combined_alpha=0.5, 
    val_loader=val_loader, 
    val_step_frequency=CONFIG["Val"]["val_step_frequency"],
    checkpoint_dir=CONFIG["training"]["checkpoint_dir"],
    finetune_mode=CONFIG["training"]["finetune_mode"],
    teacher_model=teacher_model,
)
print("=============end training============")
