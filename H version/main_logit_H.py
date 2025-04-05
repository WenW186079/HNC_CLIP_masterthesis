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
import copy

from load_data import LoadHNCPair, show_batches
from train_hnc_logit_H import train_clip_model
from val_H import evaluate_cosine_similarities, load_image

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
val_json_path   = CONFIG["dataset"]['val_json_path']
test_json_path  = CONFIG["dataset"]['test_json_path']
image_folder_path = CONFIG["dataset"]["image_folder_path"]

model_name = CONFIG["training"]["model_name"]
num_epochs = CONFIG["training"]["num_epochs"]
batch_size = CONFIG["deepspeed"]["config"]["train_batch_size"]
train_micro_batch_size_per_gpu = CONFIG["deepspeed"]["config"]["train_micro_batch_size_per_gpu"]

betas=(0.9, 0.98)
eps=1e-6


# Initialize CLIP Models
# Load CLIP model: "ViT-B/16","ViT-B/32","ViT-L/14","ViT-L/14@336px"
# openai/clip-vit-large-patch14-336
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# Load data
with open(train_json_path, 'r') as f:
    train_annotations = json.load(f)
train_dataset = LoadHNCPair(annotations=train_annotations,image_folder=image_folder_path)

# Load validation dataset
if val_json_path is not None:
    with open(val_json_path, 'r') as f:
        val_annotations = json.load(f)
    val_dataset = LoadHNCPair(annotations=val_annotations, image_folder=image_folder_path)
else:
    val_dataset = None

# Load test dataset
if test_json_path is not None:
    with open(test_json_path, 'r') as f:
        test_annotations = json.load(f)
    test_dataset = LoadHNCPair(annotations=test_annotations, image_folder=image_folder_path, is_test=True)
else:
    test_dataset = None

sampler = DistributedSampler(
    train_dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True
)
data_loader = DataLoader(
    train_dataset,
    batch_size=train_micro_batch_size_per_gpu, 
    sampler=sampler,
)

if dist.get_rank() == 0:
    logging.info(f"Train Dataset size: {len(train_dataset)}, Batch size: {train_micro_batch_size_per_gpu}")
    logging.info(f"Number of batches: {len(data_loader)}")
    if val_dataset is not None:
        logging.info(f"Validation Dataset size: {len(val_dataset)}")
    if test_dataset is not None:
        logging.info(f"Test Dataset size: {len(test_dataset)}")

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


# for param in model.parameters():
#     if not param.is_contiguous():
#         param.data = param.data.contiguous()

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
logging.info("=============start training============")
train_clip_model(
    model_engine=model_engine,
    processor=processor,
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
    val_dataset=val_dataset, 
    val_step_frequency=CONFIG["Val"]["val_step_frequency"],
    val_size=CONFIG["Val"]["val_size"],
    checkpoint_dir=CONFIG["training"]["checkpoint_dir"]
)

# Evaluate the pre–fine tuning (baseline) model on the test dataset.
if test_dataset is not None:
    # Set the pre-trained model to evaluation mode.
    model.eval()
    # Evaluate using a single batch by setting batch_size to the size of the test dataset.
    baseline_avg_pos, baseline_avg_neg, baseline_margin = evaluate_cosine_similarities(
        model, test_dataset, processor, device, batch_size=len(test_dataset)
    )
    print(f"Baseline Test Scores:")
    print(f"  Average Positive Cosine Similarity: {baseline_avg_pos:.4f}")
    print(f"  Average Negative Cosine Similarity: {baseline_avg_neg:.4f}")
    print(f"  Margin (Positive - Negative): {baseline_margin:.4f}")
   

# Evaluate the fine-tuned model on the test dataset.
if test_dataset is not None:
    model_engine.eval()  # Set fine-tuned model to evaluation mode.
    final_avg_pos, final_avg_neg, final_margin = evaluate_cosine_similarities(
        model_engine, test_dataset, processor, device, batch_size=len(test_dataset)
    )
    print(f"Final Test Scores after Fine-Tuning:")
    print(f"  Average Positive Cosine Similarity: {final_avg_pos:.4f}")
    print(f"  Average Negative Cosine Similarity: {final_avg_neg:.4f}")
    print(f"  Margin (Positive - Negative): {final_margin:.4f}")
    