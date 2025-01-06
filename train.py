import os
import json
import random
from PIL import Image  
import torch
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms  
from torch.optim import AdamW
from torch.cuda.amp import autocast
import huggingface_hub
import logging
import clip
import deepspeed

from load_data import HNCCLIPDataset, load_data_pairs 
from Loss_func import HNC_Loss

def train_clip_with_hnc_loss(model, train_loader, val_loader, criterion, epochs=3, config_path="deepspeed_config.json"):

    local_rank = int(os.environ.get('LOCAL_RANK', 0))  
    device = torch.device(f'cuda:{local_rank}') 

    logger = logging.getLogger("HNC_CLIP_Logger")

    # DeepSpeed 初始化
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=config_path
    )
    
    for epoch in range(epochs):
        total_loss = 0
        logger.info(f"Starting epoch {epoch + 1}/{epochs}...")

        # Training loop
        for batch_idx, (images, tokenized_pos, tokenized_hnc_neg, in_batch_neg) in enumerate(train_loader):
            
            images = images.to(device)
            if images.dim() == 3:
                images = images.unsqueeze(0)  # 确保输入形状正确
            print(f"Images shape: {images.shape}")  # 应输出 [batch_size, 3, 224, 224]


            tokenized_pos = tokenized_pos.to(device)
            tokenized_hnc_neg = tokenized_hnc_neg.to(device)
            in_batch_neg = [neg.to(device) for neg in in_batch_neg]

            # 检查形状
            assert images.dim() == 4, f"Expected 4D input for images, got shape {images.shape}"
            assert tokenized_pos.dim() == 2, f"Expected 2D input for tokenized_pos, got shape {tokenized_pos.shape}"


            # 自动混合精度
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                v_i = model_engine.module.encode_image(images)
                print(f"Image embedding shape: {v_i.shape}")

                u_i_pos = model_engine.module.encode_text(tokenized_pos)
                u_i_hnc_neg = model_engine.module.encode_text(tokenized_hnc_neg)
                u_i_batch_neg = model_engine.module.encode_text(in_batch_neg)

                loss = criterion(v_i, u_i_pos, u_i_hnc_neg, u_i_batch_neg, {name: param for name, param in model_engine.named_parameters()})
            
            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN or Inf detected in loss at batch {batch_idx + 1}")

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/{epochs}] Average Training Loss: {avg_loss:.4f}")

        # Validation loop
        model_engine.module.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (images, tokenized_pos, tokenized_hnc_neg, in_batch_neg) in enumerate(val_loader):
                images, tokenized_pos, tokenized_hnc_neg = images.to(device), tokenized_pos.to(device), tokenized_hnc_neg.to(device)
                in_batch_neg = torch.stack(in_batch_neg).to(device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    v_i = model_engine.module.encode_image(images)
                    u_i_pos = model_engine.module.encode_text(tokenized_pos)
                    u_i_hnc_neg = model_engine.module.encode_text(tokenized_hnc_neg)
                    u_i_batch_neg = model_engine.module.encode_text(in_batch_neg)

                    if not torch.isfinite(v_i).all() or not torch.isfinite(u_i_pos).all():
                        raise RuntimeError(f"NaN detected in validation embeddings at batch {batch_idx + 1}")

                    loss = criterion(v_i, u_i_pos, u_i_hnc_neg, u_i_batch_neg, {name: param for name, param in model_engine.named_parameters()})
                    val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch [{epoch + 1}/{epochs}] Average Validation Loss: {avg_val_loss:.4f}")
        model_engine.module.train()

    return model_engine

def push_vision_encoder_to_hub(model, repo_name, hf_username, model_dir="hnc_clip_vision_encoder"):
    # Save only vision encoder weights
    vision_encoder_state_dict = {
        name: param for name, param in model.state_dict().items() if "visual" in name
    }

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(vision_encoder_state_dict, os.path.join(model_dir, "pytorch_model.bin"))

    # Create model card
    model_card = f"""
    # HNC_CLIP_ViT Vision Encoder

    This is a fine-tuned CLIP vision encoder (ViT) with a frozen text encoder for the HNC task.

    - Vision Encoder: Fine-tuned ViT architecture
    - Text Encoder: Frozen (unchanged)
    - Task: Hard Negative Captions (HNC)
    
    Fine-tuned by: {hf_username}
    """
    with open(os.path.join(model_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    # Push to Hugging Face Hub
    repo_url = f"{hf_username}/{repo_name}"
    huggingface_hub.create_repo(repo_url, exist_ok=True)
    huggingface_hub.push_to_hub(repo_id=repo_url, repo_type="model", local_dir=model_dir)
    print(f"Model pushed to Hugging Face Hub: https://huggingface.co/{repo_url}")
