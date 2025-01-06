import os
import json
import random
from PIL import Image  
import torch
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms  
from torch.optim import AdamW
import huggingface_hub
import logging
import clip

from load_data import HNCCLIPDataset, load_data_pairs 
from Loss_func import HNC_Loss

# logger 
logger = logging.getLogger("HNC_CLIP_Logger")

def train_clip_with_hnc_loss(model, train_loader, val_loader, criterion, optimizer, device="cuda", epochs=1):
    model.to(device)
    #print(model)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        logger.info(f"Starting epoch {epoch + 1}/{epochs}...")

        # Training
        for batch_idx, (images, tokenized_pos, tokenized_hnc_neg, in_batch_neg) in enumerate(train_loader):
            images = images.to(device)
            tokenized_pos = tokenized_pos.to(device)
            tokenized_hnc_neg = tokenized_hnc_neg.to(device)
            in_batch_neg = [neg.to(device) for neg in in_batch_neg]

            v_i = model.encode_image(images)
            u_i_pos = model.encode_text(tokenized_pos)
            u_i_hnc_neg = model.encode_text(tokenized_hnc_neg)

            u_i_batch_neg = torch.cat([model.encode_text(neg) for neg in in_batch_neg])

            loss = criterion(v_i, u_i_pos, u_i_hnc_neg, u_i_batch_neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/{epochs}] Average Training Loss: {avg_loss:.4f}")



        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/{epochs}] Average Training Loss: {avg_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, tokenized_pos, tokenized_hnc_neg, in_batch_neg in val_loader:
                # Move data to device
                images = images.to(device)
                tokenized_pos = tokenized_pos.to(device)
                tokenized_hnc_neg = tokenized_hnc_neg.to(device)
                in_batch_neg = [neg.to(device) for neg in in_batch_neg]

                # Get image and text embeddings
                v_i = model.encode_image(images)  # Image embeddings
                u_i_pos = model.encode_text(tokenized_pos)  # Positive text embeddings
                u_i_hnc_neg = model.encode_text(tokenized_hnc_neg)  # HNC negative embeddings
                u_i_batch_neg = torch.cat([model.encode_text(neg) for neg in in_batch_neg])  # In-batch negatives

                # Calculate validation loss
                loss = criterion(v_i, u_i_pos, u_i_hnc_neg, u_i_batch_neg)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Epoch [{epoch + 1}/{epochs}] Average Validation Loss: {avg_val_loss:.4f}")
        model.train()


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
