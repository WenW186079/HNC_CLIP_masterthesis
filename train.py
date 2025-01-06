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
        for batch_idx, (images, tokenized_captions, labels, sources) in enumerate(train_loader):
            images, tokenized_captions, labels = (
                images.to(device), tokenized_captions.to(device), labels.to(device)
            )

            v_i = model.encode_image(images)
            u_i = model.encode_text(tokenized_captions)

            # Separate positive and negative samples
            pos_indices = torch.where(torch.tensor(sources) == "pos")[0]
            hnc_indices = torch.where(torch.tensor(sources) == "hnc")[0]
            random_indices = torch.where(torch.tensor(sources) == "random")[0]

            v_i_pos = v_i[pos_indices]
            u_i_pos = u_i[pos_indices]

            v_i_neg_hnc = v_i[hnc_indices]
            u_i_neg_hnc = u_i[hnc_indices]

            v_i_neg_rand = v_i[random_indices]
            u_i_neg_rand = u_i[random_indices]

            # Compute loss
            loss = criterion(v_i_pos, u_i_pos, u_i_neg_hnc, u_i_neg_rand, {name: param for name, param in model.named_parameters()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/{epochs}] Average Training Loss: {avg_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, tokenized_captions, labels, sources in val_loader:
                images, tokenized_captions, labels = (
                    images.to(device), tokenized_captions.to(device), labels.to(device)
                )

                v_i = model.encode_image(images)  # Image embeddings
                u_i = model.encode_text(tokenized_captions)  # Text embeddings

                # Separate positive and negative samples based on `sources`
                pos_indices = [i for i, s in enumerate(sources) if s == "pos"]
                hnc_indices = [i for i, s in enumerate(sources) if s == "hnc"]
                random_indices = [i for i, s in enumerate(sources) if s == "random"]

                v_i_pos = v_i[pos_indices]
                u_i_pos = u_i[pos_indices]

                v_i_neg_hnc = v_i[hnc_indices]
                u_i_neg_hnc = u_i[hnc_indices]

                v_i_neg_rand = v_i[random_indices]
                u_i_neg_rand = u_i[random_indices]

                # Calculate loss
                loss = criterion(
                    v_i_pos, u_i_pos, u_i_neg_hnc, u_i_neg_rand, {name: param for name, param in model.named_parameters()}
                )
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
