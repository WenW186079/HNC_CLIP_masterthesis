import os
import json
import random
from PIL import Image  
import torch
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms  
from torch.optim import AdamW
import logging
import open_clip 

from load_data import HNCCLIPDataset, load_data_pairs 
from Loss_func import HNC_Loss

def train_clip_with_hnc_loss(model, train_loader, val_loader, criterion, optimizer, device="cuda", epochs=1):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        logger.info(f"Starting epoch {epoch + 1}/{epochs}...")

        # Training
        for batch_idx, (images, pos_texts, hnc_negs, random_negs, _) in enumerate(train_loader):
            images, pos_texts, hnc_negs, random_negs = (
                images.to(device), pos_texts.to(device), hnc_negs.to(device), random_negs.to(device)
            )

            v_i = model.encode_image(images)
            u_i_pos = model.encode_text(pos_texts)
            u_i_hnc_neg = model.encode_text(hnc_negs)
            u_i_rand_neg = model.encode_text(random_negs)

            loss = criterion(v_i, u_i_pos, u_i_hnc_neg, u_i_rand_neg, {name: param for name, param in model.named_parameters()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/{epochs}] Average Training Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, pos_texts, hnc_negs, random_negs, _ in val_loader:
                images, pos_texts, hnc_negs, random_negs = (
                    images.to(device), pos_texts.to(device), hnc_negs.to(device), random_negs.to(device)
                )

                v_i = model.encode_image(images)
                u_i_pos = model.encode_text(pos_texts)
                u_i_hnc_neg = model.encode_text(hnc_negs)
                u_i_rand_neg = model.encode_text(random_negs)

                loss = criterion(v_i, u_i_pos, u_i_hnc_neg, u_i_rand_neg, {name: param for name, param in model.named_parameters()})
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Epoch [{epoch + 1}/{epochs}] Average Validation Loss: {avg_val_loss:.4f}")
        model.train()

