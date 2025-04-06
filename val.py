import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
import random


def evaluate_cosine_similarities(model, eval_loader, device):
    """
    Evaluates the average cosine similarity for image-positive and image-negative pairs 
    using the provided evaluation DataLoader.
    
    Args:
        model: The CLIP model.
        eval_loader: DataLoader built from the evaluation dataset.
        device: Device (e.g., "cuda") for inference.
        
    Returns:
        avg_pos: Average cosine similarity for image-positive pairs.
        avg_neg: Average cosine similarity for image-negative pairs.
        margin: Difference between avg_pos and avg_neg.
    """
    model.eval()
    pos_similarities = []
    neg_similarities = []

    with torch.no_grad():
        for batch in eval_loader:
            pixel_values = batch["pixel_values"].to(device)
            pos_text = batch["pos_text"].to(device)
            neg_text = batch["neg_text"].to(device)

            with torch.cuda.amp.autocast():
                image_features = model.encode_image(pixel_values)
                pos_features = model.encode_text(pos_text)
                neg_features = model.encode_text(neg_text)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)

            pos_cos_sim = F.cosine_similarity(image_features, pos_features, dim=-1)
            neg_cos_sim = F.cosine_similarity(image_features, neg_features, dim=-1)

            pos_similarities.extend(pos_cos_sim.cpu().tolist())
            neg_similarities.extend(neg_cos_sim.cpu().tolist())

    avg_pos = sum(pos_similarities) / len(pos_similarities)
    avg_neg = sum(neg_similarities) / len(neg_similarities)
    margin = avg_pos - avg_neg

    return avg_pos, avg_neg, margin