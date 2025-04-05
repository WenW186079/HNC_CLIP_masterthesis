import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
import random

from load_data import LoadHNCPair


def load_image(image_path):
    """Loads an image from disk and converts it to RGB."""
    return Image.open(image_path).convert("RGB")

def evaluate_cosine_similarities(model, eval_dataset, processor, device, batch_size=32):
    model.eval()
    pos_similarities = []
    neg_similarities = []

    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in eval_loader:
            image_paths, positive_texts, hard_negative_texts = batch

            images = [load_image(path) for path in image_paths]
            # image_inputs = torch.stack([preprocess(image) for image in images]).to(device)
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            pos_text_inputs = processor(text=positive_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            neg_text_inputs = processor(text=hard_negative_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # pos_text_inputs = clip.tokenize(positive_texts).to(device)
            # neg_text_inputs = clip.tokenize(hard_negative_texts).to(device)

            image_features = model.get_image_features(**image_inputs)
            pos_features = model.get_text_features(**pos_text_inputs)
            neg_features = model.get_text_features(**neg_text_inputs)

            # image_features = model.encode_image(image_inputs)
            # pos_features = model.encode_text(pos_text_inputs)
            # neg_features = model.encode_text(neg_text_inputs)
            
            # Normalize the features so cosine similarity is equivalent to the dot product.
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarities.
            pos_cos_sim = F.cosine_similarity(image_features, pos_features, dim=-1)
            neg_cos_sim = F.cosine_similarity(image_features, neg_features, dim=-1)
            
            pos_similarities.extend(pos_cos_sim.cpu().tolist())
            neg_similarities.extend(neg_cos_sim.cpu().tolist())
    
    avg_pos = sum(pos_similarities) / len(pos_similarities)
    avg_neg = sum(neg_similarities) / len(neg_similarities)
    margin = avg_pos - avg_neg

    return avg_pos, avg_neg, margin
