import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

def compute_cosine_similarities(model, pixel_values, pos_text, neg_text):
    with torch.cuda.amp.autocast():
        image_features = model.encode_image(pixel_values)
        pos_features = model.encode_text(pos_text)
        neg_features = model.encode_text(neg_text)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
    neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)

    pos_cos_sim = F.cosine_similarity(image_features, pos_features, dim=-1)
    neg_cos_sim = F.cosine_similarity(image_features, neg_features, dim=-1)

    return pos_cos_sim, neg_cos_sim

def get_caption_ratios(model, eval_loader, device, epsilon=1e-8):
    ratios = []
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            pixel_values = batch["pixel_values"].to(device)
            pos_text = batch["pos_text"].to(device)
            neg_text = batch["neg_text"].to(device)

            pos_cos_sim, neg_cos_sim = compute_cosine_similarities(model, pixel_values, pos_text, neg_text)
            batch_ratios = (pos_cos_sim / (neg_cos_sim + epsilon)).cpu().tolist()
            ratios.extend(batch_ratios)
    return ratios
    
def evaluate_cosine_similarities(model, eval_loader, device):
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

def evaluate_cosine_similarities_and_plot(
    model,
    eval_loader,
    device,
    plot_title="Cosine Similarity: POS+NEG",
    figsize=(12, 6),
    save_plot=False,
    plot_path="similarity_matrix.png"
    ):
    
    model.eval()

    pos_similarities = []
    neg_similarities = []
    
    all_img_embs = []
    all_pos_embs = []
    all_neg_embs = []

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
            
            all_img_embs.append(image_features.cpu())
            all_pos_embs.append(pos_features.cpu())
            all_neg_embs.append(neg_features.cpu())

    avg_pos = sum(pos_similarities) / len(pos_similarities)
    avg_neg = sum(neg_similarities) / len(neg_similarities)
    margin = avg_pos - avg_neg

    all_img_embs = torch.cat(all_img_embs, dim=0).float()  # Shape: [N, D]
    all_pos_embs = torch.cat(all_pos_embs, dim=0).float()  # Shape: [N, D]
    all_neg_embs = torch.cat(all_neg_embs, dim=0).float()  # Shape: [N, D]

    pos_matrix = all_img_embs @ all_pos_embs.t()  # Shape: [N, N] for image vs. positive texts.
    neg_matrix = all_img_embs @ all_neg_embs.t()  # Shape: [N, N] for image vs. negative texts.

    combined_matrix = torch.cat([pos_matrix, neg_matrix], dim=1)  # [N, 2N]

    plt.figure(figsize=figsize)
    im = plt.imshow(combined_matrix.numpy(), cmap="viridis")
    plt.title(plot_title)
    plt.xlabel("Text Samples (first N = Positives, second N = Negatives)")
    plt.ylabel("Image Samples (N)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_path, dpi=200)

    plt.show()
    plt.close()

    return avg_pos, avg_neg, margin, combined_matrix

def evaluate_cosine_similarities_random_negtive(model, eval_loader, device):
    model.eval()
    pos_similarities = []
    neg_similarities = []
    
    image_emb_list = []
    pos_emb_list = []
    image_path_list = []  

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
            pos_features   = pos_features   / pos_features.norm(dim=-1, keepdim=True)
            neg_features   = neg_features   / neg_features.norm(dim=-1, keepdim=True)
            
            pos_cos_sim = F.cosine_similarity(image_features, pos_features, dim=-1)
            neg_cos_sim = F.cosine_similarity(image_features, neg_features, dim=-1)
            
            pos_similarities.extend(pos_cos_sim.cpu().tolist())
            neg_similarities.extend(neg_cos_sim.cpu().tolist())
            
            image_emb_list.append(image_features.cpu())
            pos_emb_list.append(pos_features.cpu())
            
            if "image_path" in batch:
                image_path_list.extend(batch["image_path"])
    
    avg_pos = sum(pos_similarities) / len(pos_similarities)
    avg_neg = sum(neg_similarities) / len(neg_similarities)
    margin  = avg_pos - avg_neg
    
    images = torch.cat(image_emb_list, dim=0).float()   
    positives = torch.cat(pos_emb_list, dim=0).float()     
    N = images.size(0)
    
    if image_path_list:
        seen = {}
        unique_indices = []
        for idx, path in enumerate(image_path_list):
            if path not in seen:
                seen[path] = True
                unique_indices.append(idx)
        
        # Apply deduplication
        images = images[unique_indices]
        positives = positives[unique_indices]
        N = images.size(0)  # update N after deduplication
    
    # random permutation for random mismatched pairing
    perm = torch.randperm(N)
    # avoid pairing an image with its own positive.
    for i in range(N):
        if perm[i] == i:
            swap_idx = 0 if i == N - 1 else i + 1
            perm[i], perm[swap_idx] = perm[swap_idx], perm[i]
    
    rand_cos_sims = (images * positives[perm]).sum(dim=-1)
    avg_rand_neg = rand_cos_sims.mean().item()
    
    return avg_pos, avg_neg, avg_rand_neg, margin

def evaluate_caption_accuracy(model, eval_loader, device, threshold=1.0):
    model.eval()
    ratios = []       
    num_correct = 0   
    total_samples = 0
    epsilon = 1e-8   

    with torch.no_grad():
        for batch in eval_loader:
            pixel_values = batch["pixel_values"].to(device)
            pos_text = batch["pos_text"].to(device)
            neg_text = batch["neg_text"].to(device)
            
            pos_cos_sim, neg_cos_sim = compute_cosine_similarities(model, pixel_values, pos_text, neg_text)
            
            batch_ratios = (pos_cos_sim / (neg_cos_sim + epsilon)).cpu().tolist()
            ratios.extend(batch_ratios)
            
            for ratio in batch_ratios:
                total_samples += 1
                if ratio >= threshold:
                    num_correct += 1

    accuracy = num_correct / total_samples if total_samples > 0 else 0
    return accuracy, ratios
