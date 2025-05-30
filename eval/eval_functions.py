import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import pandas as pd

def compute_cosine_similarities(
    model,
    batch: dict,
    device: torch.device,
    return_embeddings: bool = False
):
    imgs = batch["pixel_values"].to(device)
    pos_txts = batch["pos_text"].to(device)
    neg_txts = batch["neg_text"].to(device)

    with torch.cuda.amp.autocast():
        img_feats = model.encode_image(imgs)
        pos_feats = model.encode_text(pos_txts)
        neg_feats = model.encode_text(neg_txts)

    img_feats = F.normalize(img_feats, dim=-1)
    pos_feats = F.normalize(pos_feats, dim=-1)
    neg_feats = F.normalize(neg_feats, dim=-1)

    pos_cos = F.cosine_similarity(img_feats, pos_feats, dim=-1)
    neg_cos = F.cosine_similarity(img_feats, neg_feats, dim=-1)

    if return_embeddings:
        return pos_cos, neg_cos, img_feats, pos_feats, neg_feats
    else:
        return pos_cos, neg_cos


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
    pos_sims, neg_sims = [], []

    with torch.no_grad():
        for batch in eval_loader:
            pos_cos, neg_cos = compute_cosine_similarities(
                model, batch, device
            )
            pos_sims.extend(pos_cos.cpu().tolist())
            neg_sims.extend(neg_cos.cpu().tolist())

    avg_pos = sum(pos_sims) / len(pos_sims)
    avg_neg = sum(neg_sims) / len(neg_sims)
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
    pos_sims, neg_sims = [], []
    all_img_embs, all_pos_embs, all_neg_embs = [], [], []
    
    with torch.no_grad():
        for batch in eval_loader:
            pos_cos, neg_cos, img_feats, pos_feats, neg_feats = compute_cosine_similarities(
                    model, batch, device, return_embeddings=True
                )
            pos_sims.extend(pos_cos.cpu().tolist())
            neg_sims.extend(neg_cos.cpu().tolist())

            all_img_embs.append(img_feats.cpu())
            all_pos_embs.append(pos_feats.cpu())
            all_neg_embs.append(neg_feats.cpu())

    avg_pos = sum(pos_sims) / len(pos_sims)
    avg_neg = sum(neg_sims) / len(neg_sims)
    margin = avg_pos - avg_neg

    all_img_embs = torch.cat(all_img_embs, dim=0).float()  
    all_pos_embs = torch.cat(all_pos_embs, dim=0).float()  
    all_neg_embs = torch.cat(all_neg_embs, dim=0).float()  

    pos_matrix = all_img_embs @ all_pos_embs.t()  
    neg_matrix = all_img_embs @ all_neg_embs.t()  
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

def evaluate_cosine_similarities_random_negative(model, eval_loader, device):
    model.eval()
    pos_sims, neg_sims = [], []
    all_img_embs, all_pos_embs = [], []
    paths = [] 

    with torch.no_grad():
        for batch in eval_loader:
            pos_cos, neg_cos, img_feats, pos_feats, _ = compute_cosine_similarities(
                model, batch, device, return_embeddings=True
            )
            pos_sims.extend(pos_cos.cpu().tolist())
            neg_sims.extend(neg_cos.cpu().tolist())

            all_img_embs.append(img_feats.cpu())
            all_pos_embs.append(pos_feats.cpu())
            paths.extend(batch.get("image_path", []))
    
    avg_pos = sum(pos_sims) / len(pos_sims)
    avg_neg = sum(neg_sims) / len(neg_sims)
    margin  = avg_pos - avg_neg
    
    images = torch.cat(all_img_embs, dim=0).float()   
    positives = torch.cat(all_pos_embs, dim=0).float()     

    if paths:
        seen, uniq_idxs = set(), []
        for i, p in enumerate(paths):
            if p not in seen:
                seen.add(p)
                uniq_idxs.append(i)
        images    = images[uniq_idxs]
        positives = positives[uniq_idxs]

    N = images.size(0)  
    
    torch.manual_seed(42)
    perm = torch.randperm(N)
    
    # avoid pairing an image with its own positive.
    for i in range(N):
        if perm[i] == i:
            swap_idx = 0 if i == N - 1 else i + 1
            perm[i], perm[swap_idx] = perm[swap_idx], perm[i]
    
    rand_cos_sims = (images * positives[perm]).sum(dim=-1)
    avg_rand_neg = rand_cos_sims.mean().item()
    
    return avg_pos, avg_neg, avg_rand_neg, margin

def evaluate_thresholds_accuracy(model, eval_loader, device,
                                 thresholds=[1, 1.1, 1.2, 1.5, 2, 3]):

    model.eval()
    ratios = []
    epsilon=1e-8

    with torch.no_grad():
        for batch in eval_loader:
            pos_cos, neg_cos = compute_cosine_similarities(model, batch, device)
            ratios.extend((pos_cos / (neg_cos + epsilon)).cpu().tolist())

    total = len(ratios)
    accuracies = {}
    for τ in thresholds:
        correct = sum(1 for r in ratios if r >= τ)
        accuracies[τ] = correct / total if total > 0 else 0.0

    return accuracies

def evaluate_random_and_thresholds(
    model,
    eval_loader,
    device,
    thresholds=None
):
    epsilon=1e-8

    if thresholds is None:
        thresholds = [1, 1.1, 1.2, 1.5, 2, 3]

    avg_pos, avg_neg, avg_rand_neg, margin = evaluate_cosine_similarities_random_negative(
        model, eval_loader, device
    )
    accs = evaluate_thresholds_accuracy(model, eval_loader, device, thresholds)
    return avg_pos, avg_neg, avg_rand_neg, margin, accs

def evaluate_by_type_random_thresholds_total(model, type_wrapper, device, thresholds, batch_size):
    results = []
    all_loaders = type_wrapper.get_all_loaders(batch_size=batch_size, shuffle=False)

    total_count = 0
    agg = {
        'Avg_Pos': 0.0,
        'Avg_Neg': 0.0,
        'Avg_Rand_Neg': 0.0,
        'Margin': 0.0,
    }
    agg_acc = {τ: 0.0 for τ in thresholds}

    for type_name, (loader, count) in all_loaders.items():
        print(f"\n Evaluating type: {type_name} | Samples: {count}")
        avg_pos, avg_neg, avg_rand, margin, accs = evaluate_random_and_thresholds(
            model, loader, device, thresholds
        )

        row = {
            'Type': type_name,
            'Avg_Pos': round(avg_pos, 4),
            'Avg_Neg': round(avg_neg, 4),
            'Avg_Rand_Neg': round(avg_rand, 4),
            'Margin': round(margin, 4)
        }
        for τ in thresholds:
            row[f"Acc@{τ}"] = round(accs[τ], 4)
            agg_acc[τ] += accs[τ] * count

        agg['Avg_Pos'] += avg_pos * count
        agg['Avg_Neg'] += avg_neg * count
        agg['Avg_Rand_Neg'] += avg_rand * count
        agg['Margin'] += margin * count
        total_count += count

        results.append(row)

    if total_count > 0:
        avg_row = {
            'Type': 'ALL',
            'Avg_Pos': round(agg['Avg_Pos'] / total_count, 4),
            'Avg_Neg': round(agg['Avg_Neg'] / total_count, 4),
            'Avg_Rand_Neg': round(agg['Avg_Rand_Neg'] / total_count, 4),
            'Margin': round(agg['Margin'] / total_count, 4)
        }
        for τ in thresholds:
            avg_row[f"Acc@{τ}"] = round(agg_acc[τ] / total_count, 4)

        results.append(avg_row)

    df = pd.DataFrame(results)
    return df
