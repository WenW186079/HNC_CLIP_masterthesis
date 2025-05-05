import random
import numpy as np
import torch
import clip
from PIL import Image

from evaluation import load_finetuned_clip_model, set_determinism, format_model_name

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from load_data import load_split

# -----------------------------------------------
CHECKPOINT_PATH = "./models/HNC_1_full_small_ref/epoch_1_full_encoder.pt"  
TEST_JSON       = "./data/HNC/hnc_clean_strict_test.json"
IMAGES_PATH     = "./data/gqa_dataset/images/images"
LOADER_TYPE     = "hnc"  # "hnc" or "coco"
BATCH_SIZE      = 32
FIRST_K         = 10
FINETUNE_MODE   = "full_encoder"  # "text_encoder", "vision_encoder", or "full_encoder"
# -----------------------------------------------

def main():
    set_determinism(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_finetuned_clip_model("ViT-B/32",CHECKPOINT_PATH,device,FINETUNE_MODE)

    result = load_split(TEST_JSON,'test', IMAGES_PATH, clip.tokenize, preprocess, BATCH_SIZE, subset_size=None,loader_type=LOADER_TYPE)
    test_loader, test_dataset = result if len(result) == 2 else (result[0], result[1])

    ds = getattr(test_dataset, 'dataset', test_dataset)
    if not hasattr(ds, 'data_pairs'):
        raise AttributeError("Dataset has no 'data_pairs'")

    raw_pairs = ds.data_pairs[:FIRST_K]
    if len(raw_pairs) < FIRST_K:
        print(f"Warning: dataset not enough")

    images = []
    pos_captions = []
    neg_captions = []
    paths = []
    for path, pos_caption, neg_caption in raw_pairs:
        img = Image.open(path).convert('RGB')
        images.append(preprocess(img))
        pos_captions.append(pos_caption)
        neg_captions.append(neg_caption)
        paths.append(path)
    images = torch.stack(images).to(device)

    with torch.no_grad():
        img_feats = model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=1, keepdim=True)

        pos_tokens = clip.tokenize(pos_captions).to(device)
        txt_pos = model.encode_text(pos_tokens)
        txt_pos = txt_pos / txt_pos.norm(dim=1, keepdim=True)
        pos_scores = (img_feats * txt_pos).sum(dim=1).cpu().tolist()

        neg_tokens = clip.tokenize(neg_captions).to(device)
        txt_neg = model.encode_text(neg_tokens)
        txt_neg = txt_neg / txt_neg.norm(dim=1, keepdim=True)
        neg_scores = (img_feats * txt_neg).sum(dim=1).cpu().tolist()

    print(f"\nFirst {FIRST_K} samples and their scores for model '{format_model_name(CHECKPOINT_PATH or "ViT-B/32")}':\n")
    for i in range(len(raw_pairs)):
        print(f"{i+1:2d}. Image Path: {paths[i]}")
        print(f"     Pos: \"{pos_captions[i]}\"  (score = {pos_scores[i]:.4f})")
        print(f"     Neg: \"{neg_captions[i]}\"  (score = {neg_scores[i]:.4f})\n")

if __name__ == "__main__":
    main()
