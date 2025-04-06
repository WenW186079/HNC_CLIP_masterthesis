import os
import logging
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader, DistributedSampler,Subset
import random
import torch.distributed as dist
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class LoadCLIPDataset(Dataset):
    def __init__(self, annotations, image_folder, tokenizer, transform, is_test=False):
        """
        Initializes the dataset by creating (image_path, positive_caption, negative_caption) pairs.
        
        annotations: a dictionary where each key corresponds to an image and its associated caption info.
        image_folder: folder containing the images.
        tokenizer: tokenizer function from OpenCLIP.
        transform: image transformation (preprocessing) to apply.
        is_test: whether to use test logic (grouping by caption type) or training logic.
        """
        self.annotations = annotations
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_test = is_test
        self.data_pairs = []
        self.hnc_count = 0
        self.positive_count = 0

        missing_images_count = 0

        for img_key, data in annotations.items():
            if self.is_test:
                image_filename = img_key
                captions_dict = data
            else:
                image_filename = f"{img_key}.jpg"
                captions_dict = data.get("captions", {})

            image_path = os.path.join(self.image_folder, image_filename)
            if os.path.exists(image_path):
                if self.is_test:
                    # Group captions by type.
                    type_groups = {}
                    for cap_id, cap_data in captions_dict.items():
                        cap_type = cap_data.get("type", "default")
                        type_groups.setdefault(cap_type, []).append((cap_id, cap_data))
                    
                    for cap_type, cap_list in type_groups.items():
                        # Separate positive and negative captions.
                        pos_list = [(cap_id, cap_data) for cap_id, cap_data in cap_list if cap_data["label"] == 1]
                        neg_list = [(cap_id, cap_data) for cap_id, cap_data in cap_list if cap_data["label"] == 0]
                        # Sort by caption id (assuming numeric strings).
                        try:
                            pos_list.sort(key=lambda x: int(x[0]))
                            neg_list.sort(key=lambda x: int(x[0]))
                        except ValueError:
                            pass

                        if len(pos_list) != len(neg_list):
                            logger.warning(f"Unequal positives and negatives for {image_filename} in type '{cap_type}'. Skipping.")
                            continue

                        for ((pos_cap_id, pos_data), (neg_cap_id, neg_data)) in zip(pos_list, neg_list):
                            pos_caption = pos_data["caption"]
                            neg_caption = neg_data["caption"]
                            self.data_pairs.append((image_path, pos_caption, neg_caption))
                            self.hnc_count += 1
                            self.positive_count += 1
                else:
                    # Training logic: pair negatives with corresponding positive captions using cpt_p_id.
                    pos_caption_map = {
                        cap_id: cap_data["caption"]
                        for cap_id, cap_data in captions_dict.items() if cap_data["label"] == 1
                    }
                    self.positive_count += len(pos_caption_map)
                    for cap_id, cap_data in captions_dict.items():
                        if cap_data["label"] == 0:
                            neg_caption = cap_data["caption"]
                            cpt_p_id = cap_data.get("cpt_p_id")
                            if not cpt_p_id:
                                logger.warning(f"Missing cpt_p_id for cap_id {cap_id} in {image_filename}. Skipping.")
                                continue
                            cpt_p_id = str(cpt_p_id)
                            if cpt_p_id not in pos_caption_map:
                                logger.warning(f"Invalid cpt_p_id {cpt_p_id} in image {image_filename}.")
                            else:
                                pos_caption = pos_caption_map[cpt_p_id]
                                self.data_pairs.append((image_path, pos_caption, neg_caption))
                                self.hnc_count += 1
            else:
                missing_images_count += 1
                logger.warning(f"Missing image: {image_path}")

        logger.info(f"Finished creating pairs. Total pairs: {len(self.data_pairs)}. Missing images: {missing_images_count}.")
        logger.info(f"Total Positive samples: {self.positive_count}")
        logger.info(f"Total HNC samples: {self.hnc_count}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, pos_caption, neg_caption = self.data_pairs[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
    
        pos_tokens = self.tokenizer([pos_caption])[0]
        neg_tokens = self.tokenizer([neg_caption])[0]
        return {
            "image_path": image_path,
            "pixel_values": image,    # image tensor (preprocessed)
            "pos_text": pos_tokens,     # positive caption tokens
            "neg_text": neg_tokens      # negative caption tokens
        }

def get_dataset(json_path, image_folder_path, tokenizer, transform, train_batch_size, dataset='train', subset_size=None):
    """
    Load dataset for training, validation, or testing.
    """
    is_test = (dataset == 'test')
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    ds = LoadCLIPDataset(annotations=annotations,
                     image_folder=image_folder_path,
                     tokenizer=tokenizer,
                     transform=transform,
                     is_test=is_test)
    
    if dataset != 'train' and subset_size is not None:
        if len(ds) > subset_size:
            indices = random.sample(range(len(ds)), subset_size)
            ds = Subset(ds, indices)
    
    if dataset == 'train':
        sampler = DistributedSampler(
            ds,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        ) if dist.is_initialized() else None

        loader = DataLoader(
            ds,
            batch_size=train_batch_size,
            sampler=sampler,
            shuffle=(sampler is None)
        )
    else:
        loader = DataLoader(
            ds,
            batch_size=train_batch_size,
            shuffle=False
        )
    
    logging.info(f"Dataset '{dataset}' loaded with {len(ds)} samples.")
    return loader, ds

def load_split(json_path, dataset_type, image_folder_path, tokenizer, transform, batch_size, subset_size=None):
    if json_path is None:
        return None, None
    return get_dataset(
        json_path=json_path,
        image_folder_path=image_folder_path,
        tokenizer=tokenizer,
        transform=transform,
        train_batch_size=batch_size,
        dataset=dataset_type,
        subset_size=subset_size
    )

def deduplicate_batch(batch, device):
    """
    Removes duplicate images from a batch based on image_path and concatenates 
    the corresponding positive and negative text tokens.
    
    Args:
        batch (dict): Batch dictionary with keys "image_path", "pixel_values", "pos_text", "neg_text".
        device: The device (e.g., "cuda") to which the tensors are moved.
        
    Returns:
        images (torch.Tensor): A tensor of unique, preprocessed images.
        text_inputs (torch.Tensor): A tensor obtained by concatenating positive and negative text tokens.
    """
    seen = set()
    unique_pixel_values = []
    unique_pos_texts = []
    unique_neg_texts = []

    # Use the already preprocessed image tensors.
    for i, path in enumerate(batch["image_path"]):
        if path not in seen:
            seen.add(path)
            unique_pixel_values.append(batch["pixel_values"][i])
            unique_pos_texts.append(batch["pos_text"][i])
            unique_neg_texts.append(batch["neg_text"][i])
    
    images = torch.stack(unique_pixel_values).to(device)
    pos_text_inputs = torch.stack(unique_pos_texts).to(device)
    neg_text_inputs = torch.stack(unique_neg_texts).to(device)
    text_inputs = torch.cat([pos_text_inputs, neg_text_inputs], dim=0)

    return images, text_inputs




class LoadCOCOPair(Dataset):
    def __init__(self, annotations, image_folder):
        self.annotations = annotations
        self.image_folder = image_folder
        self.data_pairs = [] 
        self.neg_count = 0
        self.positive_count = 0

        logger.info("Creating COCO image-POS-NEG pairs...")
        missing_images_count = 0

        for data in annotations:
            image_filename = data.get("image")
            image_path = os.path.join(self.image_folder, image_filename)
            
            if os.path.exists(image_path):
                pos_caption = data.get("text")  
                neg_caption = data.get("neg_text")  
                
                if pos_caption and neg_caption:
                    self.data_pairs.append((image_path, pos_caption, neg_caption, "hnc"))
                    self.positive_count += 1
                    self.neg_count += 1
            else:
                missing_images_count += 1
                logger.warning(f"❗️Missing image: {image_path}")

        logger.info(f"Finished creating pairs. Total pairs: {len(self.data_pairs)}. Missing images: {missing_images_count}.")
        logger.info(f"Total Pos samples: {self.positive_count}")
        logger.info(f"Total COCONEG samples: {self.neg_count}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        image_path, pos_caption, neg_caption, source = self.data_pairs[idx]
        
        return image_path, pos_caption, neg_caption

