import os
import random
import json
import logging
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader, DistributedSampler,Subset
import torch.distributed as dist
from collections import defaultdict, Counter

      
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class LoadCLIPDataset(Dataset):
    def __init__(self, annotations, image_folder, tokenizer, transform, is_test=False):
        self.annotations = annotations
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_test = is_test
        self.data_pairs = []
        self.hnc_count = 0
        self.positive_count = 0
        self.negative_count = 0
        self.skipped_pos = 0
        self.skipped_neg = 0
        missing_images_count = 0

        for img_key, data in annotations.items():
            image_filename = img_key if is_test else f"{img_key}.jpg"
            captions_dict = data if is_test else data.get("captions", {})
            image_path = os.path.join(self.image_folder, image_filename)

            if not os.path.exists(image_path):
                missing_images_count += 1
                logger.warning(f"Missing image: {image_path}")
                continue

            if is_test:
                sorted_caps = sorted(captions_dict.items(), key=lambda x: int(x[0]))
                last_pos = None

                for _, cap_data in sorted_caps:
                    label = cap_data.get("label")
                    caption = cap_data.get("caption")
                    if not caption:
                        continue

                    if label == 1:
                        self.positive_count += 1
                        last_pos = caption
                    elif label == 0:
                        self.negative_count += 1
                        if last_pos:
                            self.data_pairs.append((image_path, last_pos, caption))
                            self.hnc_count += 1
                            last_pos = None
                        else:
                            self.skipped_neg += 1

                if last_pos and (len(self.data_pairs) == 0 or self.data_pairs[-1][1] != last_pos):
                    self.skipped_pos += 1

            else:
                pos_caption_map = {
                    cap_id: cap_data["caption"]
                    for cap_id, cap_data in captions_dict.items()
                    if cap_data.get("label") == 1
                }
                self.positive_count += len(pos_caption_map)

                for cap_id, cap_data in captions_dict.items():
                    if cap_data.get("label") == 0:
                        self.negative_count += 1
                        neg_caption = cap_data.get("caption")
                        cpt_p_id = str(cap_data.get("cpt_p_id", ""))
                        if not neg_caption or cpt_p_id not in pos_caption_map:
                            self.skipped_neg += 1
                            continue
                        pos_caption = pos_caption_map[cpt_p_id]
                        self.data_pairs.append((image_path, pos_caption, neg_caption))
                        self.hnc_count += 1

        logger.info(f"✅ Finished creating pairs. Total pairs: {len(self.data_pairs)}. Missing images: {missing_images_count}")
        logger.info(f"📊 Total positive captions: {self.positive_count}, Total negative captions: {self.negative_count}")
        logger.info(f"🚫 Skipped positives: {self.skipped_pos}, Skipped negatives: {self.skipped_neg}")
        logger.info(f"✅ HNC pairs created: {self.hnc_count}")

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
            "pixel_values": image,
            "pos_text": pos_tokens,
            "neg_text": neg_tokens
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
    
    if subset_size is not None and len(ds) > subset_size:
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
            shuffle=True
        )
    
    logging.info(f"Dataset '{dataset}' loaded with {len(ds)} samples.")
    return loader, ds

def load_split(json_path, dataset_type, image_folder_path, tokenizer, transform, batch_size, subset_size=None,loader_type="hnc"):
    if json_path is None:
        return None, None

    if loader_type.lower() == "coco":
        print('=============COCO dataset=========')
        return get_dataset_COCO(
            json_path=json_path,
            image_folder_path=image_folder_path,
            tokenizer=tokenizer,
            transform=transform,
            batch_size=batch_size,
            subset_size=subset_size
        )
    else:
        print('=============HNC dataset=========')
        return get_dataset(
            json_path=json_path,
            image_folder_path=image_folder_path,
            tokenizer=tokenizer,
            transform=transform,
            train_batch_size=batch_size,
            dataset=dataset_type,
            subset_size=subset_size
        )

def deduplicate_and_refill(batch, dataset, device, batch_size, mode=None):
    base_dataset = getattr(dataset, 'dataset', dataset)

    paths = batch["image_path"]
    pixs  = batch["pixel_values"]
    poss  = batch["pos_text"]
    negs  = batch["neg_text"]

    seen = set()
    unique_idxs = []
    
    for i, p in enumerate(paths):
        if p not in seen:
            seen.add(p)
            unique_idxs.append(i)

    unique_pix = [pixs[i] for i in unique_idxs]
    unique_pos = [poss[i] for i in unique_idxs]
    unique_neg = [negs[i] for i in unique_idxs]

    num_missing = batch_size - len(unique_idxs)
    if num_missing > 0:
        all_idxs = set(range(len(base_dataset.data_pairs)))
        seen_set_idxs = {
            idx for idx, (img_p, _, _) in enumerate(base_dataset.data_pairs)
            if img_p in seen
        }
        candidates = list(all_idxs - seen_set_idxs)
        extra = random.sample(candidates, num_missing)
        for idx in extra:
            img_p, pos_cap, neg_cap = base_dataset.data_pairs[idx]

            img = Image.open(img_p).convert("RGB")
            if base_dataset.transform:
                img = base_dataset.transform(img)
            pos_tok = base_dataset.tokenizer([pos_cap])[0]
            neg_tok = base_dataset.tokenizer([neg_cap])[0]

            unique_pix.append(img)
            unique_pos.append(pos_tok)
            unique_neg.append(neg_tok)

    images = torch.stack(unique_pix).to(device)
    pos_ts = torch.stack(unique_pos).to(device)
    neg_ts = torch.stack(unique_neg).to(device)

    if mode and mode.lower() == 'standard':
        return images, pos_ts
    else:
        return images, torch.cat([pos_ts, neg_ts], dim=0)


def split_train_val(dataset, val_size=1000, seed=42):
    from torch.utils.data import random_split
    if len(dataset) < val_size:
        raise ValueError("Dataset size is smaller than the desired validation size.")
    
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


class LoadCOCOPair(Dataset):
    def __init__(self, annotations, image_folder, tokenizer, transform, is_test=False):
        self.annotations = annotations
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_test = is_test
        self.data_pairs = []
        self.positive_count = 0
        self.neg_count = 0

        missing_images_count = 0

        for data in annotations:
            image_filename = data.get("image")
            image_path = os.path.join(self.image_folder, image_filename)
            
            if os.path.exists(image_path):
                pos_caption = data.get("text")   
                neg_caption = data.get("neg_text") 
                
                if pos_caption and neg_caption:
                    self.data_pairs.append((image_path, pos_caption, neg_caption))
                    self.positive_count += 1
                    self.neg_count += 1
            else:
                missing_images_count += 1
                logger.warning(f"❗️Missing image: {image_path}")

        logger.info(f"✅ Finished creating pairs. Total pairs: {len(self.data_pairs)}. Missing images: {missing_images_count}.")
        logger.info(f"📊 Total Pos samples: {self.positive_count}")
        logger.info(f"📊 Total COCO NEG samples: {self.neg_count}")

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
            "pixel_values": image,    # preprocessed image tensor
            "pos_text": pos_tokens,   # tokenized positive caption
            "neg_text": neg_tokens    # tokenized negative caption
        }


def get_dataset_COCO(json_path, image_folder_path, tokenizer, transform, batch_size, subset_size=None):
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    ds = LoadCOCOPair(
        annotations=annotations,
        image_folder=image_folder_path,
        tokenizer=tokenizer,
        transform=transform,
        is_test=True  
    )
    
    if subset_size is not None and len(ds) > subset_size:
        indices = random.sample(range(len(ds)), subset_size)
        ds = Subset(ds, indices)
    
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False
    )
    
    logging.info(f"COCO Dataset loaded with {len(ds)} samples.")
    return loader, ds

class TestTypeDataset(Dataset):
    def __init__(self, json_path, image_folder, tokenizer, transform):
        self.samples = []
        self.tokenizer = tokenizer
        self.transform = transform

        with open(json_path, 'r') as f:
            data = json.load(f)

        for image_id, caption_dict in data.items():
            img_path = os.path.join(image_folder, image_id)
            if not os.path.exists(img_path):
                continue

            sorted_items = sorted(caption_dict.items(), key=lambda x: int(x[0]))
            for i in range(0, len(sorted_items) - 1, 2):
                a_idx, a_data = sorted_items[i]
                b_idx, b_data = sorted_items[i + 1]

                if a_data["label"] == 1 and b_data["label"] == 0:
                    pos_caption = a_data["caption"]
                    neg_caption = b_data["caption"]
                    pair_type = b_data.get("type", "unknown")
                elif a_data["label"] == 0 and b_data["label"] == 1:
                    pos_caption = b_data["caption"]
                    neg_caption = a_data["caption"]
                    pair_type = a_data.get("type", "unknown")
                else:
                    continue 

                self.samples.append({
                    "image_path": img_path,
                    "pos_caption": pos_caption,
                    "neg_caption": neg_caption,
                    "type": pair_type,
                    "image_id": image_id
                })

        print(f"✅ Loaded {len(self.samples)} valid caption pairs from JSON.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        pos_text = self.tokenizer([sample["pos_caption"]])[0]
        neg_text = self.tokenizer([sample["neg_caption"]])[0]

        return {
            "image_path": sample["image_path"],
            "pixel_values": image,
            "pos_text": pos_text,
            "neg_text": neg_text,
            "type": sample["type"]
        }

class TypeTestDatasetWrapper:
        def __init__(self, dataset):
            self.dataset = dataset
            self.type_to_indices = defaultdict(list)

            for idx in range(len(dataset)):
                item = dataset[idx]
                sample_type = item.get("type", "unknown")
                self.type_to_indices[sample_type].append(idx)

            self.types = list(self.type_to_indices.keys())

        def get_loader_for_type(self, type_name, batch_size=32, shuffle=False):
            indices = self.type_to_indices.get(type_name, [])
            subset = Subset(self.dataset, indices)
            return DataLoader(subset, batch_size=batch_size, shuffle=shuffle), len(subset)

        def get_all_types(self):
            return self.types

        def get_all_loaders(self, batch_size=32, shuffle=False):
            return {
                t: self.get_loader_for_type(t, batch_size, shuffle)
                for t in self.types
            }
